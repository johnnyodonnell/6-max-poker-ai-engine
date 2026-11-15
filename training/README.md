# 6-max poker — self-play training

Search-free self-play **PPO** for the browser bot, written in Rust. Self-play,
training, evaluation and ONNX export all run in one binary, `poker`, using
`tch`/libtorch on the GPU. The browser (`src/engine/neural.ts`) runs the exported
ONNX with onnxruntime-web: one forward pass per decision, sampling from the policy.

## Layout

```
poker_core/        pure Rust (no torch): rules (port of src/engine/game.ts), 7-card
                   evaluator, NN encoder + action space (mirrors encode.ts /
                   actions.ts), training-table sampler, all-in EV
trainer/           the `poker` binary (tch): nets + ONNX writer, self-play,
                   PPO update, daemon loop, evaluator, probes, export
scripts/           TS run with tsx: fixture dumper, rule checks, ONNX parity,
                   browser-engine e2e
sync_to_box.sh     rsync this working tree to the box
run_daemon.sh      start/resume training on the box
promote.sh         export the top-rated snapshot to public/models/current.onnx
                   and verify it (ONNX parity + browser-engine e2e)
```

## The model

- **Actions:** ten discrete actions: fold, check/call, min-raise, bet/raise to
  33/50/75/100/150/250% of the pot after calling, all-in. A size is offered
  only when it lands strictly between the previous offered size and all-in.
  (Run1's continuous size head collapsed to min-raises.)
- **Input:** raw cards (suit-canonical), this hand's betting history (48
  tokens), and table state, all from the acting player's point of view.
- **Policy net (~3.2M params):**
  - cards enter twice: as embeddings summed per group, and as suit × rank
    planes (hole, flop, turn, river, board, all cards, plus rank and suit
    counts), so pairs, sets, flushes and straights are simple patterns;
  - attention over the history;
  - a residual MLP;
  - a logits head over the ten actions.
- **Critic (~2.0M params, training only):** the same body plus every live
  opponent's hole cards, the mover's equity against them, and the opponent
  count. It outputs Q(s, a) per action and is never exported.
- **Training (`train.rs`):**
  - every seat plays the current policy, sampling at temperature 1;
  - each hand's chip result ÷ 200 is the return for all of that seat's
    decisions, with all-in EV adjustment when the board runs out with no
    decisions left;
  - each table is sampled per hand (40% 6×100bb, otherwise 2–6 players
    splitting 1,200 chips);
  - self-play tables persist across cohorts, so no hand is discarded at a
    cohort boundary;
  - advantages are all-actions with an action-dependent control variate
    (`--cv all`). A sampled residual z − Q(a) is added to Q(b) − V for every
    legal action. Fold's value is exact (the chips already put in, lost). The
    estimate stays unbiased whatever the critic's error;
  - the loss is clipped PPO for 2 epochs of 32k-row minibatches, stopping
    early once KL(behavior‖π) passes 1.5 × 0.02;
  - a forward-KL pull toward the **anchor** policy (`--c-anchor 0.05`) and an
    entropy bonus;
  - AdamW at lr 1e-4 with β₂ 0.99.
- **Anchor:** the anchor starts as the initial policy. The first snapshot that
  significantly beats random replaces it. After that, any snapshot that
  significantly beats the current anchor head to head replaces it.
- **Batch size:** the gradient noise scale of the policy term is measured
  every 4th cohort. The cohort grows toward it, from 131k to 524k decisions,
  so updates stay mostly signal as the policy improves.

## Box setup (asus-nvidia)

Everything lives under `~/Workspace/6-max-poker-ai-engine/`:

```
repo/    this repository (synced with training/sync_to_box.sh)
deps/    venv with torch==2.13.0 (cu130): only a source of libtorch for tch 0.26
runs/    training runs (checkpoints, snapshots, logs)
```

```sh
cd ~/Workspace/6-max-poker-ai-engine
~/.local/bin/uv venv deps/venv --python 3.12
~/.local/bin/uv pip install --python deps/venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu130 torch==2.13.0
cd repo/training && cargo build --release      # LIBTORCH comes from .cargo/config.toml
./target/release/poker gpu-check
```

No Python runs at build or train time. The binary finds libtorch through its rpath.

## Train

```sh
RUN=run2 training/run_daemon.sh            # on the box; resumes if runs/run2 exists
tail -f ~/Workspace/6-max-poker-ai-engine/runs/run2/train.log
```

Each cohort logs one JSON line:
- losses, entropy, KL to the behavior policy and to the anchor, clip fraction;
- epochs and steps taken;
- the critic's loss and explained variance (`critic_ev`);
- the noise scale (`b_noise`) and the next cohort size;
- timings;
- HUD stats: VPIP, PFR, 3-bet, c-bet, WTSD, the mix of the ten actions and
  bet-size quantiles.

Throughput on the GB10 (default cohort of 131k decisions, 4,096 hands in flight)
is about 26 s per cohort:
- ~2.5 s of self-play, about 4,000 hands/s, including the critic's equity
  inputs;
- ~23 s for the update: 10 PPO minibatch steps, one critic pass, and the
  anchor and critic forwards.

That works out to roughly 1.4M hands per hour, with 10 policy steps per
cohort; run1 did 4M hands per hour with 1 step.

## Evaluate and promote

Snapshots start every 30 minutes. An improvement counts as significant when a
snapshot beats the previous one head to head by more than its 95% CI. After
two significant improvements in a row, the interval steps up to 1 hour, then
after two more to 4 hours. The schedule state is kept in `latest.json`.
`--snapshot-every` fixes the interval instead.

Each snapshot is probed (`[probe]` in the log; also `poker probe --weights
W`). The probes check fold rates by preflop equity, how often pocket pairs
are folded, and flop fold rates for sets, top pair and air. Each snapshot is
then evaluated automatically and recorded in `runs/<run>/pool.json`. A match puts the candidate in one seat against five
copies of the random bot or of an earlier snapshot. Each deal is replayed with
the candidate in all six seats, with all-in EV applied. Ratings are
least-squares fits of snapshot-vs-snapshot bb/100, with the first snapshot
pinned at 0. Results against random are reported separately (`vs_random`) and
left out of the fit: a sound, tight policy wins *less* against random maniacs
than a loose one does, so they would distort the ranking.

```sh
poker match --a <snap>.safetensors --b random --deals 2000   # ad hoc, on the box
poker best --run-dir <run>                                   # top-rated snapshot
training/promote.sh            # here: export best -> public/models/current.onnx,
                               # parity + e2e checks; then commit the model
```

## Parity gates

```sh
# here (Node):
npx tsx training/scripts/check_rules.ts          # game.ts rule scenarios
npx tsx training/scripts/dump_fixtures.ts        # -> training/fixtures/ (gitignored)
training/sync_to_box.sh
# on the box:
cargo test --release -p poker_core               # rules, evaluator, encoder parity
poker export --init 7 --out m.onnx --fixture fx.json
# copy both back to training/fixtures/parity_{model.onnx,fx.json}, then here:
npx tsx training/scripts/test_onnx_parity.ts     # onnxruntime-web vs tch
```
