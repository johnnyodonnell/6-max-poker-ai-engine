# 6-max poker — self-play training

Search-free **REINFORCE** self-play for the browser bot, written in Rust. Self-play,
training, evaluation and ONNX export all run in one binary, `poker`, using
`tch`/libtorch on the GPU. The browser (`src/engine/neural.ts`) runs the exported
ONNX with onnxruntime-web: one forward pass per decision, sampling from the policy.

## Layout

```
poker_core/        pure Rust (no torch): rules (port of src/engine/game.ts), 7-card
                   evaluator, NN encoder + action space (mirrors encode.ts /
                   actions.ts), training-table sampler, all-in EV
trainer/           the `poker` binary (tch): net + ONNX writer, self-play,
                   REINFORCE step, daemon loop, evaluator, export
scripts/           TS run with tsx: fixture dumper, rule checks, ONNX parity,
                   browser-engine e2e
sync_to_box.sh     rsync this working tree to the box
run_daemon.sh      start/resume training on the box
```

## The model

- **Actions:** fold, check/call, raise, all-in. A raise's size `s = ln(raise
  increment / pot after calling)` is drawn from a normal `(mu, sigma)` truncated
  to the legal range. All-in is its own action.
- **Input:** raw cards (suit-canonical), this hand's betting history (48
  tokens), and table state, all from the acting player's point of view.
- **Net (~3.1M params):** card embeddings, then attention over the history, then a
  residual MLP, with type, size and value heads.
- **Training:**
  - every seat plays the current net, sampling at temperature 1;
  - each hand's chip result ÷ 200 is the reward for all of that seat's
    decisions, with all-in EV adjustment when the board runs out with no
    decisions left;
  - each table is sampled per hand (40% 6×100bb, otherwise 2–6 players
    splitting 1,200 chips);
  - one strictly on-policy AdamW step per cohort, with a value baseline,
    a type-entropy bonus and a log-σ bonus.

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
RUN=run1 training/run_daemon.sh            # on the box; resumes if runs/run1 exists
tail -f ~/Workspace/6-max-poker-ai-engine/runs/run1/train.log
```

Each cohort logs one JSON line: losses, entropy, mean σ, KL, timings, and HUD
stats (VPIP, PFR, 3-bet, c-bet, WTSD, action mix, raise-size quantiles).
`logp_gap` should stay near 0. It compares the sampler's and the trainer's
log-probs, which is the on-policy sanity check.

Throughput on the GB10 (default cohort of 131k decisions, 4,096 hands in flight)
is about 11 s per cohort: ~3 s of self-play (~4,000 hands/s) and ~8 s for the
update. That works out to roughly 4M hands per hour.

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
