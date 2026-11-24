# CFR engine

The bots' engine: a precomputed CFR strategy (a static "blueprint") for
6-max no-limit hold'em, trained and served on the box (`asus-nvidia`).
Everything here is Rust and is built and run on the box; this machine only
edits code, dumps the TS parity fixtures and builds the web app. It follows
the heads-up project's engine (`~/Code/heads-up-poker-ai-engine/cfr`),
generalized to six seats, without live solving.

How it plays: for each decision the engine replays the hand through the
rules port, picks the tree for the table size (the 6-handed ladder, or the
heads-up ladder once two players are left) and the stack depth nearest the
bot's effective stack, maps the real betting onto the abstract tree
(off-menu sizes via the pseudo-harmonic mapping), looks up the bot's card
group, and samples from the strategy there, turning the chosen pot fraction
back into legal chips.

## Layout

```
poker_core/       the game: a port of src/engine/game.ts (rules), cards, a 7-card
                  hand evaluator, equity and all-in EV; parity tests against TS
sixmax_cfr/       the engine:
  abs/, classes.rs  card groups: 169 preflop classes, k-means groups per street
                    (copied from the heads-up engine)
  menu.rs, tree.rs  the bet menu and the abstract trees (6 seats or heads-up),
                    built by driving the rules port
  deal.rs           dealing: every seat's card groups and showdown strength
  mccfr.rs          external-sampling MCCFR (Pluribus-style schedule)
  strategy.rs       exported strategies (1 byte per action, memory-mapped)
  engine.rs         play time: replay, seat mapping, depth choice, translation,
                    the equity fallback
  eval.rs           fast matches on the abstract game; scripted bots
  toy.rs            validation games: Kuhn (2 and 3 players), Leduc
sixmax/           the `sixmax` command line: abstraction, abscheck, tree, train,
                  export, eval, probe, match, serve
scripts/          TS run with tsx: rule scenario checks, parity fixture dumper
fixtures/         parity fixtures dumped from the TS engine (gitignored)
box_env.sh        environment for the box (sourced by everything below)
box.sh            run a command on the box inside repo/cfr
box_bg.sh         start a long job on the box, detached, logging to $SIXMAX_HOME/logs
sync_to_box.sh    rsync this working tree to the box
train_all.sh      train every table size and depth of a run, one after another
serve.sh, pm2.config.cjs   the engine service under pm2 (port 8791)
deploy_web.sh     build the web app and copy it to the box
```

## The box

Everything for this project on the box lives under
`~/Workspace/6-max-poker-ai-engine/` (`$SIXMAX_HOME`): `repo/` (this repo),
`tmp/`, `data/` (card-group tables), `runs/` (training checkpoints),
`strategies/` (exported strategies), `logs/`, and `web/` (the built app).
Cargo and the Rust toolchain stay in `~/.cargo` and `~/.rustup`.

```sh
cfr/sync_to_box.sh                          # copy the working tree to the box
cfr/box.sh cargo build --release            # build there
cfr/box.sh cargo test --release             # parity, tree, solver and toy-game tests
```

## Pipeline

```sh
# 1. card groups: 169 preflop classes, 200 groups per street after (2.6 min)
cfr/box_bg.sh abs-g200 ./target/release/sixmax abstraction --name g200 --buckets 169,200,200,200
cfr/box.sh ./target/release/sixmax abscheck --name g200
# 2. train every table size and depth of a run (about 6-7 days)
cfr/box_bg.sh train-v1 ./train_all.sh v1 g200
cfr/box.sh tail -n 3 '$SIXMAX_HOME/logs/train-v1.log'
# 3. check it
cfr/box.sh ./target/release/sixmax probe --run v1 --players 6 --stack 200        # opening charts
cfr/box.sh ./target/release/sixmax eval --players 6 --stack 200 --strat V=v1 --vs-bots V
cfr/box.sh ./target/release/sixmax match --run v1 --vs random --hands 20000      # real rules
# 4. serve it (restart to pick up newly exported depths)
cfr/box.sh 'echo v1 > $SIXMAX_HOME/current_run && pm2 restart sixmax-engine'
```

The service is started once with `pm2 start cfr/pm2.config.cjs && pm2 save`
from `repo/`; `cfr/deploy_web.sh` updates the web app it serves
(http://192.168.0.185:8791). `GET /api/health` shows what's loaded.

Logs of play in the app, in `$SIXMAX_HOME/logs/`: `engine.log` (one text
line per decision), `decisions.jsonl` (every decision: the app's request and
the engine's debug output) and `hands.jsonl` (every finished hand, sent by
the app).

## The abstract game

- **Stack depths:** every table size is trained with equal stacks at
  100/70/50/35/25/12/6bb (200/140/100/70/50/24/12 chips). The engine uses
  the depth nearest (in ratio) the bot's stack against the deepest other
  player still in the hand. Deeper stacks play the 100bb strategy (a 150bb
  depth is in `notes.txt`).
- **Table sizes:** 5-, 4- and 3-handed play are the 6-handed tree with the
  early positions (UTG, then MP, then CO) folded: same blinds, same order.
  Heads-up has its own ladder (the button posts the small blind).
- **Bet menu** (fractions of the pot after calling; `menu.rs`): preflop opens
  of 0.6 and 1.0 (about 2.5 and 3.5bb from UTG), one 3-bet and one 4-bet size
  (1.0), all-in; heads-up pots after the flop bet 33%, 75% or 150% and raise
  pot; pots of three or more bet 50% and raise pot, at most two bets and
  raises a street. Fold, check and call are always offered.
- **Card groups (`g200`):** the 169 preflop classes; after the flop 200
  groups per street, by k-means on hand-strength distributions (flop and
  turn) and on equity against groups of opponent hands (river).
- **Sizes at 100bb:** 37.8M decision nodes, 16.35B (information set,
  action) entries: 61 GiB of 4-byte regrets.

## Training

External-sampling MCCFR (`mccfr.rs`), with Pluribus's recipe:

- Linear-CFR discounting every 0.5% of the run for its first 20%, then
  negative-regret pruning on 95% of iterations (the heads-up engine's
  thresholds).
- 4-byte regrets (hundredths of a chip). The average strategy is kept for
  preflop only, updated on 1 iteration in 32 (adding to the hottest
  preflop sums from 20 threads on every iteration halved the speed).
  After the flop the bot plays the final current strategy, which beat
  Pluribus-style snapshot averaging by 1.2 bb/100 in a 25bb test.
- Each depth gets about 6,000 regret updates per entry, the heads-up v1c
  level: a short warm-up measures the updates per traversal, which sets the
  number of iterations.
- Checkpoints every 2 hours (zstd), resumable; `train_all.sh` skips depths
  already exported.

## Checks

- `cargo test --release -p poker_core`: the rules port replays 300 random
  matches recorded from `game.ts` exactly, and the evaluator orders 20,000
  hands like `evaluator.ts` (`npx tsx cfr/scripts/dump_fixtures.ts`, then
  sync).
- `cargo test --release -p sixmax_cfr`: trees agree with the rules engine on
  random walks, payouts are zero-sum, short tables are subtrees; the solver
  reaches Kuhn's equilibrium (exploitability 0.0002, value -1/18), Leduc's
  (0.003, -0.0861) and an equilibrium of 3-player Kuhn (best-response gains
  0.0003); checkpoints round-trip.
- `sixmax eval`: strategies against tables of scripted bots on the abstract
  game. An 8-minute 25bb run beat random, station, maniac, min-raiser,
  shover and nit tables by +22 to +157 bb/100.
- `sixmax match`: the play-time engine under the real rules, against scripted
  bots or itself at random uneven stacks; every decision checked legal, the
  fallback rate reported. With short dev strategies: +125/+154/+87 bb/100
  against random/station/maniac, 0-2% fallback.

## Results

Filled in when the v1 run finishes.
