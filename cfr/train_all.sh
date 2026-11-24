#!/usr/bin/env bash
# Train every table size and stack depth of a run, one after another (on the
# box; start it with box_bg.sh). Strategies are exported to
# $SIXMAX_HOME/strategies/RUN/<players>p-s<chips>/ as each one finishes, and
# an interrupted run resumes from its checkpoints when started again
# (finished depths are skipped).
#
#   cfr/box_bg.sh train-v1 ./train_all.sh v1 [ABSTRACTION=g200]
#
# Order: the heads-up ladder (minutes), then 6-max from 100bb down to 6bb
# (100bb about 3 days, the rest about 3 more). Each depth gets about 6,000
# regret updates per entry (the heads-up engine's v1c level); its warm-up
# measures the updates per traversal that sets its length.
set -euo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
RUN=$1
ABS=${2:-g200}
STACKS="200 140 100 70 50 24 12"
train() {
  local players=$1 stack=$2 warmup=$3
  if [ -f "$SIXMAX_HOME/strategies/$RUN/${players}p-s${stack}/strategy.u8" ]; then
    echo "=== $RUN: ${players} players, stack $stack already exported; skipping"
    return
  fi
  echo "=== $RUN: ${players} players, stack $stack"
  ./target/release/sixmax train --run "$RUN" --abs "$ABS" --players "$players" --stack "$stack" --calibrate "$warmup"
}
for S in $STACKS; do train 2 "$S" 30; done
for S in $STACKS; do
  case $S in
    200|140) W=240 ;;
    100|70) W=180 ;;
    *) W=60 ;;
  esac
  train 6 "$S" "$W"
done
echo "=== $RUN: all depths done"
