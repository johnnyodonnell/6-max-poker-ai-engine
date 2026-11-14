#!/usr/bin/env bash
# Start (or resume) the self-play training daemon on the box, detached.
#
#   RUN=run1 training/run_daemon.sh [extra `poker train` flags...]
#
# Output goes to ~/Workspace/6-max-poker-ai-engine/runs/$RUN/ (checkpoints,
# snapshots/, pool.json, train.log). Re-running resumes from latest.*.
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
W="$(cd "$DIR/../.." && pwd)" # ~/Workspace/6-max-poker-ai-engine
RUN="${RUN:-run1}"
OUT="$W/runs/$RUN"
mkdir -p "$OUT"
cd "$DIR"
nohup ./target/release/poker train --run-dir "$OUT" "$@" >>"$OUT/train.log" 2>&1 &
echo "started pid $! -> $OUT/train.log"
