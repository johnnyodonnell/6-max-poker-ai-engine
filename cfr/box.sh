#!/usr/bin/env bash
# Run a command on the box inside repo/cfr, with the project environment
# (cfr/box_env.sh) loaded. Sync first with cfr/sync_to_box.sh.
#
#   cfr/box.sh cargo test --release -p poker_core
#   cfr/box.sh ./target/release/sixmax --help
set -euo pipefail
if [ $# -eq 0 ]; then
  echo "usage: cfr/box.sh <command...>" >&2
  exit 2
fi
ssh asus-nvidia "cd Workspace/6-max-poker-ai-engine/repo/cfr && source ./box_env.sh && $*"
