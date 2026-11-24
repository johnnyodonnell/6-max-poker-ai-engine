#!/usr/bin/env bash
# Start a long-running command on the box, fully detached, inside repo/cfr
# with the project environment; output goes to $SIXMAX_HOME/logs/NAME.log.
#
#   cfr/box_bg.sh abs-g200 ./target/release/sixmax abstraction --name g200 ...
#   cfr/box.sh tail -f '$SIXMAX_HOME/logs/abs-g200.log'
set -euo pipefail
if [ $# -lt 2 ]; then
  echo "usage: cfr/box_bg.sh NAME <command...>" >&2
  exit 2
fi
NAME=$1
shift
ssh -n asus-nvidia "cd Workspace/6-max-poker-ai-engine/repo/cfr && source ./box_env.sh && \
  setsid -f bash -c 'exec $* > \$SIXMAX_HOME/logs/$NAME.log 2>&1 < /dev/null' && echo started $NAME"
