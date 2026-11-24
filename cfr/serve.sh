#!/usr/bin/env bash
# Start the engine service on the box (pm2 runs this; see pm2.config.cjs).
# Serves the run named in $SIXMAX_HOME/current_run (override with
# SIXMAX_RUN) and the built web app in $SIXMAX_HOME/web, on port 8791
# (override with SIXMAX_PORT).
#
# Serve another run, or pick up newly exported depths:
#   echo RUN > $SIXMAX_HOME/current_run && pm2 restart sixmax-engine
set -euo pipefail
cd "$(dirname "$0")"
source ./box_env.sh
RUN="${SIXMAX_RUN:-$(cat "$SIXMAX_HOME/current_run")}"
exec ./target/release/sixmax serve --run "$RUN" --port "${SIXMAX_PORT:-8791}"
