# Source this on asus-nvidia before building or running anything.
#
# Everything this project puts on the box lives under one folder,
# $SIXMAX_HOME:
#
#   repo/          this repository (synced by cfr/sync_to_box.sh)
#   tmp/           scratch space (TMPDIR)
#   data/          card-group (abstraction) tables
#   runs/          training runs: checkpoints and logs
#   strategies/    exported strategies the engine serves
#   logs/          job and engine service logs
#   web/           the built web app the engine serves
#
# The Rust toolchain and Cargo's cache stay in ~/.rustup and ~/.cargo.

export SIXMAX_HOME="$HOME/Workspace/6-max-poker-ai-engine"
export TMPDIR="$SIXMAX_HOME/tmp"
export PATH="$HOME/.cargo/bin:$PATH"
mkdir -p "$TMPDIR" "$SIXMAX_HOME/data" "$SIXMAX_HOME/runs" \
  "$SIXMAX_HOME/strategies" "$SIXMAX_HOME/logs" "$SIXMAX_HOME/web"
