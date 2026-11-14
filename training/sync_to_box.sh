#!/usr/bin/env bash
# Copy this working tree (including uncommitted changes) to the training box's
# checkout: asus-nvidia:~/Workspace/6-max-poker-ai-engine/repo/
#
#   training/sync_to_box.sh
#
# Build output, node_modules, and the box's own .git are left alone, so the box
# keeps its compiled crates between syncs.
set -euo pipefail
cd "$(dirname "$0")/.."
rsync -a --delete \
  --exclude .git/ \
  --exclude node_modules/ \
  --exclude dist/ \
  --exclude .playwright-mcp/ \
  --exclude 'training/target/' \
  --exclude '*.tsbuildinfo' \
  ./ asus-nvidia:Workspace/6-max-poker-ai-engine/repo/
