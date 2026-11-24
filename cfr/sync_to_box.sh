#!/usr/bin/env bash
# Copy this working tree (including uncommitted changes) to the box:
#   asus-nvidia:~/Workspace/6-max-poker-ai-engine/repo/
#
#   cfr/sync_to_box.sh
#
# Build output, node_modules and .git stay put, so the box keeps its compiled
# crates between syncs. Parity fixtures (cfr/fixtures/, gitignored) are synced.
set -euo pipefail
cd "$(dirname "$0")/.."
ssh asus-nvidia 'mkdir -p Workspace/6-max-poker-ai-engine/repo'
rsync -a --delete \
  --exclude .git/ \
  --exclude node_modules/ \
  --exclude dist/ \
  --exclude .playwright-mcp/ \
  --exclude 'cfr/target/' \
  --exclude '*.tsbuildinfo' \
  ./ asus-nvidia:Workspace/6-max-poker-ai-engine/repo/
