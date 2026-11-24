#!/usr/bin/env bash
# Build the web app here and copy it to the box, where the engine serves it
# ($SIXMAX_HOME/web). No restart needed.
set -euo pipefail
cd "$(dirname "$0")/.."
npm run build
rsync -a --delete dist/ asus-nvidia:Workspace/6-max-poker-ai-engine/web/
echo "deployed to asus-nvidia:~/Workspace/6-max-poker-ai-engine/web"
