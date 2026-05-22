#!/bin/bash
# scripts/setup_livekit_wakeword.sh
# Clones livekit-wakeword + installs into spot-env + pre-downloads training data.
# Step 4 (training) is run separately via scripts/train_wake_hey_spot.sh.
set -euo pipefail

LKWW_ROOT=/mnt/ssd/livekit-wakeword
DATA_DIR=/mnt/ssd/livekit-wakeword-data
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ ! -d "$LKWW_ROOT" ]; then
  git clone https://github.com/livekit/livekit-wakeword "$LKWW_ROOT"
fi

cd "$LKWW_ROOT"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

"$REPO_ROOT/spot-env/bin/pip" install --no-deps -e "$LKWW_ROOT"

mkdir -p "$DATA_DIR"
"$REPO_ROOT/spot-env/bin/livekit-wakeword" setup --config "$REPO_ROOT/configs/wake/wakeword_hey_spot.yaml"

ls -lh "$DATA_DIR"
