#!/bin/bash
# scripts/training/train_wake_hey_spot.sh — runs the livekit-wakeword pipeline end-to-end.
# Prerequisites: ./scripts/training/setup_livekit_wakeword.sh has been run and the
# ~20 GB training data is present under /mnt/ssd/livekit-wakeword-data/.
# Training takes ~2-4 h on AGX Orin (contends with llama-server VRAM).
# Consider cloud GPU + scp the resulting ONNX to the Jetson if Orin is busy.
set -euo pipefail

CONFIG=configs/wake/wakeword_hey_spot.yaml
WAKE_OUT=/mnt/ssd/wake-models/hey_spot.onnx

mkdir -p "$(dirname "$WAKE_OUT")"

spot-env/bin/livekit-wakeword run "$CONFIG"

EXPORT_DIR=$(grep -E '^output_dir' "$CONFIG" 2>/dev/null | awk -F'"' '{print $2}' || echo "output")
SRC=$(find "$EXPORT_DIR" -name "hey_spot*.onnx" -print -quit)
[ -n "$SRC" ] && cp "$SRC" "$WAKE_OUT"
ls -lh "$WAKE_OUT"
