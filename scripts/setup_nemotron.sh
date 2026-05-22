#!/bin/bash
# scripts/setup_nemotron.sh
set -euo pipefail

NEMO_DIR=/mnt/ssd/nemotron-models
NEMO_ARCHIVE=sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25
NEMO_URL=https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/${NEMO_ARCHIVE}.tar.bz2

mkdir -p "$NEMO_DIR"
cd "$NEMO_DIR"

if [ -d "$NEMO_ARCHIVE" ]; then
  echo "[skip] already extracted at $NEMO_DIR/$NEMO_ARCHIVE"
  exit 0
fi

wget -nc "$NEMO_URL"
tar xvf "${NEMO_ARCHIVE}.tar.bz2"
rm "${NEMO_ARCHIVE}.tar.bz2"

ls -lh "$NEMO_DIR/$NEMO_ARCHIVE"
