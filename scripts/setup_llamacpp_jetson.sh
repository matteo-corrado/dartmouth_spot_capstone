#!/bin/bash
# scripts/setup_llamacpp_jetson.sh — pull and smoke-test NVIDIA Jetson llama.cpp container
set -euo pipefail

IMAGE="ghcr.io/nvidia-ai-iot/llama_cpp:latest-jetson-orin"

echo "=== pulling $IMAGE (one-time, ~2 GB) ==="
docker pull "$IMAGE"

echo "=== container smoke test ==="
docker run --rm --runtime=nvidia "$IMAGE" llama-server --help 2>&1 | head -10 || true

echo "=== check NVIDIA runtime ==="
docker run --rm --runtime=nvidia --gpus all "$IMAGE" nvidia-smi || {
  echo "WARNING: nvidia-smi did not run inside container; verify NVIDIA Container Toolkit is installed"
}

echo "=== done ==="
echo "Image cached. T4 will run the daemon."
