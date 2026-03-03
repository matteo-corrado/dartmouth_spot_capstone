#!/bin/bash
# cleanup_local_inference.sh — Remove Riva and Ollama from Jetson
#
# This frees ~30-40 GB of disk and ~14 GB of VRAM.
# Run once after migrating to Dartmouth cloud APIs.
#
# Usage:
#   chmod +x scripts/cleanup_local_inference.sh
#   ./scripts/cleanup_local_inference.sh
set -e

echo "=== Cleaning up local inference stack ==="
echo ""

# 1. Stop and remove Riva containers + images
echo "--- Removing Riva Docker containers and images ---"
docker stop riva-speech 2>/dev/null || true
docker rm riva-speech 2>/dev/null || true
docker rmi nvcr.io/nvidia/riva/riva-speech:2.18.0-l4t-aarch64 2>/dev/null || true
docker rmi nvcr.io/nvidia/riva/riva-speech:2.18.0-servicemaker-l4t-aarch64 2>/dev/null || true
docker volume prune -f 2>/dev/null || true
rm -rf ~/riva_quickstart

# 2. Remove Ollama and all its models
echo "--- Removing Ollama ---"
ollama rm qwen2.5:7b 2>/dev/null || true
ollama rm qwen2.5vl:7b 2>/dev/null || true
sudo systemctl stop ollama 2>/dev/null || true
sudo systemctl disable ollama 2>/dev/null || true
sudo apt remove -y ollama 2>/dev/null || sudo rm -f /usr/local/bin/ollama
rm -rf ~/.ollama

# 3. Docker layer cleanup
echo "--- Pruning Docker ---"
docker system prune -af 2>/dev/null || true

echo ""
echo "=== Done ==="
echo "Estimated space freed: ~30-40 GB"
echo "Remaining local models: Kokoro TTS (~340MB), KWS (~5MB), YOLO (~35MB)"
echo ""
df -h /
