#!/usr/bin/env python3
"""Download the WeSpeaker ResNet34-LM speaker-embedding ONNX model (Task 18).

    Wespeaker/wespeaker-voxceleb-resnet34-LM -> voxceleb_resnet34_LM.onnx (~26 MB)

Target: <project_root>/models/speaker/voxceleb_resnet34_LM.onnx
Used by src/voice_control/speaker_embed.py (SpeakerEmbedder).

Usage:
    python scripts/setup_wespeaker.py
"""
import sys
from pathlib import Path

REPO_ID = "Wespeaker/wespeaker-voxceleb-resnet34-LM"
FILENAME = "voxceleb_resnet34_LM.onnx"
MIN_SIZE = 20 * 1024 * 1024  # ~26 MB expected

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "speaker"
MODEL_PATH = MODEL_DIR / FILENAME


def main() -> int:
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_SIZE:
        print(f"[wespeaker] already present: {MODEL_PATH}")
        return 0

    from huggingface_hub import hf_hub_download

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[wespeaker] downloading {REPO_ID}/{FILENAME} -> {MODEL_DIR}")
    path = Path(hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=str(MODEL_DIR)))
    size = path.stat().st_size
    if size < MIN_SIZE:
        print(f"[wespeaker] ERROR: {path} is only {size} bytes")
        return 1
    print(f"[wespeaker] ok: {path} ({size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
