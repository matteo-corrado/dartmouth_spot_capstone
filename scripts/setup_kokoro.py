#!/usr/bin/env python3
"""Download sherpa-onnx Kokoro TTS model pack for Spot voice control.

Downloads and extracts the kokoro-en-v0_19 model pack (~340MB) which includes:
    model.onnx, voices.bin, tokens.txt, espeak-ng-data/

Uses streaming download+extract to avoid doubling disk usage (important on
Jetson eMMC with limited space).

Usage:
    python scripts/setup_kokoro.py
"""
import sys
import subprocess
import shutil
from pathlib import Path

MODEL_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
MODEL_DIR_NAME = "kokoro-en-v0_19"

# Target: project_root/models/tts/kokoro-en-v0_19/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TTS_DIR = PROJECT_ROOT / "models" / "tts"
MODEL_DIR = TTS_DIR / MODEL_DIR_NAME

REQUIRED_FILES = ["model.onnx", "voices.bin", "tokens.txt"]


def main():
    print("=" * 50)
    print("Kokoro TTS Model Setup (sherpa-onnx)")
    print("=" * 50)
    print(f"\n  Model: {MODEL_DIR_NAME} (English, 11 speakers)")
    print(f"  Target: {MODEL_DIR}\n")

    # Check if already downloaded
    if MODEL_DIR.exists() and all((MODEL_DIR / f).exists() for f in REQUIRED_FILES):
        model_mb = (MODEL_DIR / "model.onnx").stat().st_size / (1024 * 1024)
        print(f"  Model already exists ({model_mb:.0f}MB). Delete to re-download:")
        print(f"    rm -rf {MODEL_DIR}")
        return 0

    # Check disk space
    import shutil as _sh
    total, used, free = _sh.disk_usage("/")
    free_mb = free / (1024 * 1024)
    print(f"  Disk free: {free_mb:.0f}MB")
    if free_mb < 500:
        print(f"  WARNING: Low disk space! Need ~400MB. Free up space first.")
        response = input("  Continue anyway? [y/N] ").strip().lower()
        if response != "y":
            return 1

    # Check wget is available
    if not shutil.which("wget"):
        print("  ERROR: wget not found. Install with: sudo apt install wget")
        return 1

    TTS_DIR.mkdir(parents=True, exist_ok=True)

    # Stream download + extract (avoids saving .tar.bz2 to disk)
    print(f"  Downloading and extracting (~340MB)...")
    print(f"  URL: {MODEL_URL}\n")

    result = subprocess.run(
        f'wget -q --show-progress -O- "{MODEL_URL}" | tar xj -C "{TTS_DIR}"',
        shell=True,
    )

    if result.returncode != 0:
        print(f"\n  ERROR: Download/extract failed (exit code {result.returncode})")
        print(f"  Try manual download:")
        print(f"    wget {MODEL_URL}")
        print(f"    tar xf {MODEL_DIR_NAME}.tar.bz2 -C {TTS_DIR}")
        return 1

    # Verify
    print(f"\n  Verifying...")
    missing = [f for f in REQUIRED_FILES if not (MODEL_DIR / f).exists()]
    if missing:
        print(f"  ERROR: Missing files after extract: {missing}")
        return 1

    model_mb = (MODEL_DIR / "model.onnx").stat().st_size / (1024 * 1024)
    voices_mb = (MODEL_DIR / "voices.bin").stat().st_size / (1024 * 1024)
    print(f"  model.onnx:  {model_mb:.0f}MB")
    print(f"  voices.bin:  {voices_mb:.1f}MB")
    print(f"  tokens.txt:  OK")

    espeak_dir = MODEL_DIR / "espeak-ng-data"
    if espeak_dir.exists():
        print(f"  espeak-ng-data/: OK")
    else:
        print(f"  WARNING: espeak-ng-data/ not found (phonemizer may not work)")

    print(f"\n  Ready! Test with:")
    print(f"    python src/voice_control/spot_tts.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
