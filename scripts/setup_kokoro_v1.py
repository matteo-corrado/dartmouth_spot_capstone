#!/usr/bin/env python3
"""Download Kokoro v1.0 TTS model files for kokoro-onnx GPU pipeline.

Downloads two files from the kokoro-onnx GitHub releases:
    kokoro-v1.0.fp16-gpu.onnx  (~170MB, FP16 model tuned for CUDA)
    voices-v1.0.bin            (~27MB,  54 voice embeddings)

Target directory: <project_root>/models/tts/kokoro-v1.0/

Why fp16-gpu and not the full fp32 model?
    The fp16 weights are half the size, fit comfortably in disk and VRAM,
    and run faster on the Jetson AGX Orin's CUDA Execution Provider with
    no audible quality loss for our voice (af_sarah).

Usage:
    python scripts/setup_kokoro_v1.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

RELEASE_BASE = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
MODEL_FILE = "kokoro-v1.0.fp16-gpu.onnx"
VOICES_FILE = "voices-v1.0.bin"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TTS_DIR = PROJECT_ROOT / "models" / "tts"
MODEL_DIR = TTS_DIR / "kokoro-v1.0"

# Approximate sizes for sanity-check after download
EXPECTED_MIN_SIZES = {
    MODEL_FILE: 150 * 1024 * 1024,  # ~170 MB
    VOICES_FILE: 20 * 1024 * 1024,  # ~27 MB
}


def _download(url: str, dest: Path) -> bool:
    print(f"  Downloading {dest.name}\n    from {url}")
    result = subprocess.run(
        ["wget", "-q", "--show-progress", "-O", str(dest), url]
    )
    if result.returncode != 0:
        print(f"  ERROR: wget exited with {result.returncode}")
        if dest.exists():
            dest.unlink()
        return False
    return True


def main() -> int:
    print("=" * 50)
    print("Kokoro v1.0 TTS Model Setup (kokoro-onnx + GPU)")
    print("=" * 50)
    print(f"\n  Target: {MODEL_DIR}\n")

    if not shutil.which("wget"):
        print("  ERROR: wget not found. Install with: sudo apt install wget")
        return 1

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Disk space check (need ~250 MB headroom)
    _, _, free = shutil.disk_usage("/")
    free_mb = free / (1024 * 1024)
    print(f"  Disk free: {free_mb:.0f} MB")
    if free_mb < 300:
        print("  WARNING: Low disk space! Need ~250 MB. Free up space first.")
        response = input("  Continue anyway? [y/N] ").strip().lower()
        if response != "y":
            return 1

    targets = [
        (MODEL_FILE, MODEL_DIR / MODEL_FILE),
        (VOICES_FILE, MODEL_DIR / VOICES_FILE),
    ]

    for name, path in targets:
        if path.exists() and path.stat().st_size >= EXPECTED_MIN_SIZES[name]:
            mb = path.stat().st_size / (1024 * 1024)
            print(f"  {name}: already present ({mb:.0f} MB) — skipping")
            continue
        if path.exists():
            print(f"  {name}: present but too small, re-downloading")
            path.unlink()
        url = f"{RELEASE_BASE}/{name}"
        if not _download(url, path):
            print(f"\n  Try manual download:")
            print(f"    wget -O {path} {url}")
            return 1

    # Verify
    print("\n  Verifying...")
    for name, path in targets:
        if not path.exists():
            print(f"  ERROR: missing {name}")
            return 1
        size = path.stat().st_size
        if size < EXPECTED_MIN_SIZES[name]:
            print(f"  ERROR: {name} is only {size / 1024 / 1024:.1f} MB "
                  f"(expected ≥ {EXPECTED_MIN_SIZES[name] / 1024 / 1024:.0f} MB)")
            return 1
        print(f"  {name}: {size / 1024 / 1024:.0f} MB OK")

    print("\n  Ready! Test with:")
    print("    python src/voice_control/spot_tts.py 'Hello from Spot.'")
    print("\n  Note: requires kokoro-onnx and onnxruntime-gpu installed.")
    print("        See requirements.txt for the install commands.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
