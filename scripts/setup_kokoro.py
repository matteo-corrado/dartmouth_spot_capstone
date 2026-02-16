#!/usr/bin/env python3
"""Download Kokoro TTS model files for Spot voice control.

Downloads the ONNX model and voice pack to models/tts/.
Run once after setting up the Jetson.

Usage:
    python scripts/setup_kokoro.py              # default (full precision, 310MB)
    python scripts/setup_kokoro.py --fp16       # half precision (169MB)
    python scripts/setup_kokoro.py --int8       # quantized (88MB, smallest)
"""
import argparse
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"

MODELS = {
    "full": ("kokoro-v1.0.onnx", "310MB, highest quality"),
    "fp16": ("kokoro-v1.0.fp16.onnx", "169MB, good quality/size tradeoff"),
    "int8": ("kokoro-v1.0.int8.onnx", "88MB, smallest"),
}
VOICES_FILE = "voices-v1.0.bin"

# Target name that spot_tts.py expects
TARGET_MODEL_NAME = "kokoro-v1.0.onnx"


def download_file(url: str, dest: Path):
    """Download file with progress indicator."""
    print(f"  Downloading: {url}")
    print(f"  To: {dest}")

    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = min(100, downloaded * 100 // total_size)
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            print(f"\r  [{pct:3d}%] {mb:.1f}/{total_mb:.1f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, str(dest), reporthook=progress_hook)
    print()  # newline after progress


def main():
    parser = argparse.ArgumentParser(description="Download Kokoro TTS model files")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--full", action="store_true", default=True,
                       help="Full precision model (310MB, default)")
    group.add_argument("--fp16", action="store_true",
                       help="FP16 quantized (169MB)")
    group.add_argument("--int8", action="store_true",
                       help="INT8 quantized (88MB, smallest)")
    args = parser.parse_args()

    if args.int8:
        variant = "int8"
    elif args.fp16:
        variant = "fp16"
    else:
        variant = "full"

    model_filename, description = MODELS[variant]

    # Target directory: project_root/models/tts/
    project_root = Path(__file__).resolve().parents[1]
    model_dir = project_root / "models" / "tts"
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("Kokoro TTS Model Setup")
    print("=" * 50)
    print(f"\n  Variant: {variant} ({description})")
    print(f"  Target:  {model_dir}\n")

    # Download model
    model_dest = model_dir / TARGET_MODEL_NAME
    if model_dest.exists():
        print(f"  Model already exists: {model_dest}")
        print(f"  Delete it to re-download.")
    else:
        url = f"{BASE_URL}/{model_filename}"
        download_file(url, model_dest)
        print(f"  Model saved as: {model_dest}")

    # Download voices
    voices_dest = model_dir / VOICES_FILE
    if voices_dest.exists():
        print(f"  Voices already exist: {voices_dest}")
    else:
        url = f"{BASE_URL}/{VOICES_FILE}"
        download_file(url, voices_dest)
        print(f"  Voices saved: {voices_dest}")

    # Verify
    print(f"\nVerifying...")
    if model_dest.exists() and voices_dest.exists():
        model_mb = model_dest.stat().st_size / (1024 * 1024)
        voices_mb = voices_dest.stat().st_size / (1024 * 1024)
        print(f"  Model:  {model_mb:.1f} MB")
        print(f"  Voices: {voices_mb:.1f} MB")
        print(f"\nReady! Test with:")
        print(f"  python src/voice_control/spot_tts.py")
    else:
        print("  ERROR: Files missing after download!")
        sys.exit(1)


if __name__ == "__main__":
    main()
