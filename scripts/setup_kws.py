#!/usr/bin/env python3
"""Download sherpa-onnx keyword spotting model for 'Hey Spot' wake word.

Model: sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01
  - English keyword spotter trained on GigaSpeech XL
  - int8 quantized (~5MB total)
  - Runs on CPU via sherpa-onnx (no onnxruntime needed)

Usage:
    python scripts/setup_kws.py          # download + setup
    python scripts/setup_kws.py --check  # verify existing install
"""

import os
import sys
import pathlib
import tarfile
import shutil

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "kws"
MODEL_NAME = "sherpa-onnx-kws-zipformer-gigaspeech-3.3M-2024-01-01"
BASE_URL = f"https://huggingface.co/csukuangfj/{MODEL_NAME}/resolve/main"

REQUIRED_FILES = [
    "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx",
    "tokens.txt",
    "bpe.model",
]

KEYWORD_BPE = "▁HE Y ▁SP O T"  # "HEY SPOT" BPE-encoded


def check_install():
    """Check if all required files exist."""
    missing = []
    for f in REQUIRED_FILES:
        if not (MODEL_DIR / f).exists():
            missing.append(f)
    if not (MODEL_DIR / "keywords.txt").exists():
        missing.append("keywords.txt")
    return missing


def download_file(url, dest):
    """Download a file with progress."""
    import urllib.request
    print(f"  Downloading {dest.name}...", end=" ", flush=True)
    urllib.request.urlretrieve(url, dest)
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"({size_mb:.1f} MB)")


def setup():
    """Download model files and create keywords.txt."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download missing files
    for f in REQUIRED_FILES:
        dest = MODEL_DIR / f
        if dest.exists():
            print(f"  {f} — already exists")
            continue
        download_file(f"{BASE_URL}/{f}", dest)

    # Create keywords.txt with "HEY SPOT"
    kw_path = MODEL_DIR / "keywords.txt"
    kw_path.write_text(KEYWORD_BPE + "\n")
    print(f"  keywords.txt — created ({KEYWORD_BPE})")

    # Verify
    missing = check_install()
    if missing:
        print(f"\nERROR: Still missing: {', '.join(missing)}")
        return False

    total_size = sum((MODEL_DIR / f).stat().st_size for f in os.listdir(MODEL_DIR)
                     if (MODEL_DIR / f).is_file())
    print(f"\nKWS model ready at: {MODEL_DIR}")
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Setup sherpa-onnx KWS model")
    parser.add_argument("--check", action="store_true", help="Check existing install")
    args = parser.parse_args()

    if args.check:
        missing = check_install()
        if missing:
            print(f"Missing files: {', '.join(missing)}")
            print("Run: python scripts/setup_kws.py")
            sys.exit(1)
        else:
            print(f"KWS model OK at: {MODEL_DIR}")
            sys.exit(0)

    print(f"Setting up sherpa-onnx KWS model...")
    print(f"Model: {MODEL_NAME}")
    print(f"Target: {MODEL_DIR}\n")

    if setup():
        print("\nDone! Test with: python src/voice_control/wake_word.py --test-mic")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
