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

KEYWORD_BPE = "▁HE Y ▁SP O T"  # "HEY SPOT" BPE-encoded (fallback without --phrases-file)
KWS_SCORE = 1.5
KWS_THRESHOLD = 0.05


def build_keyword_lines(phrases, tokenize, score=KWS_SCORE, threshold=KWS_THRESHOLD):
    """One keywords.txt line per phrase: '<bpe tokens> :<score> #<thr> @<SLUG>'.

    tokenize(phrase) -> list of BPE token strings.
    """
    lines = []
    for phrase in phrases:
        slug = "_".join(phrase.upper().split())
        lines.append(f"{' '.join(tokenize(phrase))} :{score} #{threshold} @{slug}")
    return lines


def _bpe_tokenizer():
    import sherpa_onnx
    def tokenize(phrase):
        return sherpa_onnx.text2token(
            [phrase], tokens=str(MODEL_DIR / "tokens.txt"),
            tokens_type="bpe", bpe_model=str(MODEL_DIR / "bpe.model"))[0]
    return tokenize


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


def setup(phrases_file=None):
    """Download model files and create keywords.txt."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Download missing files
    for f in REQUIRED_FILES:
        dest = MODEL_DIR / f
        if dest.exists():
            print(f"  {f} — already exists")
            continue
        download_file(f"{BASE_URL}/{f}", dest)

    kw_path = MODEL_DIR / "keywords.txt"
    if phrases_file:
        sys.path.insert(0, str(PROJECT_ROOT))
        from src.voice_control.wake.phrases import load_wake_phrases
        phrases = load_wake_phrases(phrases_file)
        kw_lines = build_keyword_lines(phrases, _bpe_tokenizer())
        kw_path.write_text("\n".join(kw_lines) + "\n")
        print(f"  keywords.txt — created ({len(kw_lines)} phrases: {', '.join(phrases)})")
    else:
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
    parser.add_argument("--phrases-file", default=None,
                        help="Wake phrase list, one per line (e.g. config/wake_phrases.txt)")
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

    if setup(args.phrases_file):
        print("\nDone! Test with: python src/voice_control/wake/sherpa_onnx.py --test-mic")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
