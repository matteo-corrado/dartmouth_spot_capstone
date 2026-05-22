#!/usr/bin/env python3
"""Download Gemma 4 E4B GGUF + mmproj to /mnt/ssd/llamacpp-models/.

Repos verified May 2026:
  ggml-org/gemma-4-E4B-it-GGUF       → Q8_0 (~8 GB) — canonical source (llama.cpp team).
                                     ggml-org ships Q4_K_M / Q8_0 / bf16; no Q6_K available.
                                     Q8_0 chosen for max trust; throughput verified in T4 step 7.
                                     → mmproj-gemma-4-E4B-it-Q8_0.gguf (~1 GB) — Q8 projector
                                     matching weights precision tier.
"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

MODELS_DIR = Path("/mnt/ssd/llamacpp-models")

MODELS = [
    {
        # Q8_0: ~8 GB, perplexity within ~0.05% of BF16 (better than Q6_K's 0.3%).
        # Plan asked for Q6_K but ggml-org only ships Q4_K_M / Q8_0 / bf16.
        # Q8_0 honors plan's "quality bump over Q4_K_M baseline" rationale.
        # T4 step 7 will measure tok/s; swap to lmstudio-community Q6_K if <15 tok/s.
        "repo_id": "ggml-org/gemma-4-E4B-it-GGUF",
        "filename": "gemma-4-E4B-it-Q8_0.gguf",
        "local_name": "gemma-4-E4B-Q8_0.gguf",
    },
    {
        "repo_id": "ggml-org/gemma-4-E4B-it-GGUF",
        "filename": "mmproj-gemma-4-E4B-it-Q8_0.gguf",
        "local_name": "mmproj-gemma4-e4b-Q8_0.gguf",
    },
]


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for m in MODELS:
        target = MODELS_DIR / m["local_name"]
        if target.exists():
            print(f"[skip] {m['local_name']} ({target.stat().st_size / 1e9:.2f} GB)")
            continue
        print(f"[pull] {m['repo_id']}/{m['filename']} -> {target}")
        downloaded = hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            local_dir=str(MODELS_DIR),
        )
        downloaded_path = Path(downloaded)
        if downloaded_path.name != m["local_name"]:
            downloaded_path.rename(target)
        print(f"  done: {target.stat().st_size / 1e9:.2f} GB")

    print("\n=== llamacpp models ===")
    for p in sorted(MODELS_DIR.glob("*.gguf")):
        print(f"  {p.name:50s}  {p.stat().st_size / 1e9:6.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
