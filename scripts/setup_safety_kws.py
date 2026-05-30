#!/usr/bin/env python3
"""Generate models/kws/safety_keywords.txt for the always-on safety KWS (C1).

The safety detector (src/voice_control/wake/safety_kws.py) spots stop/freeze/
estop at the frame level in WAKE_WORD state so a cold safety command halts the
robot with no wake required, regardless of SPOT_WAKE_BACKEND.

sherpa-onnx keyword files are BPE-token lines. We tokenize each safety phrase
with the model's own bpe.model (sentencepiece) and append a `@<id>` custom
keyword id so detection returns a clean label that check_safety_command maps
to an intent.
"""
import pathlib

import sentencepiece as spm

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "kws"
BPE_MODEL = MODEL_DIR / "bpe.model"
OUT = MODEL_DIR / "safety_keywords.txt"

# phrase -> custom id (id is what get_result returns; check_safety_command maps it)
SAFETY_WORDS = {
    "STOP": "stop",
    "FREEZE": "freeze",
    "E STOP": "estop",
}


def build_keyword_lines(bpe_model_path: str) -> list[str]:
    """Return one sherpa keyword line per safety phrase: '<bpe tokens> @<id>'."""
    sp = spm.SentencePieceProcessor()
    sp.load(bpe_model_path)
    lines = []
    for phrase, kw_id in SAFETY_WORDS.items():
        pieces = sp.encode_as_pieces(phrase)  # e.g. ['▁S', 'TO', 'P']
        lines.append(f"{' '.join(pieces)} @{kw_id}")
    return lines


def main() -> int:
    if not BPE_MODEL.exists():
        print(f"[safety-kws] missing {BPE_MODEL} — run scripts/setup_kws.py first")
        return 1
    lines = build_keyword_lines(str(BPE_MODEL))
    OUT.write_text("\n".join(lines) + "\n")
    print(f"[safety-kws] wrote {OUT} ({len(lines)} keywords):")
    for ln in lines:
        print(f"  {ln}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
