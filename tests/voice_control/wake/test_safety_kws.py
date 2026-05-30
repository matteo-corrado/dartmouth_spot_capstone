# tests/voice_control/wake/test_safety_kws.py
"""Tests for the always-on safety KWS (Stage 2F C1)."""
import pathlib
import pytest

from scripts.setup_safety_kws import build_keyword_lines, SAFETY_WORDS

BPE = pathlib.Path("models/kws/bpe.model")


@pytest.mark.skipif(not BPE.exists(), reason="KWS model not set up (run scripts/setup_kws.py)")
def test_build_keyword_lines_covers_all_safety_words():
    lines = build_keyword_lines(str(BPE))
    assert len(lines) == len(SAFETY_WORDS)
    # Each line ends with its @custom-id so detection returns a clean label.
    ids = {ln.split("@")[-1].strip() for ln in lines}
    assert ids == {"stop", "freeze", "estop"}


@pytest.mark.skipif(not BPE.exists(), reason="KWS model not set up")
def test_keyword_lines_are_bpe_tokens_not_raw_text():
    lines = build_keyword_lines(str(BPE))
    # BPE tokenization prefixes word-starts with the sentencepiece meta symbol.
    assert any("▁" in ln for ln in lines)  # ▁
