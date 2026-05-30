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


from src.voice_control.client_mic import check_safety_command


def test_detector_labels_map_to_safety_intents():
    # The detector returns the @<id> label; check_safety_command must map each
    # to the correct intent so execute_on_spot fires the right halt.
    assert check_safety_command("stop")["intent"] == "stop"
    assert check_safety_command("freeze")["intent"] == "freeze"
    # estop label "e stop" must match the emergency-stop pattern
    assert check_safety_command("e stop")["intent"] == "estop"


def test_make_safety_detector_importable_and_handles_missing_model(tmp_path):
    # Construction must never raise even if the keyword file is absent;
    # it degrades to is_available() == False (fail-closed handled by caller).
    from src.voice_control.wake.safety_kws import SafetyKeywordDetector
    det = SafetyKeywordDetector(model_dir=tmp_path)  # empty dir -> unavailable
    assert det.is_available() is False
    assert det.process_frame(b"\x00\x00" * 480) == ""
