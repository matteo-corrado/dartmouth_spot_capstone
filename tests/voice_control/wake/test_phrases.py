import numpy as np
import pytest

from scripts.setup_kws import build_keyword_lines
from src.voice_control.wake.phrases import (
    load_wake_phrases, peak_safe_gain, strip_wake_phrase,
)

OPERATOR = ["HEY SPOT", "LISTEN UP SPOT", "EXCUSE ME SPOT", "HEY THERE SPOT"]


def test_strip_exact():
    assert strip_wake_phrase("hey spot stand up", ["HEY SPOT"]) == "stand up"


def test_strip_asr_variant():
    assert strip_wake_phrase("hello spotty, what is this", ["HELLO SPOTTY"]) == "what is this"
    assert strip_wake_phrase("hello spotted what is this", ["HELLO SPOTTY"]) == "what is this"


def test_no_strip_mid_sentence():
    assert strip_wake_phrase("is the spot free", ["HEY SPOT"]) == "is the spot free"


@pytest.mark.parametrize("text,expected", [
    ("hey spot, stand up", "stand up"),
    ("Listen up spot, what is this", "what is this"),
    ("excuse me spot stand up", "stand up"),
    ("hey there spot sit", "sit"),
    ("a spot stand up", "stand up"),
    ("excuse me spots, sit", "sit"),
])
def test_operator_phrases(text, expected):
    assert strip_wake_phrase(text, OPERATOR) == expected


@pytest.mark.parametrize("text", [
    "please listen up spot is here",
    "tell him excuse me spot later",
    "the spot is free",
    "hey there buddy",
])
def test_operator_phrases_no_mid_sentence_strip(text):
    assert strip_wake_phrase(text, OPERATOR) == text


def test_peak_safe_gain_caps_loud_frames():
    assert peak_safe_gain(np.full(480, 0.4, np.float32), 0.5) == pytest.approx(1.25)
    assert peak_safe_gain(np.full(480, 0.01, np.float32), 0.5) == 8.0


def test_load_wake_phrases(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("hey spot\n\n# note\nListen up spot\n")
    assert load_wake_phrases(f) == ["HEY SPOT", "LISTEN UP SPOT"]
    assert load_wake_phrases(tmp_path / "missing.txt") == ["HEY SPOT"]


def test_repo_phrase_file_has_operator_phrases():
    from src.voice_control.wake.phrases import WAKE_PHRASES
    assert WAKE_PHRASES == OPERATOR


def test_build_keyword_lines_with_stub_tokenizer():
    lines = build_keyword_lines(["HEY SPOT", "LISTEN UP SPOT"],
                                lambda p: ["▁" + w for w in p.split()],
                                score=1.5, threshold=0.05)
    assert lines == ["▁HEY ▁SPOT :1.5 #0.05 @HEY_SPOT",
                     "▁LISTEN ▁UP ▁SPOT :1.5 #0.05 @LISTEN_UP_SPOT"]


class _StubStream:
    def __init__(self):
        self.waves = []

    def accept_waveform(self, rate, samples):
        self.waves.append(np.array(samples))


class _StubSpotter:
    def is_ready(self, stream):
        return False

    def get_result(self, stream):
        return ""


def _detector():
    from src.voice_control.wake.sherpa_onnx import WakeWordDetector
    det = WakeWordDetector(model_dir="/nonexistent")  # no model -> unavailable
    det._spotter, det._stream, det._available = _StubSpotter(), _StubStream(), True
    return det


def _pcm(peak):
    return (np.full(480, peak, np.float32) * 32768).astype(np.int16)


def test_wake_quiet_frame_gets_fixed_x8():
    det = _detector()
    frame = _pcm(0.05)
    det.process_frame(frame.tobytes())
    sent = det._stream.waves[0]
    np.testing.assert_allclose(sent, frame.astype(np.float32) / 32768.0 * 8.0, rtol=1e-6)


def test_wake_loud_frame_scaled_to_target_peak_no_clip():
    from src.voice_control.wake.sherpa_onnx import TARGET_PEAK
    det = _detector()
    det.process_frame(_pcm(0.4).tobytes())
    sent = det._stream.waves[0]
    assert np.max(np.abs(sent)) == pytest.approx(TARGET_PEAK, rel=1e-3)
    assert np.max(np.abs(sent)) < 1.0


def test_strip_stay_filler():
    assert strip_wake_phrase("stay spot what is this", ["HEY SPOT"]) == "what is this"
    assert strip_wake_phrase("heh spot what is this", ["HEY SPOT"]) == "what is this"


def test_build_keyword_lines_score_threshold():
    lines = build_keyword_lines(["HEY SPOT"], lambda p: ["A"], score=2.5, threshold=0.1)
    assert lines == ["A :2.5 #0.1 @HEY_SPOT"]
