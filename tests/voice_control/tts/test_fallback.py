"""B8: per-sentence Kokoro fallback when the primary TTS backend fails."""
import time

import pytest

import src.voice_control.tts as tts_mod
from src.voice_control.tts import synthesize_with_fallback


@pytest.fixture(autouse=True)
def _reset_sticky(monkeypatch):
    monkeypatch.setattr(tts_mod, "_primary_skip_until", float("-inf"))


class Good:
    def __init__(self):
        self.voices = []

    def synthesize(self, t, v):
        self.voices.append(v)
        return b"\x01\x02"


class Bad:
    def __init__(self):
        self.calls = 0

    def synthesize(self, t, v):
        self.calls += 1
        raise ConnectionError("wifi")


def test_fallback_on_primary_error():
    audio, used = synthesize_with_fallback("hi", "x", Bad(), Good(), "af_sarah")
    assert audio == b"\x01\x02" and used == "fallback"


def test_fallback_on_empty_audio_uses_fallback_voice():
    class Empty:
        def synthesize(self, t, v): return b""
    good = Good()
    audio, used = synthesize_with_fallback("hi", "x", Empty(), good, "af_sarah")
    assert used == "fallback" and good.voices == ["af_sarah"]


def test_primary_ok():
    good = Good()
    audio, used = synthesize_with_fallback("hi", "x", good, Good(), "af_sarah")
    assert used == "primary" and good.voices == ["x"]


def test_slow_primary_times_out_to_fallback():
    class Slow:
        def synthesize(self, t, v):
            time.sleep(0.5)
            return b"\x09"
    t0 = time.monotonic()
    audio, used = synthesize_with_fallback("hi", "x", Slow(), Good(), "af_sarah", budget_s=0.05)
    assert used == "fallback" and audio == b"\x01\x02"
    assert time.monotonic() - t0 < 0.4


def test_sticky_window_skips_primary_then_retries(monkeypatch):
    bad = Bad()
    synthesize_with_fallback("a", "x", bad, Good(), "af_sarah")
    assert bad.calls == 1
    # Inside the 60 s window the primary is not even tried.
    _, used = synthesize_with_fallback("b", "x", bad, Good(), "af_sarah")
    assert used == "fallback" and bad.calls == 1
    # Window over -> primary retried.
    monkeypatch.setattr(tts_mod, "_primary_skip_until", time.monotonic() - 1)
    good = Good()
    _, used = synthesize_with_fallback("c", "x", good, Good(), "af_sarah")
    assert used == "primary"
