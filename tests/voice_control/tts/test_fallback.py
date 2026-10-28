"""B8: per-sentence Kokoro fallback when the primary TTS backend fails."""
from src.voice_control.tts import synthesize_with_fallback


class Good:
    def __init__(self):
        self.voices = []

    def synthesize(self, t, v):
        self.voices.append(v)
        return b"\x01\x02"


def test_fallback_on_primary_error():
    class Bad:  # primary
        def synthesize(self, t, v): raise ConnectionError("wifi")
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
