import pytest
from src.voice_control.tts import TTSBackend, get_backend, register_backend


class FakeBackend:
    def synthesize(self, text, voice_id): return b"PCM"
    def stream(self, text, voice_id):
        yield b"PCM"
    def list_voices(self): return []


def test_register_and_get_backend(monkeypatch):
    register_backend("fake", lambda: FakeBackend())
    monkeypatch.setenv("SPOT_TTS_BACKEND", "fake")
    backend = get_backend()
    assert backend.synthesize("hi", "v1") == b"PCM"


def test_get_backend_unknown_raises(monkeypatch):
    monkeypatch.setenv("SPOT_TTS_BACKEND", "nonexistent_xyz")
    with pytest.raises(ValueError) as exc:
        get_backend()
    assert "nonexistent_xyz" in str(exc.value)


def test_get_backend_default_when_env_unset(monkeypatch):
    monkeypatch.delenv("SPOT_TTS_BACKEND", raising=False)
    register_backend("elevenlabs", lambda: FakeBackend())  # DEFAULT_BACKEND
    register_backend("kokoro", lambda: FakeBackend())      # FALLBACK_BACKEND
    backend = get_backend()
    assert backend is not None


def test_get_backend_fallback_on_init_failure(monkeypatch):
    """If the chosen backend cannot be instantiated (e.g. ElevenLabs without
    an API key), get_backend() falls back to Kokoro so Spot keeps talking."""
    from src.voice_control.tts import _INSTANCES
    _INSTANCES.clear()
    monkeypatch.setenv("SPOT_TTS_BACKEND", "elevenlabs")

    def _bad_factory():
        raise RuntimeError("simulated init failure (no API key)")

    register_backend("elevenlabs", _bad_factory)
    register_backend("kokoro", lambda: FakeBackend())
    backend = get_backend()
    assert isinstance(backend, FakeBackend)
