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
    from src.voice_control.tts import _INSTANCES
    _INSTANCES.clear()  # don't let a cached singleton from another test pass vacuously
    monkeypatch.delenv("SPOT_TTS_BACKEND", raising=False)
    register_backend("elevenlabs", lambda: FakeBackend())  # DEFAULT_BACKEND
    register_backend("kokoro", lambda: FakeBackend())      # FALLBACK_BACKEND
    backend = get_backend()
    assert isinstance(backend, FakeBackend)


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


def test_get_backend_fallback_is_cached(monkeypatch):
    """After ElevenLabs init fails once and we fall back, subsequent calls
    must NOT re-run the broken factory (which would do a network round-trip
    per sentence chunk). Cache the fallback under the requested name."""
    from src.voice_control.tts import _INSTANCES, _REGISTRY
    _INSTANCES.clear()
    monkeypatch.setenv("SPOT_TTS_BACKEND", "elevenlabs")

    call_count = {"n": 0}

    def _bad_factory():
        call_count["n"] += 1
        raise RuntimeError("simulated")

    register_backend("elevenlabs", _bad_factory)
    register_backend("kokoro", lambda: FakeBackend())
    get_backend()
    get_backend()
    get_backend()
    assert call_count["n"] == 1, "fallback should be cached; bad factory ran once, not per-call"


def test_get_backend_dual_failure_preserves_primary_error(monkeypatch):
    """If both ElevenLabs AND Kokoro init fail, the operator needs to see
    the primary (ElevenLabs) error — not just the fallback failure."""
    from src.voice_control.tts import _INSTANCES
    _INSTANCES.clear()
    monkeypatch.setenv("SPOT_TTS_BACKEND", "elevenlabs")

    def _primary_bad():
        raise RuntimeError("ELEVENLABS_API_KEY missing")

    def _fallback_bad():
        raise RuntimeError("kokoro ONNX file not found")

    register_backend("elevenlabs", _primary_bad)
    register_backend("kokoro", _fallback_bad)
    with pytest.raises(RuntimeError) as exc:
        get_backend()
    msg = str(exc.value)
    assert "ELEVENLABS_API_KEY missing" in msg
    assert "kokoro ONNX file not found" in msg


def test_get_active_backend_name(monkeypatch):
    """get_active_backend_name() reflects the *actual* class returned, not
    the requested env var — so callers picking persona voice slots get the
    right key after a silent fallback."""
    from src.voice_control.tts import _INSTANCES, get_active_backend_name
    _INSTANCES.clear()
    monkeypatch.setenv("SPOT_TTS_BACKEND", "elevenlabs")

    class FakeKokoroBackend(FakeBackend):
        pass

    register_backend("elevenlabs", lambda: (_ for _ in ()).throw(RuntimeError("no key")))
    register_backend("kokoro", lambda: FakeKokoroBackend())
    assert get_active_backend_name() == "kokoro"
