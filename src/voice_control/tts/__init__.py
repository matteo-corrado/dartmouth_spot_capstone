"""TTS backend registry + dispatch.

Backends register a factory under a name (e.g. "kokoro", "elevenlabs").
`get_backend()` reads SPOT_TTS_BACKEND env var and returns an instance.
Default backend: "elevenlabs" (premium quality). Falls back to "kokoro"
(local, no API key needed) if the primary backend cannot be instantiated.
"""
import os
from typing import Iterator, Protocol, runtime_checkable


@runtime_checkable
class TTSBackend(Protocol):
    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Render full PCM bytes (blocking). Returns audio at backend's native sample rate."""
        ...

    def stream(self, text: str, voice_id: str) -> Iterator[bytes]:
        """Yield PCM chunks as they're produced."""
        ...

    def list_voices(self) -> list:
        """Return list of available voice descriptors."""
        ...


_REGISTRY: dict = {}
_INSTANCES: dict = {}  # name -> singleton TTSBackend (lazily constructed)
DEFAULT_BACKEND = "elevenlabs"
FALLBACK_BACKEND = "kokoro"


def register_backend(name: str, factory) -> None:
    """Register a backend factory. Factory takes no args and returns a TTSBackend."""
    _REGISTRY[name] = factory


def get_backend() -> TTSBackend:
    """Resolve the backend named by SPOT_TTS_BACKEND env var. Memoized per
    backend name so the underlying model loads ONCE — spot_tts._render
    calls this per sentence chunk and Kokoro init re-loads ~85MB of
    ONNX weights + allocates a CUDA context every time without the cache.

    If the requested backend fails to instantiate (e.g. ElevenLabs without
    ELEVENLABS_API_KEY or with no network) we fall back to FALLBACK_BACKEND
    so Spot keeps talking. Caller-explicit SPOT_TTS_BACKEND=kokoro short-
    circuits the fallback (no double-init).
    """
    name = os.environ.get("SPOT_TTS_BACKEND", DEFAULT_BACKEND)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown TTS backend '{name}'. Registered: {sorted(_REGISTRY.keys())}"
        )
    if name not in _INSTANCES:
        try:
            _INSTANCES[name] = _REGISTRY[name]()
        except Exception as e:
            if name == FALLBACK_BACKEND or FALLBACK_BACKEND not in _REGISTRY:
                raise
            print(f"[TTS] '{name}' init failed ({type(e).__name__}: {e}); "
                  f"falling back to '{FALLBACK_BACKEND}'.")
            if FALLBACK_BACKEND not in _INSTANCES:
                _INSTANCES[FALLBACK_BACKEND] = _REGISTRY[FALLBACK_BACKEND]()
            return _INSTANCES[FALLBACK_BACKEND]
    return _INSTANCES[name]
