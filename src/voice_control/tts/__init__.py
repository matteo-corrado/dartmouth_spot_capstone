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
    so Spot keeps talking. The fallback instance is cached under BOTH the
    requested name AND FALLBACK_BACKEND so subsequent calls do not re-try
    the broken primary factory (which would do a per-utterance auth round-
    trip). Caller-explicit SPOT_TTS_BACKEND=kokoro short-circuits the
    fallback (no double-init).

    Use `get_active_backend_name()` instead of reading SPOT_TTS_BACKEND when
    you need to know what is *actually* running — they diverge after a
    silent fallback.
    """
    name = os.environ.get("SPOT_TTS_BACKEND", DEFAULT_BACKEND)
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown TTS backend '{name}'. Registered: {sorted(_REGISTRY.keys())}"
        )
    if name not in _INSTANCES:
        try:
            _INSTANCES[name] = _REGISTRY[name]()
        except Exception as primary_err:
            if name == FALLBACK_BACKEND:
                raise
            if FALLBACK_BACKEND not in _REGISTRY:
                raise RuntimeError(
                    f"'{name}' init failed and FALLBACK_BACKEND "
                    f"'{FALLBACK_BACKEND}' is not registered; cannot recover."
                ) from primary_err
            primary_msg = str(primary_err).replace("\n", " | ")
            print(f"[TTS] '{name}' init failed "
                  f"({type(primary_err).__name__}: {primary_msg}); "
                  f"falling back to '{FALLBACK_BACKEND}'.")
            if FALLBACK_BACKEND not in _INSTANCES:
                try:
                    _INSTANCES[FALLBACK_BACKEND] = _REGISTRY[FALLBACK_BACKEND]()
                except Exception as fb_err:
                    # Preserve the original ElevenLabs error so the operator
                    # can diagnose what blocked the primary, not just the
                    # fallback failure.
                    raise RuntimeError(
                        f"Both '{name}' "
                        f"({type(primary_err).__name__}: {primary_msg}) and "
                        f"fallback '{FALLBACK_BACKEND}' "
                        f"({type(fb_err).__name__}: {fb_err}) failed."
                    ) from primary_err
            # Cache fallback under the *requested* name so the next call
            # short-circuits to it without re-attempting the broken primary.
            _INSTANCES[name] = _INSTANCES[FALLBACK_BACKEND]
    return _INSTANCES[name]


def get_active_backend_name() -> str:
    """Return the canonical short name ('kokoro' / 'elevenlabs') of the
    backend get_backend() actually returns, accounting for fallback.

    Callers that need to pick persona-mapped voice_ids correctly even
    after a silent ElevenLabs→Kokoro fallback should use this, NOT
    `os.environ['SPOT_TTS_BACKEND']` which still reflects the *requested*
    backend, not the *active* one.
    """
    backend = get_backend()
    cls = type(backend).__name__
    if "Kokoro" in cls:
        return "kokoro"
    if "ElevenLabs" in cls:
        return "elevenlabs"
    return cls.lower()
