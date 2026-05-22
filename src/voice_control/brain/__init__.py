"""Brain runtime backends. Selected via SPOT_BRAIN_BACKEND env var."""
import os

_DEFAULT = "llamacpp"


def get_backend():
    name = os.environ.get("SPOT_BRAIN_BACKEND", _DEFAULT).lower()
    if name == "llamacpp":
        from .llamacpp_backend import LlamaCppBackend
        return LlamaCppBackend()
    if name == "ollama":
        from .ollama_backend import OllamaBackend
        return OllamaBackend()
    raise ValueError(f"Unknown SPOT_BRAIN_BACKEND={name!r}; valid: llamacpp|ollama")
