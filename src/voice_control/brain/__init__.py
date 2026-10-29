"""Moved to src/spot/brain/backends/. Compatibility alias; removed in Task 19."""
import importlib
import sys

_m = importlib.import_module("src.spot.brain.backends")
for _sub in ("llamacpp_backend", "ollama_backend"):
    sys.modules[f"{__name__}.{_sub}"] = importlib.import_module(f"src.spot.brain.backends.{_sub}")
sys.modules[__name__] = _m
