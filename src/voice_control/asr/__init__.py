"""Moved to src/spot/stt/backends/. Compatibility alias; removed in Task 19."""
import importlib, sys
_m = importlib.import_module("src.spot.stt.backends")
for _sub in ("parakeet", "nemotron"):
    sys.modules[f"{__name__}.{_sub}"] = importlib.import_module(f"src.spot.stt.backends.{_sub}")
sys.modules[__name__] = _m
