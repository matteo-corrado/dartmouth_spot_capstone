"""Moved to src/spot/wake/. Compatibility alias; removed in Task 19."""
import importlib
import sys

_m = importlib.import_module("src.spot.wake")
for _sub in ("sherpa_onnx", "livekit", "phrases"):
    sys.modules[f"{__name__}.{_sub}"] = importlib.import_module(f"src.spot.wake.{_sub}")
sys.modules[f"{__name__}.safety_kws"] = importlib.import_module("src.spot.safety.kws")
sys.modules[__name__] = _m
