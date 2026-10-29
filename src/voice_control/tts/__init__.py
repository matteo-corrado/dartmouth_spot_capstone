"""Moved to src/spot/tts/. Compatibility alias; removed in Task 19."""
import importlib
import sys

_m = importlib.import_module("src.spot.tts")
for _sub in ("kokoro", "elevenlabs"):
    sys.modules[f"{__name__}.{_sub}"] = importlib.import_module(f"src.spot.tts.{_sub}")
sys.modules[__name__] = _m
