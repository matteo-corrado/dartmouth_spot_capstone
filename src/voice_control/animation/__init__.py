"""Moved to src/spot/features/mouth/. Compatibility alias; removed in Task 19."""
import importlib
import sys

_m = importlib.import_module("src.spot.features.mouth")
sys.modules[f"{__name__}.mouth"] = importlib.import_module("src.spot.features.mouth.driver")
sys.modules[__name__] = _m
