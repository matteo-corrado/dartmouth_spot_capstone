"""Moved to src/spot/features/gaze/math.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.features.gaze import math as _m
sys.modules[__name__] = _m
