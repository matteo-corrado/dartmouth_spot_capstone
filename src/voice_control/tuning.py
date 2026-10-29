"""Moved to src/spot/core/tuning.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.core import tuning as _m
sys.modules[__name__] = _m
