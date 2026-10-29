"""Moved to src/spot/core/latency.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.core import latency as _m
sys.modules[__name__] = _m
