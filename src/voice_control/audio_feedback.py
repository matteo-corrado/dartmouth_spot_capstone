"""Moved to src/spot/audio/chimes.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.audio import chimes as _m
sys.modules[__name__] = _m
