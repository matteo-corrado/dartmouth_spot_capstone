"""Moved to src/spot/brain/intent.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.brain import intent as _m
sys.modules[__name__] = _m
