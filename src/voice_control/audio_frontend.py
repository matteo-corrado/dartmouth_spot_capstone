"""Moved to src/spot/audio/frontend.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.audio import frontend as _m
sys.modules[__name__] = _m
