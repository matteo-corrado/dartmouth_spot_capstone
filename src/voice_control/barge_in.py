"""Moved to src/spot/audio/barge_in.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.audio import barge_in as _m
sys.modules[__name__] = _m
