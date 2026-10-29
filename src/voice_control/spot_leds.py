"""Moved to src/spot/mic/spot_leds.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.mic import spot_leds as _m
sys.modules[__name__] = _m
