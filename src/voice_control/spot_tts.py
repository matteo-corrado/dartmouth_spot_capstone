"""Moved to src/spot/tts/speaker.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.tts import speaker as _m
sys.modules[__name__] = _m
