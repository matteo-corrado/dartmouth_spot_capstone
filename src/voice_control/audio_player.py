"""Moved to src/spot/audio/player.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.audio import player as _m
sys.modules[__name__] = _m
