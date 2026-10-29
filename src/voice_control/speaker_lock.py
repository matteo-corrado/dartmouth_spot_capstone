"""Moved to src/spot/conversation/speaker_lock.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.conversation import speaker_lock as _m
sys.modules[__name__] = _m
