"""Moved to src/spot/mic/state_feedback.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.mic import state_feedback as _m
sys.modules[__name__] = _m
