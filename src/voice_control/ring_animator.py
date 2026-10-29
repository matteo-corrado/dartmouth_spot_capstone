"""Moved to src/spot/mic/ring_animator.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.mic import ring_animator as _m
sys.modules[__name__] = _m
