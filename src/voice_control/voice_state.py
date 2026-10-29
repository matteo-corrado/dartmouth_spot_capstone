"""Moved to src/spot/core/voice_state.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.core import voice_state as _m
sys.modules[__name__] = _m
