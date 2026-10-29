"""Moved to src/spot/conversation/voiceprint.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.conversation import voiceprint as _m
sys.modules[__name__] = _m
