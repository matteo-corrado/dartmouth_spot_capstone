"""Moved to src/spot/brain/llm_brain.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.brain import llm_brain as _m
sys.modules[__name__] = _m
