"""Moved to src/spot/features/visual_nav/navigator.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.features.visual_nav import navigator as _m
sys.modules[__name__] = _m
