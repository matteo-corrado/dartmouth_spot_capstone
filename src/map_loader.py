"""Moved to src/spot/nav/map_loader.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.nav import map_loader as _m
sys.modules[__name__] = _m
