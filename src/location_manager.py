"""Moved to src/spot/nav/location_manager.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.nav import location_manager as _m
sys.modules[__name__] = _m
