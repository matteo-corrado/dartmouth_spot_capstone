"""Moved to src/spot/nav/graph_nav_utils.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.nav import graph_nav_utils as _m
sys.modules[__name__] = _m
