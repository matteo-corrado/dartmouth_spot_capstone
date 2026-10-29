"""Moved to src/spot/core/startup_status.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.core import startup_status as _m
sys.modules[__name__] = _m
