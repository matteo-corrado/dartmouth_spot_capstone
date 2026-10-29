"""Moved to src/spot/safety/estop.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.safety import estop as _m
sys.modules[__name__] = _m
