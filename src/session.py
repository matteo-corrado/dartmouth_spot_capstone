"""Moved to src/spot/robot/session.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.robot import session as _m
sys.modules[__name__] = _m
