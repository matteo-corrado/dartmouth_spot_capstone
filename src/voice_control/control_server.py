"""Moved to src/spot/operator/control_server.py. Compatibility alias; removed in Task 19."""
import sys
from src.spot.operator import control_server as _m
sys.modules[__name__] = _m
