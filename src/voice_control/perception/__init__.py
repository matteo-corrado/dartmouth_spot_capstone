"""Moved to src/spot/perception/. Compatibility alias; removed in Task 19."""
import importlib, sys
_m = importlib.import_module("src.spot.perception")
for _sub in ("camera_intrinsics", "local_grid_helpers", "obstacle_query", "raycast_helpers"):
    sys.modules[f"{__name__}.{_sub}"] = importlib.import_module(f"src.spot.perception.{_sub}")
sys.modules[__name__] = _m
