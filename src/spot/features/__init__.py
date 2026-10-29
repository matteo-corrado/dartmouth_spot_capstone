"""Spot features package. REGISTRY holds the registered feature actions
(empty until Wave 4 registers features; every intent still goes to the
legacy spot_dispatch branches)."""
from src.spot.core.registry import Registry

REGISTRY = Registry()
from src.spot.features.enabled import ENABLED
REGISTRY.load(ENABLED)
