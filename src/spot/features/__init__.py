"""Spot features package. REGISTRY holds the enabled features (enabled.py).
Their actions are dispatched through src/spot/robot/dispatcher.py, and the
LLM grammar, prompt action list, regex intent table and intent sets are
generated from them plus the core intents (src/spot/core/core_intents.py)."""
from src.spot.core.registry import Registry

REGISTRY = Registry()
from src.spot.features.enabled import ENABLED
REGISTRY.load(ENABLED)
