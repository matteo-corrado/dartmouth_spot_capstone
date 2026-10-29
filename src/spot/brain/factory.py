"""Brain factory. The LLM backend is still picked inside llm_brain from
SPOT_BRAIN_BACKEND (llm_brain._backend), exactly as today."""


def make_brain(model=None, cls=None):
    """LLMBrain(model). `cls` lets main pass its LLMBrain name."""
    if cls is None or model is None:
        from src.spot.brain.llm_brain import LLMBrain, DEFAULT_MODEL
        cls = cls or LLMBrain
        model = model or DEFAULT_MODEL
    return cls(model=model)
