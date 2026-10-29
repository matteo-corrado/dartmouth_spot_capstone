"""Vision factory: NullVision until a real vision service is wired."""
from src.spot.perception.vision import NullVision, Vision


def make_vision() -> Vision:
    return NullVision()
