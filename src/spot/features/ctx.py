"""What a registered feature handler gets: the owner objects plus the turn's
speak/brain callables."""
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class Ctx:
    connection: Any
    arm: Any
    safety: Any
    nav: Any
    speak: Optional[Callable[[str], None]] = None
    brain: Any = None
    bus: Any = None
    tuning: Any = None
