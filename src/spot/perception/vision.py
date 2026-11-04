"""Vision interface (people / objects query surface). NullVision is the only
implementation: nothing consumes it yet and it has no behaviour."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Vision(Protocol):
    def people(self) -> list[dict]: ...

    def objects(self) -> list[dict]: ...


class NullVision:
    """Sees nothing."""

    def people(self) -> list[dict]:
        return []

    def objects(self) -> list[dict]:
        return []
