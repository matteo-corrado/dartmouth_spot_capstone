"""WakeDetector interface: the sherpa-onnx spotter's public methods."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class WakeDetector(Protocol):
    def is_available(self) -> bool: ...

    def process_frame(self, pcm16_bytes: bytes) -> bool:
        """Feed one frame; True when the wake phrase fired."""

    def reset(self) -> None: ...
