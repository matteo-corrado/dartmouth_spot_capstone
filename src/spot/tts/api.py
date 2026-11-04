"""TextToSpeech interface: SpotTTS's existing public surface."""
from typing import Callable, Protocol, runtime_checkable


@runtime_checkable
class TextToSpeech(Protocol):
    def is_available(self) -> bool: ...

    def set_volume(self, volume: float) -> float: ...

    def speak(self, text: str, voice: str | None = None) -> None:
        """Queue `text` for synthesis + playback; returns immediately."""

    def speak_sync(self, text: str) -> None: ...

    def wait(self) -> None: ...

    def enqueue_streaming(self, voice: str | None = None) -> Callable[[str], None]: ...
