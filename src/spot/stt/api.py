"""SpeechToText interface: what the conversation needs from ASR."""
from typing import Protocol, runtime_checkable


@runtime_checkable
class SpeechToText(Protocol):
    def transcribe(self, pcm16: bytes, timeout: float | None) -> str:
        """Transcript of one utterance ('' on error or timeout)."""
