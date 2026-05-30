# src/voice_control/barge_in.py
"""Pure helpers for the post-TTS detector-state reset (Stage 2F P0a).

No audio device, no threads, no VoiceState — by construction the reset can
never change the loop's `state`, so it cannot bounce the state machine or
break command chaining. The live loop (client_mic.py) supplies the actual
vad/wake/drain callables.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


def should_reset_after_player(response_pending: bool, was_busy: bool, busy: bool) -> bool:
    """True exactly on the player busy->idle edge, and only if a TTS response
    armed the reset. The wake beep does NOT arm `response_pending`, so the
    in-breath command audio kept after a wake fire is never drained."""
    return response_pending and was_busy and not busy


@dataclass
class DetectorState:
    """Mutable view of the loop's detector buffers, so the reset is testable."""
    preroll: list = field(default_factory=list)
    pending: list = field(default_factory=list)
    consecutive: int = 0
    is_speaking: bool = False
    speech_buffer: bytearray = field(default_factory=bytearray)
    speech_float: list = field(default_factory=list)

    def reset(self, vad_reset: Callable[[], None], wake_reset: Callable[[], None],
              drain: Callable[[], None]) -> None:
        """Clear streaming-detector state so stale pre-mute decisions and the
        speaker tail can't corrupt the next detection. Returns None — there is
        deliberately no return value that could be used to mutate `state`."""
        self.preroll.clear()
        self.pending.clear()
        self.consecutive = 0
        self.is_speaking = False
        self.speech_buffer.clear()
        self.speech_float.clear()
        vad_reset()
        wake_reset()
        drain()
        return None
