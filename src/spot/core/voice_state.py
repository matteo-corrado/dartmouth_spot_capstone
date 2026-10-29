# src/voice_control/voice_state.py
"""Shared VoiceState enum for the voice pipeline.

Lives in its own leaf module so state_feedback / spot_leds can import it
without pulling in (or cycling through) the heavy client_mic module.
client_mic runs as ``__main__``, so importing VoiceState *from* client_mic
re-imports client_mic a second time under its package name — the
partial-init circular import that used to crash startup.
"""
from __future__ import annotations

from enum import Enum, auto


class VoiceState(Enum):
    WAKE_WORD = auto()   # Waiting for "hey spot" (detected via ASR, not a separate model)
    LISTENING = auto()   # Wake word heard, waiting for speech
    THINKING = auto()    # Utterance captured — ASR + LLM running (mic gated)
    RESPONDING = auto()  # TTS playing the response (mic gated)
