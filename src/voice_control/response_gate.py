# src/voice_control/response_gate.py
"""Pure predicate for reopening the mic after a response (Stage 2F P6).

The mic is hard-gated from the moment Spot commits to a response (THINKING)
through the end of TTS plus a short settle, so no speech uttered during that
window is ever processed (no barge-in, no late phantom commands). This module
only answers "has the settle elapsed?" — the live loop owns the flag and the
actual drain/reset. No audio, no threads, no VoiceState.
"""
from __future__ import annotations

# Seconds the mic stays gated after playback goes idle, before re-listening.
# Covers the OS audio-buffer drain + room echo of Spot's own voice.
RESPONSE_REOPEN_DELAY_S = 0.5


def should_arm_reopen(gated: bool, reopen_at, player_busy: bool) -> bool:
    """True when the post-response settle timer should be (re)started.

    Armed only while gated, the player is idle, and no timer is pending. The
    release is therefore IDLE-based, not tied to a TTS busy->idle edge — so a
    turn that plays no audio at all (e.g. a noise burst that yields an empty
    ASR transcript) still releases the gate and can never wedge it shut.
    """
    return gated and not player_busy and reopen_at is None


def should_reopen_mic(gated: bool, reopen_at, now: float) -> bool:
    """True once the post-response settle has elapsed and the mic may reopen.

    gated:     is the mic currently gated for a response?
    reopen_at: monotonic time the settle ends (None until playback goes idle).
    now:       current monotonic time.
    """
    return gated and reopen_at is not None and now >= reopen_at
