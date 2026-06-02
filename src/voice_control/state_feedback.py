# src/voice_control/state_feedback.py
"""Pure state -> indicator mapping (Stage 2F P6/P7).

No audio device, no robot, no threads. The live loop (client_mic.py) calls
`chime_for(new_state)` to pick a beep method and `led_color_for(state)` /
`LED_PERIOD_S[state]` to drive the onboard LEDs. Kept pure so the policy is
unit-testable and the loop stays thin.
"""
from __future__ import annotations

from src.voice_control.client_mic import VoiceState

# Chime fired when ENTERING a state. None = silent transition.
# Names map to methods on the audio_feedback `beep` singleton.
_CHIME = {
    VoiceState.WAKE_WORD: None,    # dropping back to idle is silent
    VoiceState.LISTENING: "ready",   # "go ahead" cue (also = response-done)
    VoiceState.THINKING: "thinking",  # "got it, working"
    VoiceState.RESPONDING: None,    # the speech itself is the cue
}

# RGB 0-255 per state. Vector magnitude kept <= 255 (BD AV-LED warranty cap).
# Distinct hues: idle=dim blue, listening=green, thinking=amber, speaking=cyan.
_LED_COLOR = {
    VoiceState.WAKE_WORD: (0, 0, 60),
    VoiceState.LISTENING: (0, 120, 0),
    VoiceState.THINKING: (150, 80, 0),
    VoiceState.RESPONDING: (0, 90, 120),
}

# Pulse period (seconds) per state. Slow = calm/idle, fast = active.
LED_PERIOD_S = {
    VoiceState.WAKE_WORD: 3.0,
    VoiceState.LISTENING: 1.5,
    VoiceState.THINKING: 0.7,
    VoiceState.RESPONDING: 1.0,
}


def chime_for(state: VoiceState):
    """Name of the beep method to play when entering `state`, or None."""
    return _CHIME.get(state)


def led_color_for(state: VoiceState):
    """(r, g, b) 0-255 for the state's LED pulse."""
    return _LED_COLOR.get(state, (0, 0, 0))
