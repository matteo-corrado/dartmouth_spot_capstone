# src/voice_control/state_feedback.py
"""Pure state -> indicator mapping (Stage 2F P6/P7).

No audio device, no robot, no threads. The live loop (client_mic.py) calls
`chime_for(new_state)` to pick a beep method and `led_color_for(state)` /
`LED_PERIOD_S[state]` to drive the onboard LEDs. Kept pure so the policy is
unit-testable and the loop stays thin.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.voice_control.voice_state import VoiceState

# Chime fired when ENTERING a state. None = silent transition.
# Names map to methods on the audio_feedback `beep` singleton.
# Turn-taking cues are distinct from the power-on ``ready`` chime.
# ``released_cue`` is played explicitly on speaker-lock release.
_CHIME = {
    VoiceState.WAKE_WORD: None,          # dropping back to idle is silent
    VoiceState.LISTENING: "listen_cue",  # "go ahead, mic is open"
    VoiceState.THINKING: "think_cue",    # "got it, working"
    VoiceState.RESPONDING: "speak_cue",  # soft tick just before speech
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


# ---------------------------------------------------------------------------
# XVF3800 LED ring (primary turn signal). 12 LEDs, 30 degrees apart.
# ---------------------------------------------------------------------------
LED_EFFECT_BREATH = 1
LED_EFFECT_SOLID = 3
LED_EFFECT_RING = 5
RING_LEDS = 12
RING_BRIGHTNESS = 255

COLOR_IDLE = 0x202020       # dim white
COLOR_LISTEN = 0x00C000     # green
COLOR_THINK = 0xC06000      # amber
COLOR_SPEAK = 0x0040FF      # blue
COLOR_MUTED = 0xC00000      # red


@dataclass(frozen=True)
class RingSpec:
    """What the mic ring should show. ``ring`` (12 colours) is only used by
    the ring effect; ``color`` by breath/solid."""
    effect: int
    color: Optional[int]
    ring: Optional[list]
    brightness: int


def ring_toward(deg: float, color: int, width: int = 3,
                ring_offset_leds: int = 0) -> list:
    """12 LED colours lighting ``width`` LEDs centred on the LED nearest
    ``deg`` (0 deg = mic 0 deg), 0 elsewhere. ``ring_offset_leds`` shifts
    the index when LED 0 is not physically at mic 0 deg (calibrated with
    ``xvf_probe leds``; Tuning carries the value)."""
    # Round half down so 345 deg lands on LED 11 (not 0): keeps the
    # lit arc on the side the angle actually is.
    centre = (math.ceil(deg / 30.0 - 0.5) + ring_offset_leds) % RING_LEDS
    ring = [0] * RING_LEDS
    for i in range(width):
        ring[(centre - width // 2 + i) % RING_LEDS] = color
    return ring


def ring_for(state: VoiceState, lock_deg: Optional[float], muted: bool,
             ring_offset_leds: int = 0) -> RingSpec:
    """Ring spec for a voice state. Muted (solid red) overrides everything."""
    if muted:
        return RingSpec(LED_EFFECT_SOLID, COLOR_MUTED, None, RING_BRIGHTNESS)
    if state == VoiceState.LISTENING:
        if lock_deg is None:
            return RingSpec(LED_EFFECT_SOLID, COLOR_LISTEN, None, RING_BRIGHTNESS)
        return RingSpec(LED_EFFECT_RING, None,
                        ring_toward(lock_deg, COLOR_LISTEN, ring_offset_leds=ring_offset_leds),
                        RING_BRIGHTNESS)
    if state == VoiceState.THINKING:
        return RingSpec(LED_EFFECT_BREATH, COLOR_THINK, None, RING_BRIGHTNESS)
    if state == VoiceState.RESPONDING:
        return RingSpec(LED_EFFECT_SOLID, COLOR_SPEAK, None, RING_BRIGHTNESS)
    return RingSpec(LED_EFFECT_BREATH, COLOR_IDLE, None, RING_BRIGHTNESS)
