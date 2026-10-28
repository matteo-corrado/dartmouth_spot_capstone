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

COLOR_IDLE = 0x595959       # white scaled 0.35, steady
COLOR_LISTEN = 0x00FF00     # green
COLOR_THINK = 0xFFB000      # light amber
COLOR_SPEAK = 0x0040FF      # blue
COLOR_MUTED = 0xFF0000      # red
SWEEP_FRAME_S = 0.07
BREATHE_FRAME_S = 0.04


@dataclass(frozen=True)
class RingSpec:
    """What the mic ring should show. All states use the per-LED ring effect.
    ``ring`` is the static 12-colour frame (fallback for animated specs);
    ``frames`` (if set) is a looping animation at ``frame_s`` per frame.
    ``effect``/``color``/``brightness`` are kept for compatibility."""
    effect: int
    color: Optional[int]
    ring: Optional[list]
    brightness: int
    frames: Optional[list] = None
    frame_s: float = 0.07


def scale_color(color: int, f: float) -> int:
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return (int(r * f) << 16) | (int(g * f) << 8) | int(b * f)


def _finite(deg) -> bool:
    try:
        return deg is not None and math.isfinite(float(deg))
    except (TypeError, ValueError):
        return False


def ring_toward(deg: float, color: int, width: int = 3,
                ring_offset_leds: int = 0, reverse: bool = False) -> list:
    """12 LED colours lighting ``width`` LEDs centred on the LED nearest
    ``deg``, 0 elsewhere.

    ``deg`` is in the MIC frame (the XVF3800's own DoA frame), where LED
    index i sits at i*30 deg. ``ring_offset_leds`` shifts the index and
    ``reverse`` flips the direction when the physical LED order differs
    (calibrated with ``xvf_probe leds``; Tuning carries the values).
    A missing or non-finite ``deg`` (None, NaN, inf) lights the whole ring."""
    if not _finite(deg):
        return [color] * RING_LEDS
    step = round(deg / 30.0) * (-1 if reverse else 1)
    centre = (step + ring_offset_leds) % RING_LEDS
    ring = [0] * RING_LEDS
    for i in range(width):
        ring[(centre - width // 2 + i) % RING_LEDS] = color
    return ring


def sweep_frames(deg, color, ring_offset_leds: int = 0, reverse: bool = False) -> list:
    """Car-indicator fill toward ``deg`` (mic frame): 7 growing frames, one
    all-lit frame, two all-off frames. Non-finite ``deg`` -> [solid colour]."""
    if not _finite(deg):
        return [[color] * RING_LEDS]
    step = round(deg / 30.0) * (-1 if reverse else 1)
    target = (step + ring_offset_leds) % RING_LEDS

    def dist(i):
        d = abs(i - target) % RING_LEDS
        return min(d, RING_LEDS - d)

    frames = [[color if dist(i) >= 6 - k else 0 for i in range(RING_LEDS)]
              for k in range(7)]
    return frames + [[color] * RING_LEDS] + [[0] * RING_LEDS] * 2


def breathe_frames(color: int, lo: float = 0.2) -> list:
    """22 all-LED frames ramping lo -> 1.0 -> lo."""
    seq = [lo + (1 - lo) * x / 10 for x in range(11)]
    return [[scale_color(color, f)] * RING_LEDS for f in seq + seq[::-1]]


def ring_for(state: VoiceState, lock_deg: Optional[float], muted: bool,
             ring_offset_leds: int = 0, reverse: bool = False) -> RingSpec:
    """Ring spec for a voice state. Muted (solid red) overrides everything.
    ``lock_deg`` is in the MIC frame (XVF3800 DoA, LED i at i*30 deg)."""
    def solid(color):
        return RingSpec(LED_EFFECT_RING, color, [color] * RING_LEDS, RING_BRIGHTNESS)

    if muted:
        return solid(COLOR_MUTED)
    if state == VoiceState.LISTENING:
        if not _finite(lock_deg):   # None / NaN / inf -> solid green
            return solid(COLOR_LISTEN)
        return RingSpec(LED_EFFECT_RING, COLOR_LISTEN,
                        ring_toward(lock_deg, COLOR_LISTEN,
                                    ring_offset_leds=ring_offset_leds, reverse=reverse),
                        RING_BRIGHTNESS,
                        sweep_frames(lock_deg, COLOR_LISTEN, ring_offset_leds, reverse),
                        SWEEP_FRAME_S)
    if state == VoiceState.THINKING:
        return RingSpec(LED_EFFECT_RING, COLOR_THINK,
                        [scale_color(COLOR_THINK, 0.6)] * RING_LEDS, RING_BRIGHTNESS,
                        breathe_frames(COLOR_THINK), BREATHE_FRAME_S)
    if state == VoiceState.RESPONDING:
        return solid(COLOR_SPEAK)
    return solid(COLOR_IDLE)
