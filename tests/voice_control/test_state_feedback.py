# tests/voice_control/test_state_feedback.py
"""Unit tests for the state -> chime/LED/ring indicator mapping."""
import math
from src.voice_control.client_mic import VoiceState
from src.voice_control.audio_feedback import AudioFeedback
from src.voice_control.state_feedback import (
    chime_for, led_color_for, LED_PERIOD_S, RingSpec, ring_for, ring_toward,
    sweep_frames, breathe_frames,
)


def test_listening_plays_listen_cue():
    assert chime_for(VoiceState.LISTENING) == "listen_cue"


def test_thinking_plays_think_cue():
    assert chime_for(VoiceState.THINKING) == "think_cue"


def test_responding_plays_speak_cue():
    assert chime_for(VoiceState.RESPONDING) == "speak_cue"


def test_wake_word_has_no_chime():
    assert chime_for(VoiceState.WAKE_WORD) is None


def test_turn_chimes_distinct_from_ready():
    names = {chime_for(s) for s in (VoiceState.LISTENING, VoiceState.THINKING, VoiceState.RESPONDING)}
    assert names == {"listen_cue", "think_cue", "speak_cue"} and "ready" not in names


def test_chime_names_are_audio_feedback_methods():
    for st in VoiceState:
        name = chime_for(st)
        if name is not None:
            assert callable(getattr(AudioFeedback, name))


def test_ring_toward_front_and_wrap():
    r = ring_toward(0, 0x00C000); assert r[0] and r[1] and r[11] and not r[6]
    r = ring_toward(330, 0x00C000); assert r[0] and r[11] and r[10]


def test_ring_toward_reverse():
    r = ring_toward(90, 0x00C000, reverse=True)   # centre -3 % 12 = 9
    assert r[8] and r[9] and r[10] and not r[3]
    assert ring_for(VoiceState.LISTENING, 90, False, reverse=True).ring == ring_toward(90, 0x00FF00, reverse=True)


def test_ring_toward_shape_and_offset():
    r = ring_toward(90, 0x00C000)
    assert len(r) == 12 and sum(1 for v in r if v) == 3 and r[3] == 0x00C000
    r = ring_toward(0, 0x00C000, ring_offset_leds=2)
    assert r[1] and r[2] and r[3] and not r[0]


def test_ring_for_states():
    assert ring_for(VoiceState.RESPONDING, None, False).color == 0x0040FF
    assert ring_for(VoiceState.LISTENING, 90, False).effect == 5
    assert ring_for(VoiceState.THINKING, 90, True).color == 0xFF0000


def test_ring_for_full_table():
    wake = ring_for(VoiceState.WAKE_WORD, None, False)
    assert wake.effect == 5 and wake.frames is None and wake.ring == [0x595959] * 12
    listen = ring_for(VoiceState.LISTENING, None, False)
    assert listen.frames is None and listen.ring == [0x00FF00] * 12
    think = ring_for(VoiceState.THINKING, None, False)
    assert think.color == 0xFFB000 and len(think.frames) == 22 and think.frame_s == 0.04
    assert len(think.ring) == 12
    speak = ring_for(VoiceState.RESPONDING, None, False)
    assert speak.frames is None and speak.ring == [0x0040FF] * 12
    locked = ring_for(VoiceState.LISTENING, 90, False, ring_offset_leds=1)
    assert isinstance(locked, RingSpec) and locked.frame_s == 0.07
    assert locked.ring == ring_toward(90, 0x00FF00, ring_offset_leds=1)
    assert locked.frames == sweep_frames(90, 0x00FF00, ring_offset_leds=1)
    muted = ring_for(VoiceState.LISTENING, 90, True)
    assert (muted.color, muted.ring, muted.frames) == (0xFF0000, [0xFF0000] * 12, None)


def test_sweep_frames_shape_and_target():
    g = 0x00FF00
    f = sweep_frames(90, g)          # target LED 3
    assert len(f) == 10
    # k=0 lights only the LED opposite the target (LED 9)
    assert f[0] == [g if i == 9 else 0 for i in range(12)]
    assert f[6][3] == g and f[6][9] == g and all(f[6])
    assert f[7] == [g] * 12 and f[8] == f[9] == [0] * 12
    assert sweep_frames(0, g, ring_offset_leds=2)[0][8] == g
    assert sweep_frames(float("nan"), g) == [[g] * 12]


def test_breathe_frames():
    f = breathe_frames(0xFFB000)
    assert len(f) == 22 and f[10] == [0xFFB000] * 12
    assert f[0] == f[21] and f[0][0] == (0x33 << 16) | (int(0xB0 * 0.2) << 8)


def test_non_finite_lock_deg_is_solid_green():
    for bad in (float("nan"), float("inf"), float("-inf"), None):
        spec = ring_for(VoiceState.LISTENING, bad, False)
        assert spec.frames is None and spec.ring == [0x00FF00] * 12
    assert ring_toward(float("nan"), 0x00C000) == [0x00C000] * 12


def test_every_state_has_an_led_color():
    for st in VoiceState:
        r, g, b = led_color_for(st)
        assert all(0 <= c <= 255 for c in (r, g, b))


def test_led_magnitudes_within_warranty_cap():
    for st in VoiceState:
        r, g, b = led_color_for(st)
        assert math.sqrt(r * r + g * g + b * b) <= 255.0


def test_states_are_visually_distinct():
    colors = {st: led_color_for(st) for st in VoiceState}
    assert len(set(colors.values())) == len(VoiceState)


def test_period_defined_for_every_state():
    for st in VoiceState:
        assert LED_PERIOD_S[st] > 0
