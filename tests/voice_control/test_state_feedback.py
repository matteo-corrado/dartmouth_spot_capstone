# tests/voice_control/test_state_feedback.py
"""Unit tests for the state -> chime/LED/ring indicator mapping."""
import math
from src.voice_control.client_mic import VoiceState
from src.voice_control.audio_feedback import AudioFeedback
from src.voice_control.state_feedback import (
    chime_for, led_color_for, LED_PERIOD_S, RingSpec, ring_for, ring_toward,
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
    assert ring_for(VoiceState.LISTENING, 90, False, reverse=True).ring == r


def test_ring_toward_shape_and_offset():
    r = ring_toward(90, 0x00C000)
    assert len(r) == 12 and sum(1 for v in r if v) == 3 and r[3] == 0x00C000
    r = ring_toward(0, 0x00C000, ring_offset_leds=2)
    assert r[1] and r[2] and r[3] and not r[0]


def test_ring_for_states():
    assert ring_for(VoiceState.RESPONDING, None, False).color == 0x0040FF
    assert ring_for(VoiceState.LISTENING, 90, False).effect == 5
    assert ring_for(VoiceState.THINKING, 90, True).color == 0xC00000


def test_ring_for_full_table():
    wake = ring_for(VoiceState.WAKE_WORD, None, False)
    assert (wake.effect, wake.color) == (1, 0x202020)
    listen = ring_for(VoiceState.LISTENING, None, False)
    assert (listen.effect, listen.color) == (3, 0x00C000)
    think = ring_for(VoiceState.THINKING, None, False)
    assert (think.effect, think.color) == (1, 0xC06000)
    locked = ring_for(VoiceState.LISTENING, 90, False, ring_offset_leds=1)
    assert isinstance(locked, RingSpec) and locked.ring == ring_toward(90, 0x00C000, ring_offset_leds=1)
    muted = ring_for(VoiceState.LISTENING, 90, True)
    assert (muted.effect, muted.color, muted.ring) == (3, 0xC00000, None)


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
