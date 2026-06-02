# tests/voice_control/test_state_feedback.py
"""Unit tests for the state -> chime/LED indicator mapping (Stage 2F P6/P7)."""
import math
from src.voice_control.client_mic import VoiceState
from src.voice_control.state_feedback import chime_for, led_color_for, LED_PERIOD_S


def test_listening_plays_ready_chime():
    assert chime_for(VoiceState.LISTENING) == "ready"


def test_thinking_plays_thinking_chime():
    assert chime_for(VoiceState.THINKING) == "thinking"


def test_responding_has_no_chime():
    # TTS itself is the audio cue; a chime over speech would be noise.
    assert chime_for(VoiceState.RESPONDING) is None


def test_wake_word_has_no_chime():
    assert chime_for(VoiceState.WAKE_WORD) is None


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
