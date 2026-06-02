# tests/voice_control/test_response_gate.py
"""Unit tests for the post-response mic-reopen predicate (Stage 2F P6)."""
from src.voice_control.response_gate import (
    should_arm_reopen, should_reopen_mic, RESPONSE_REOPEN_DELAY_S)


def test_arm_when_gated_idle_and_unarmed():
    assert should_arm_reopen(gated=True, reopen_at=None, player_busy=False) is True


def test_no_arm_while_player_busy():
    # still speaking (or chime playing) — hold the gate, don't start the settle
    assert should_arm_reopen(gated=True, reopen_at=None, player_busy=True) is False


def test_no_arm_when_timer_already_pending():
    assert should_arm_reopen(gated=True, reopen_at=10.0, player_busy=False) is False


def test_no_arm_when_not_gated():
    assert should_arm_reopen(gated=False, reopen_at=None, player_busy=False) is False


def test_no_audio_turn_still_arms():
    # The deadlock case: a turn that played no TTS (player never busy) must
    # still arm the settle so the gate releases.
    assert should_arm_reopen(gated=True, reopen_at=None, player_busy=False) is True


def test_reopens_once_settle_elapsed():
    assert should_reopen_mic(gated=True, reopen_at=10.0, now=10.0) is True
    assert should_reopen_mic(gated=True, reopen_at=10.0, now=10.6) is True


def test_stays_gated_before_settle():
    assert should_reopen_mic(gated=True, reopen_at=10.0, now=9.7) is False


def test_not_gated_never_reopens():
    assert should_reopen_mic(gated=False, reopen_at=10.0, now=99.0) is False


def test_no_pending_reopen_time():
    assert should_reopen_mic(gated=True, reopen_at=None, now=99.0) is False


def test_delay_is_half_second():
    assert RESPONSE_REOPEN_DELAY_S == 0.5
