# tests/voice_control/test_response_gate.py
"""Unit tests for the post-response mic-reopen predicate (Stage 2F P6)."""
from src.voice_control.response_gate import should_reopen_mic, RESPONSE_REOPEN_DELAY_S


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
