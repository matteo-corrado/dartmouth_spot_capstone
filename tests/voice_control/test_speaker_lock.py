import pytest

from src.voice_control.speaker_lock import (
    SpeakerLock, ang_diff, circular_median_deg, mic_to_body_deg)


def test_circular_median_wraps():
    assert ang_diff(circular_median_deg([350, 355, 5, 10, 0]), 0) <= 5


def test_circular_median_empty():
    assert circular_median_deg([]) is None


def test_ang_diff_range():
    assert ang_diff(355, 10) == pytest.approx(15)
    assert ang_diff(0, 180) == 180


def test_accept_across_zero():
    L = SpeakerLock(35, 20, 0.3); L.lock([355], now=0)
    assert L.accept([10], now=1) is True


def test_reject_other_side():
    L = SpeakerLock(35, 20, 0.3); L.lock([90], 0)
    assert L.accept([250, 255], 1) is False


def test_accept_with_no_readings_accepts():
    L = SpeakerLock(35, 20, 0.3); L.lock([90], 0)
    assert L.accept([], 1) is True


def test_not_locked_accepts():
    assert SpeakerLock(35, 20, 0.3).accept([123], 0) is True


def test_nudge_and_timeout():
    L = SpeakerLock(35, 20, 0.5); L.lock([0], 0); L.accept([20], 5)
    assert L.angle == pytest.approx(10, abs=0.5)
    assert L.expired(24) is False and L.expired(25.1) is True


def test_nudge_across_zero():
    L = SpeakerLock(35, 20, 0.5); L.lock([355], 0); L.accept([15], 1)
    assert ang_diff(L.angle, 5) < 0.5


def test_lock_with_no_readings_stays_unlocked():
    L = SpeakerLock(35, 20, 0.3)
    assert L.lock([], 0) is None and L.locked is False


def test_release():
    L = SpeakerLock(35, 20, 0.3); L.lock([10], 0); L.release()
    assert L.locked is False and L.expired(1000) is False


def test_mic_to_body():
    assert mic_to_body_deg(100, offset_deg=90, sign=1) == pytest.approx(10)
    assert mic_to_body_deg(80, offset_deg=90, sign=-1) == pytest.approx(10)
