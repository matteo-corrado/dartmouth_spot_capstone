"""Unit tests for gripper-gaze bearing math (Task 19)."""
import pytest

from src.voice_control.gaze import gaze_target, should_regaze


def test_gaze_front_and_side():
    x, y, z = gaze_target(0)
    assert x == pytest.approx(2.0) and abs(y) < 1e-6
    x, y, z = gaze_target(60)
    assert x == pytest.approx(1.0) and y == pytest.approx(2.0 * 0.8660254)


def test_gaze_height_and_right_side():
    x, y, z = gaze_target(-60, dist_m=1.0, height_m=0.5)
    assert x == pytest.approx(0.5) and y == pytest.approx(-0.8660254) and z == pytest.approx(0.5)


def test_gaze_behind_none():
    assert gaze_target(170) is None and gaze_target(-150) is None
    assert gaze_target(90) is None and gaze_target(-90) is None


def test_gaze_wraps_and_edges():
    assert gaze_target(75) is not None and gaze_target(76) is None
    x, y, _ = gaze_target(350)  # == -10 deg
    assert gaze_target(-10) == pytest.approx((x, y, 0.9))
    assert gaze_target(290) is not None  # wraps to -70 deg: in front arc
    assert gaze_target(250) is None  # wraps to -110 deg: outside the arc


def test_gaze_rejects_non_finite():
    assert gaze_target(float("nan")) is None
    assert gaze_target(float("inf")) is None


def test_regaze_threshold():
    assert should_regaze(0, 10) is False and should_regaze(0, 20) is True and should_regaze(350, 10) is True
    assert should_regaze(0, 15) is False  # strictly greater than min_change
    assert should_regaze(10, 350) is True and should_regaze(355, 5) is False
