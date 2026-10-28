"""Task 19 dispatcher side: gaze_at_bearing with a fake command client."""
import pytest

import src.voice_control.spot_dispatch as d
from tests.voice_control.test_arm_policy import FakeCmd


@pytest.fixture
def cmd(monkeypatch):
    c = FakeCmd()
    monkeypatch.setattr(d, "_spot_session", {"cmd": c, "robot": object()})
    monkeypatch.setattr(d, "_arm_pose", "talk")
    monkeypatch.setattr(d, "is_navigating", lambda: False)
    return c


def test_gaze_left_slow(cmd):
    assert d.gaze_at_bearing(90.0) is True
    (kind, proto), = cmd.sent
    assert kind == "arm_gaze"
    g = proto.synchronized_command.arm_command.arm_gaze_command
    p = g.target_trajectory_in_frame1.points[0].point
    assert abs(p.x) < 1e-6 and p.y == pytest.approx(2.0) and p.z == pytest.approx(0.9)
    assert g.frame1_name == "body"
    assert g.max_linear_velocity.value == pytest.approx(0.3)
    assert g.max_angular_velocity.value == pytest.approx(1.0)


def test_behind_ignored(cmd):
    assert d.gaze_at_bearing(170.0) is False
    assert cmd.sent == []


def test_release_returns_to_talk(cmd, monkeypatch):
    monkeypatch.setattr(d, "block_until_arm_arrives",
                        lambda *a: pytest.fail("gaze must not block"))
    assert d.gaze_at_bearing(None) is True
    assert cmd.kinds() == ["arm_ready"]


@pytest.mark.parametrize("pose", ["stowed", "walk"])
def test_only_in_talk_pose(cmd, monkeypatch, pose):
    monkeypatch.setattr(d, "_arm_pose", pose)
    assert d.gaze_at_bearing(10.0) is False
    assert d.gaze_at_bearing(None) is False
    assert cmd.sent == []


def test_not_while_walking_or_dancing(cmd, monkeypatch):
    monkeypatch.setattr(d, "is_navigating", lambda: True)
    assert d.gaze_at_bearing(10.0) is False
    assert cmd.sent == []


def test_no_session_never_connects(monkeypatch):
    monkeypatch.setattr(d, "_spot_session", None)
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("no bring-up"))
    assert d.gaze_at_bearing(0.0) is False


def test_never_raises(cmd, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rpc down")
    monkeypatch.setattr(cmd, "robot_command", boom)
    assert d.gaze_at_bearing(0.0) is False
