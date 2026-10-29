"""Operator decision 4 (2026-10-28): gaze = slow shoulder-yaw (sh0) joint move
of the folded talk pose toward the locked speaker. Fake command client only."""
import math

import pytest

import src.voice_control.spot_dispatch as d
import src.voice_control.tuning as tuning
from tests.voice_control.test_arm_policy import FakeCmd


def _sh0(proto):
    jm = proto.synchronized_command.arm_command.arm_joint_move_command
    return math.degrees(jm.trajectory.points[0].position.sh0.value)


@pytest.fixture
def cmd(monkeypatch):
    c = FakeCmd()
    monkeypatch.setattr(d, "_spot_session", {"cmd": c, "robot": object()})
    monkeypatch.setattr(d, "_arm_pose", "talk")
    monkeypatch.setattr(d, "_closing", False)
    monkeypatch.setattr(d, "_gaze_deg", 0.0)
    monkeypatch.setattr(d, "_gaze_t", 0.0)
    monkeypatch.setattr(d, "_arm_settle_until", 0.0)
    monkeypatch.setattr(d, "is_navigating", lambda: False)
    monkeypatch.setattr(d, "_standing_and_powered", lambda s: True)
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning())
    clock = {"t": 1000.0}
    monkeypatch.setattr(d.time, "monotonic", lambda: clock["t"])
    c.clock = clock
    return c


def test_pure_clamp_and_wrap():
    assert d.gaze_sh0_deg(30.0, 45.0) == pytest.approx(30.0)
    assert d.gaze_sh0_deg(80.0, 45.0) == pytest.approx(45.0)
    assert d.gaze_sh0_deg(300.0, 45.0) == pytest.approx(-45.0)   # 300 = -60 (right)
    assert d.gaze_sh0_deg(350.0, 45.0) == pytest.approx(-10.0)
    assert d.gaze_sh0_deg(175.0, 45.0) is None                    # straight behind
    assert d.gaze_sh0_deg(float("nan"), 45.0) is None
    assert d.gaze_sh0_deg(10.0, 0.0) is None                      # gaze off


def test_default_limit_is_45():
    assert tuning.Tuning().gaze_max_deg == 45.0


def test_gaze_turns_sh0_slowly_in_joints_pose(cmd, monkeypatch):
    monkeypatch.setattr(d, "block_until_arm_arrives",
                        lambda *a: pytest.fail("gaze must not block"))
    assert d.gaze_at_bearing(30.0) is True
    (kind, proto), = cmd.sent
    assert kind == "arm_joints"
    assert _sh0(proto) == pytest.approx(30.0)
    traj = proto.synchronized_command.arm_command.arm_joint_move_command.trajectory
    pos = traj.points[0].position
    assert pos.sh1.value == pytest.approx(-2.6) and pos.el0.value == pytest.approx(2.6)
    assert pos.el1.value == pos.wr0.value == pos.wr1.value == 0.0
    assert traj.maximum_velocity.value == pytest.approx(1.0)   # ARM_MAX_ANGULAR_VEL


def test_clamped_to_tuning_limit(cmd, monkeypatch):
    assert d.gaze_at_bearing(90.0) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(45.0)
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(gaze_max_deg=20.0))
    cmd.clock["t"] += 3
    assert d.gaze_at_bearing(-90.0) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(-20.0)


def test_sign_tunable(cmd, monkeypatch):
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(gaze_sh0_sign=-1))
    assert d.gaze_at_bearing(30.0) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(-30.0)


def test_rate_limited_by_change_and_time(cmd):
    assert d.gaze_at_bearing(10.0) is False          # <= 15 deg from centre
    assert d.gaze_at_bearing(30.0) is True
    cmd.clock["t"] += 3
    assert d.gaze_at_bearing(40.0) is False          # moved only 10 deg
    cmd.clock["t"] += 0.5
    assert d.gaze_at_bearing(-30.0) is True
    cmd.clock["t"] += 1.0                             # < 2 s since the last gaze
    assert d.gaze_at_bearing(30.0) is False
    cmd.clock["t"] += 1.5
    assert d.gaze_at_bearing(30.0) is True
    assert len(cmd.sent) == 3


def test_release_recentres(cmd):
    assert d.gaze_at_bearing(30.0) is True
    cmd.clock["t"] += 3
    assert d.gaze_at_bearing(None) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(0.0)
    cmd.clock["t"] += 3
    assert d.gaze_at_bearing(None) is False          # already centred


def test_behind_ignored(cmd):
    assert d.gaze_at_bearing(170.0) is False
    assert cmd.sent == []


@pytest.mark.parametrize("pose", ["stowed", "walk"])
def test_only_in_talk_pose(cmd, monkeypatch, pose):
    monkeypatch.setattr(d, "_arm_pose", pose)
    assert d.gaze_at_bearing(30.0) is False
    assert d.gaze_at_bearing(None) is False
    assert cmd.sent == []


def test_not_while_walking_or_dancing(cmd, monkeypatch):
    monkeypatch.setattr(d, "is_navigating", lambda: True)
    assert d.gaze_at_bearing(30.0) is False
    assert cmd.sent == []


def test_not_unless_standing_and_powered(cmd, monkeypatch):
    monkeypatch.setattr(d, "_standing_and_powered", lambda s: False)
    assert d.gaze_at_bearing(30.0) is False
    assert cmd.sent == []


def test_gaze_off_when_limit_zero(cmd, monkeypatch):
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(gaze_max_deg=0.0))
    assert d.gaze_at_bearing(30.0) is False
    assert cmd.sent == []


def test_blocked_after_stop(cmd, monkeypatch):
    """A stop stows the arm (and bumps the safety gen): no gaze follows it."""
    monkeypatch.setattr(d, "_close_mouth", lambda: None)
    monkeypatch.setattr(d, "_session_context", object())
    gen = d.current_safety_gen()
    d.dispatch_intent({"intent": "stop", "params": {}})
    n = len(cmd.sent)
    cmd.clock["t"] += 10
    assert d.gaze_at_bearing(30.0) is False
    assert d.gaze_at_bearing(30.0, gen=gen) is False
    assert cmd.kinds()[n:] == []


def test_stale_gen_refused_under_lock(cmd):
    """A stop that landed after the caller's snapshot wins, even if the pose
    were somehow still "talk"."""
    stale = d.current_safety_gen() - 1
    assert d.gaze_at_bearing(30.0, gen=stale) is False
    assert cmd.sent == []


def test_no_gaze_while_a_pose_move_is_in_flight(cmd):
    """A fold / halt-resume waits on its own cmd id before enabling the mouth:
    a gaze must not supersede it mid-rise."""
    d._set_arm_pose("talk")                      # e.g. the auto fold just sent
    assert d.gaze_at_bearing(30.0) is False
    cmd.clock["t"] += 3.1                        # JOINT_MOVE_SECS + 0.5 later
    assert d.gaze_at_bearing(30.0) is True


def test_pose_send_resets_gaze_memory(cmd):
    assert d.gaze_at_bearing(30.0) is True
    d._set_arm_pose("stowed")
    assert d._gaze_deg is None
    d._set_arm_pose("talk")
    assert d._gaze_deg == 0.0


def test_no_session_never_connects(monkeypatch):
    monkeypatch.setattr(d, "_spot_session", None)
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("no bring-up"))
    assert d.gaze_at_bearing(30.0) is False


def test_never_raises(cmd, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rpc down")
    monkeypatch.setattr(cmd, "robot_command", boom)
    assert d.gaze_at_bearing(30.0) is False
