"""Operator decision 4 (2026-10-28): gaze = slow shoulder-yaw (sh0) joint move
of the folded talk pose toward the locked speaker. Fake command client only."""
import math

import pytest

import src.spot.features.gaze.handlers as gaze_h
import src.spot.robot as robot_pkg
import src.spot.robot.arm as arm_mod
import src.spot.robot.dispatcher as dispatcher
import time
import src.spot.features.gaze.handlers as gaze_h
from src.spot.robot import ARM, CONNECTION, NAV, SAFETY_GEN
import src.spot.robot.arm as arm_mod
import src.spot.robot.state as state_mod
import src.spot.core.tuning as tuning
from tests.voice_control.test_arm_policy import FakeCmd
import src.spot.conversation.loop as _loop


def _sh0(proto):
    jm = proto.synchronized_command.arm_command.arm_joint_move_command
    return math.degrees(jm.trajectory.points[0].position.sh0.value)


@pytest.fixture
def cmd(monkeypatch):
    c = FakeCmd()
    monkeypatch.setattr(CONNECTION, "_session", {"cmd": c, "robot": object()})
    monkeypatch.setattr(ARM, "_pose", "talk")
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(ARM, "_gaze_deg", 0.0)
    monkeypatch.setattr(gaze_h, "_gaze_t", 0.0)
    monkeypatch.setattr(ARM, "_settle_until", 0.0)
    monkeypatch.setattr(NAV, "is_navigating", lambda: False)
    monkeypatch.setattr(state_mod, "standing_and_powered", lambda s: True)
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(gaze_max_deg=45.0))
    clock = {"t": 1000.0}
    monkeypatch.setattr(time, "monotonic", lambda: clock["t"])
    c.clock = clock
    return c


def test_pure_clamp_and_wrap():
    assert gaze_h.gaze_sh0_deg(30.0, 45.0) == pytest.approx(30.0)
    assert gaze_h.gaze_sh0_deg(80.0, 45.0) == pytest.approx(45.0)
    assert gaze_h.gaze_sh0_deg(300.0, 45.0) == pytest.approx(-45.0)   # 300 = -60 (right)
    assert gaze_h.gaze_sh0_deg(350.0, 45.0) == pytest.approx(-10.0)
    assert gaze_h.gaze_sh0_deg(175.0, 45.0) is None                    # straight behind
    assert gaze_h.gaze_sh0_deg(float("nan"), 45.0) is None
    assert gaze_h.gaze_sh0_deg(10.0, 0.0) is None                      # gaze off


def test_default_limit_is_20():
    assert tuning.Tuning().gaze_max_deg == 20.0


def test_gaze_turns_sh0_slowly_in_joints_pose(cmd, monkeypatch):
    monkeypatch.setattr(arm_mod, "block_until_arm_arrives",
                        lambda *a: pytest.fail("gaze must not block"))
    assert gaze_h.gaze_at_bearing(30.0) is True
    (kind, proto), = cmd.sent
    assert kind == "arm_joints"
    assert _sh0(proto) == pytest.approx(30.0)
    traj = proto.synchronized_command.arm_command.arm_joint_move_command.trajectory
    pos = traj.points[0].position
    assert pos.sh1.value == pytest.approx(-2.6) and pos.el0.value == pytest.approx(2.6)
    assert pos.el1.value == pos.wr0.value == pos.wr1.value == 0.0
    assert traj.maximum_velocity.value == pytest.approx(1.0)   # ARM_MAX_ANGULAR_VEL


def test_clamped_to_tuning_limit(cmd, monkeypatch):
    assert gaze_h.gaze_at_bearing(90.0) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(45.0)
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(gaze_max_deg=20.0))
    cmd.clock["t"] += 3
    assert gaze_h.gaze_at_bearing(-90.0) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(-20.0)


def test_sign_tunable(cmd, monkeypatch):
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(gaze_max_deg=45.0, gaze_sh0_sign=-1))
    assert gaze_h.gaze_at_bearing(30.0) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(-30.0)


def test_rate_limited_by_change_and_time(cmd):
    assert gaze_h.gaze_at_bearing(10.0) is False          # <= 15 deg from centre
    assert gaze_h.gaze_at_bearing(30.0) is True
    cmd.clock["t"] += 3
    assert gaze_h.gaze_at_bearing(40.0) is False          # moved only 10 deg
    cmd.clock["t"] += 0.5
    assert gaze_h.gaze_at_bearing(-30.0) is True
    cmd.clock["t"] += 1.0                             # < 2 s since the last gaze
    assert gaze_h.gaze_at_bearing(30.0) is False
    cmd.clock["t"] += 1.5
    assert gaze_h.gaze_at_bearing(30.0) is True
    assert len(cmd.sent) == 3


def test_release_recentres(cmd):
    assert gaze_h.gaze_at_bearing(30.0) is True
    cmd.clock["t"] += 3
    assert gaze_h.gaze_at_bearing(None) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(0.0)
    cmd.clock["t"] += 3
    assert gaze_h.gaze_at_bearing(None) is False          # already centred


def test_behind_ignored(cmd):
    assert gaze_h.gaze_at_bearing(170.0) is False
    assert cmd.sent == []


@pytest.mark.parametrize("pose", ["stowed", "walk"])
def test_only_in_talk_pose(cmd, monkeypatch, pose):
    monkeypatch.setattr(ARM, "_pose", pose)
    assert gaze_h.gaze_at_bearing(30.0) is False
    assert gaze_h.gaze_at_bearing(None) is False
    assert cmd.sent == []


def test_not_while_walking_or_dancing(cmd, monkeypatch):
    monkeypatch.setattr(NAV, "is_navigating", lambda: True)
    assert gaze_h.gaze_at_bearing(30.0) is False
    assert cmd.sent == []


def test_not_unless_standing_and_powered(cmd, monkeypatch):
    monkeypatch.setattr(state_mod, "standing_and_powered", lambda s: False)
    assert gaze_h.gaze_at_bearing(30.0) is False
    assert cmd.sent == []


def test_gaze_off_when_limit_zero(cmd, monkeypatch):
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(gaze_max_deg=0.0))
    assert gaze_h.gaze_at_bearing(30.0) is False
    assert cmd.sent == []


def test_blocked_after_stop(cmd, monkeypatch):
    """A stop stows the arm (and bumps the safety gen): no gaze follows it."""
    monkeypatch.setattr(ARM, "close_mouth", lambda: None)
    monkeypatch.setattr(CONNECTION, "_context", object())
    gen = dispatcher.current_safety_gen()
    dispatcher.dispatch_intent({"intent": "stop", "params": {}})
    n = len(cmd.sent)
    cmd.clock["t"] += 10
    assert gaze_h.gaze_at_bearing(30.0) is False
    assert gaze_h.gaze_at_bearing(30.0, gen=gen) is False
    assert cmd.kinds()[n:] == []


def test_stale_gen_refused_under_lock(cmd):
    """A stop that landed after the caller's snapshot wins, even if the pose
    were somehow still "talk"."""
    stale = dispatcher.current_safety_gen() - 1
    assert gaze_h.gaze_at_bearing(30.0, gen=stale) is False
    assert cmd.sent == []


def test_no_gaze_while_a_pose_move_is_in_flight(cmd):
    """A fold / halt-resume waits on its own cmd id before enabling the mouth:
    a gaze must not supersede it mid-rise."""
    arm_mod._set_arm_pose("talk")                      # e.g. the auto fold just sent
    assert gaze_h.gaze_at_bearing(30.0) is False
    cmd.clock["t"] += 3.1                        # JOINT_MOVE_SECS + 0.5 later
    assert gaze_h.gaze_at_bearing(30.0) is True


def test_pose_send_resets_gaze_memory(cmd):
    assert gaze_h.gaze_at_bearing(30.0) is True
    arm_mod._set_arm_pose("stowed")
    assert robot_pkg.ARM._gaze_deg is None
    arm_mod._set_arm_pose("talk")
    assert robot_pkg.ARM._gaze_deg == 0.0


def test_no_session_never_connects(monkeypatch):
    monkeypatch.setattr(CONNECTION, "_session", None)
    monkeypatch.setattr(CONNECTION, "ensure", lambda *a, **k: pytest.fail("no bring-up"))
    assert gaze_h.gaze_at_bearing(30.0) is False


def test_never_raises(cmd, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("rpc down")
    monkeypatch.setattr(cmd, "robot_command", boom)
    assert gaze_h.gaze_at_bearing(30.0) is False


# ---------------------------------------------------------------- safety review
def test_hard_cap_60_even_if_tuning_says_more(cmd, monkeypatch):
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(gaze_max_deg=120.0))
    assert gaze_h.gaze_at_bearing(100.0) is True
    assert _sh0(cmd.sent[-1][1]) == pytest.approx(60.0)


def test_gaze_rpc_timeout_shorter_than_safety_lock(cmd, monkeypatch):
    seen = {}

    def rc(c, end_time_secs=None, **kw):
        seen.update(kw)
        return 1
    monkeypatch.setattr(cmd, "robot_command", rc)
    assert gaze_h.gaze_at_bearing(30.0) is True
    assert seen["timeout"] == pytest.approx(0.5) and seen["timeout"] < arm_mod.SAFETY_LOCK_TIMEOUT_S


def test_walk_start_holds_gaze(cmd):
    arm_mod._hold_gaze()
    assert gaze_h.gaze_at_bearing(30.0) is False
    cmd.clock["t"] += 2.6
    assert gaze_h.gaze_at_bearing(30.0) is True


class _AliveWorker:
    def is_alive(self):
        return True


def test_client_gaze_uses_gen_from_request_time(cmd, monkeypatch):
    """I1: a stop between the request and the worker pickup wins, even though
    the arm pose may still read "talk" while the stop's join runs."""
    import queue as _q
    import src.spot.conversation.loop as cm
    monkeypatch.delenv("SPOT_DRY_RUN", raising=False)
    q = _q.Queue(maxsize=1)
    monkeypatch.setattr(_loop, "_gaze_q", q)
    monkeypatch.setattr(_loop, "_gaze_worker", _AliveWorker())  # pick up by hand
    cm._gaze_at_speaker(30.0)
    body, gen = q.get_nowait()
    assert body == 30.0 and gen == dispatcher.current_safety_gen()
    monkeypatch.setattr(SAFETY_GEN, "_gen", robot_pkg.SAFETY_GEN._gen + 1)  # stop before pickup
    assert gaze_h.gaze_at_bearing(body, gen) is False
    assert cmd.sent == []


def test_client_gaze_skipped_without_gen(monkeypatch):
    import queue as _q
    import src.spot.conversation.loop as cm
    q = _q.Queue(maxsize=1)
    monkeypatch.setattr(_loop, "_gaze_q", q)
    monkeypatch.setattr(_loop, "_safety_gen_now", lambda: None)
    monkeypatch.setattr(_loop, "_gaze_worker", _AliveWorker())
    cm._gaze_at_speaker(30.0)
    assert q.empty()


def test_explicit_tuning_yaml_arm_keys():
    t = tuning.load_tuning()
    assert t.auto_arm is True and t.gaze_max_deg == 20.0 and t.gaze_sh0_sign == 1


def test_gaze_blocked_once_shutdown_stops_worker(cmd, monkeypatch):
    """h3: a request the worker already picked up cannot move the arm once
    shutdown has set the gaze stop Event (re-checked under ARM._lock)."""
    import threading as _th
    ev = _th.Event()
    monkeypatch.setattr(_loop, "_gaze_stop", ev)
    ev.set()
    assert gaze_h.gaze_at_bearing(30.0) is False
    assert cmd.sent == []
    ev.clear()
    assert gaze_h.gaze_at_bearing(30.0) is True
