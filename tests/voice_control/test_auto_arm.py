"""Operator decision 3 (2026-10-28): the arm folds into the joints pose and the
gripper mouth turns on automatically once Spot is standing + powered."""
from types import SimpleNamespace

import pytest

import src.spot.robot.session as session_mod
import src.spot.features.mouth.handlers as mouth_h
import src.spot.nav.runner as nav_runner
import src.spot.robot as robot_pkg
import src.spot.robot.arm as arm_mod
import src.spot.robot.connection as conn_mod
import src.spot.robot.dispatcher as dispatcher
import time
import src.spot.features.mouth.handlers as mouth_h
from src.spot.robot import ARM, CONNECTION, NAV
import src.spot.robot.arm as arm_mod
import src.spot.robot.state as state_mod
import src.spot.core.tuning as tuning
from src.spot.app.main import auto_arm_wanted
from tests.voice_control.test_arm_policy import FakeCmd, FakeState


class _Mouth:
    def __init__(self):
        self.enabled, self.log = False, []

    def disable(self):
        self.enabled = False
        self.log.append("off")

    def enable(self):
        self.enabled = True
        self.log.append("on")


class _Robot:
    def has_arm(self):
        return True

    def is_powered_on(self):
        return True


@pytest.fixture
def env(monkeypatch):
    cmd = FakeCmd()
    sess = {"cmd": cmd, "state": FakeState(), "robot": _Robot()}
    m = _Mouth()
    import src.spot.tts.speaker as tts_mod
    monkeypatch.setattr(tts_mod, "get_tts", lambda: type("T", (), {"mouth": m})())
    monkeypatch.setattr(CONNECTION, "_session", sess)
    monkeypatch.setattr(CONNECTION, "_context", object())
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(arm_mod, "block_until_arm_arrives", lambda c, i, t: True)
    monkeypatch.setattr(ARM, "close_mouth", m.disable)
    monkeypatch.setattr(ARM, "_pose", "stowed")
    monkeypatch.setattr(ARM, "_pose_before_walk", None)
    monkeypatch.setattr(ARM, "_pose_during_walk", None)
    monkeypatch.setattr(ARM, "_mouth_resume", False)
    monkeypatch.setattr(ARM, "_talk_resume", False)
    monkeypatch.setattr(ARM, "_last_brain", None)
    monkeypatch.setattr(mouth_h, "_auto_arm", False)
    monkeypatch.setattr(mouth_h, "_auto_arm_pending", False)
    monkeypatch.setattr(mouth_h, "_auto_arm_thread", None)
    state = SimpleNamespace(standing=True)
    monkeypatch.setattr(state_mod, "standing_and_powered", lambda s: state.standing)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    yield SimpleNamespace(cmd=cmd, mouth=m, state=state, sess=sess)
    nav_runner._cancel_nav()


def _owe():
    """Startup state with auto-arm on (without set_auto_arm's background fold)."""
    mouth_h._auto_arm = mouth_h._auto_arm_pending = True


def _join_auto():
    t = mouth_h._auto_arm_thread
    if t is not None:
        t.join(timeout=5)


def test_talk_and_walk_are_the_same_joints_pose():
    from tests.voice_control.test_arm_policy import _kind
    c = FakeCmd()
    session_mod.send_arm_pose(c, "talk")
    session_mod.send_arm_pose(c, "walk")
    assert c.kinds() == ["arm_joints", "arm_joints"]
    a, b = (p.synchronized_command.arm_command.arm_joint_move_command.trajectory.points[0].position
            for _, p in c.sent)
    assert a == b
    assert a.sh0.value == 0.0 and a.sh1.value == pytest.approx(-2.6)
    assert a.el0.value == pytest.approx(2.6)
    assert _kind(session_mod.joints_pose_command(0.3)) == "arm_joints"


def test_set_auto_arm_with_live_session_folds_in_background(env):
    mouth_h.set_auto_arm(True)          # e.g. --map brought the robot up at launch
    _join_auto()
    assert env.cmd.kinds() == ["arm_joints"] and env.mouth.enabled


def test_auto_folds_and_enables_mouth_when_standing(env):
    _owe()
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is True
    assert env.cmd.kinds() == ["arm_joints"]
    assert env.mouth.enabled and arm_mod.get_arm_pose() == "talk"
    assert mouth_h._auto_arm_pending is False
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is False   # one-shot


def test_auto_waits_for_standing_and_powered(env):
    _owe()
    env.state.standing = False
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is False
    assert env.cmd.sent == [] and not env.mouth.enabled
    assert mouth_h._auto_arm_pending is True            # still owed
    env.state.standing = True
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is True


def test_auto_never_without_session(env, monkeypatch):
    _owe()
    monkeypatch.setattr(CONNECTION, "_session", None)
    monkeypatch.setattr(CONNECTION, "ensure", lambda *a, **k: pytest.fail("no bring-up"))
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is False
    assert mouth_h.start_auto_arm(dispatcher.current_safety_gen()) is None


def test_opt_out_never_moves_the_arm(env):
    mouth_h.set_auto_arm(False)
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is False
    dispatcher.dispatch_intent({"intent": "stand", "params": {}})
    _join_auto()
    assert env.cmd.kinds() == ["mobility"]       # the stand only
    assert not env.mouth.enabled


def test_auto_arm_wanted_flags():
    t = tuning.Tuning()
    assert auto_arm_wanted(SimpleNamespace(no_arm=False, dry_run=False), t) is True
    assert auto_arm_wanted(SimpleNamespace(no_arm=True, dry_run=False), t) is False
    assert auto_arm_wanted(SimpleNamespace(no_arm=False, dry_run=True), t) is False
    assert auto_arm_wanted(SimpleNamespace(no_arm=False, dry_run=False),
                           tuning.Tuning(auto_arm=False)) is False


def test_stand_folds_arm_after_standing(env):
    _owe()
    mouth_h._auto_arm_pending = False            # e.g. a sit cleared the fold
    assert dispatcher.dispatch_intent({"intent": "stand", "params": {}}) is True
    _join_auto()
    assert env.cmd.kinds() == ["mobility", "arm_joints"]
    assert env.mouth.enabled and arm_mod.get_arm_pose() == "talk"


def test_stand_gives_up_if_never_standing(env):
    _owe()
    env.state.standing = False
    dispatcher.dispatch_intent({"intent": "stand", "params": {}})
    _join_auto()
    assert env.cmd.kinds() == ["mobility"] and not env.mouth.enabled


def test_stop_since_gen_blocks_auto_fold(env):
    _owe()
    gen = dispatcher.current_safety_gen()
    dispatcher.dispatch_intent({"intent": "stop", "params": {}})
    n = len(env.cmd.sent)
    assert mouth_h.auto_arm_if_ready(gen) is False
    assert "arm_joints" not in env.cmd.kinds()[n:] and not env.mouth.enabled


def test_sit_during_stand_wait_blocks_auto_fold(env):
    """stand -> sit race: the sit bumps the safety gen, so no fold follows it."""
    _owe()
    gen = dispatcher.current_safety_gen()
    dispatcher.dispatch_intent({"intent": "sit", "params": {}})
    assert mouth_h.auto_arm_if_ready(gen) is False
    assert "arm_joints" not in env.cmd.kinds()


def test_estop_cancels_pending_auto_fold(env):
    _owe()
    dispatcher.dispatch_intent({"intent": "estop", "params": {}})
    assert mouth_h._auto_arm_pending is False
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is False
    assert "arm_joints" not in env.cmd.kinds()


def test_not_while_moving(env, monkeypatch):
    _owe()
    monkeypatch.setattr(NAV, "is_navigating", lambda: True)
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is False
    assert env.cmd.sent == [] and mouth_h._auto_arm_pending is True


def test_halt_resume_owns_the_arm(env, monkeypatch):
    _owe()
    monkeypatch.setattr(ARM, "_talk_resume", True)
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is False
    assert env.cmd.sent == []


def test_disable_mouth_opts_out_until_enable(env):
    _owe()
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is True
    dispatcher.dispatch_intent({"intent": "disable_mouth", "params": {}})
    assert not env.mouth.enabled and arm_mod.get_arm_pose() == "stowed"
    n = len(env.cmd.sent)
    dispatcher.dispatch_intent({"intent": "stand", "params": {}})
    _join_auto()
    assert env.cmd.kinds()[n:] == ["mobility"]    # no automatic re-fold
    assert dispatcher.dispatch_intent({"intent": "enable_mouth"}) is True
    assert env.mouth.enabled and mouth_h._auto_arm is True


def test_enable_mouth_refused_unless_standing(env):
    env.state.standing = False
    assert dispatcher.dispatch_intent({"intent": "enable_mouth"}) is False
    assert env.cmd.sent == [] and not env.mouth.enabled


def test_walk_after_auto_fold_keeps_mouth_on(env):
    _owe()
    assert mouth_h.auto_arm_if_ready(dispatcher.current_safety_gen()) is True
    n = len(env.cmd.sent)
    dispatcher.dispatch_intent({"intent": "walk", "params": {}})
    t = robot_pkg.NAV._thread
    if t is not None:
        t.join(timeout=5)
    assert env.cmd.kinds()[n:] == ["mobility"]
    assert env.mouth.enabled and env.mouth.log == ["on"]


def test_sit_and_power_off_cancel_pending_fold(env):
    for name in ("sit", "power_off"):
        _owe()
        dispatcher.dispatch_intent({"intent": name, "params": {}})
        assert mouth_h._auto_arm_pending is False


def test_close_session_cancels_pending_fold(env, monkeypatch):
    import bosdyn.client.robot_command as rc
    monkeypatch.setattr(rc, "blocking_sit", lambda c, timeout_sec=15: None)

    class Ctx:
        def __exit__(self, *a):
            pass

    class R(_Robot):
        def power_off(self, **k):
            pass
    env.sess["robot"] = R()
    monkeypatch.setattr(CONNECTION, "_context", Ctx())
    _owe()
    conn_mod.close_spot_session()
    assert mouth_h._auto_arm_pending is False
