"""Operator decision 3 (2026-10-28): the arm folds into the joints pose and the
gripper mouth turns on automatically once Spot is standing + powered."""
from types import SimpleNamespace

import pytest

import src.session as session_mod
import src.voice_control.spot_dispatch as d
import src.voice_control.tuning as tuning
from src.voice_control.client_mic import auto_arm_wanted
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
    import src.voice_control.spot_tts as tts_mod
    monkeypatch.setattr(tts_mod, "get_tts", lambda: type("T", (), {"mouth": m})())
    monkeypatch.setattr(d, "_spot_session", sess)
    monkeypatch.setattr(d, "_session_context", object())
    monkeypatch.setattr(d, "_closing", False)
    monkeypatch.setattr(d, "block_until_arm_arrives", lambda c, i, t: True)
    monkeypatch.setattr(d, "_close_mouth", m.disable)
    monkeypatch.setattr(d, "_arm_pose", "stowed")
    monkeypatch.setattr(d, "_pose_before_walk", None)
    monkeypatch.setattr(d, "_pose_during_walk", None)
    monkeypatch.setattr(d, "_mouth_resume", False)
    monkeypatch.setattr(d, "_talk_resume", False)
    monkeypatch.setattr(d, "_last_brain", None)
    monkeypatch.setattr(d, "_auto_arm", False)
    monkeypatch.setattr(d, "_auto_arm_pending", False)
    monkeypatch.setattr(d, "_auto_arm_thread", None)
    state = SimpleNamespace(standing=True)
    monkeypatch.setattr(d, "_standing_and_powered", lambda s: state.standing)
    monkeypatch.setattr(d.time, "sleep", lambda s: None)
    yield SimpleNamespace(cmd=cmd, mouth=m, state=state, sess=sess)
    d._cancel_nav()


def _owe():
    """Startup state with auto-arm on (without set_auto_arm's background fold)."""
    d._auto_arm = d._auto_arm_pending = True


def _join_auto():
    t = d._auto_arm_thread
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
    d.set_auto_arm(True)          # e.g. --map brought the robot up at launch
    _join_auto()
    assert env.cmd.kinds() == ["arm_joints"] and env.mouth.enabled


def test_auto_folds_and_enables_mouth_when_standing(env):
    _owe()
    assert d.auto_arm_if_ready(d.current_safety_gen()) is True
    assert env.cmd.kinds() == ["arm_joints"]
    assert env.mouth.enabled and d.get_arm_pose() == "talk"
    assert d._auto_arm_pending is False
    assert d.auto_arm_if_ready(d.current_safety_gen()) is False   # one-shot


def test_auto_waits_for_standing_and_powered(env):
    _owe()
    env.state.standing = False
    assert d.auto_arm_if_ready(d.current_safety_gen()) is False
    assert env.cmd.sent == [] and not env.mouth.enabled
    assert d._auto_arm_pending is True            # still owed
    env.state.standing = True
    assert d.auto_arm_if_ready(d.current_safety_gen()) is True


def test_auto_never_without_session(env, monkeypatch):
    _owe()
    monkeypatch.setattr(d, "_spot_session", None)
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("no bring-up"))
    assert d.auto_arm_if_ready(d.current_safety_gen()) is False
    assert d.start_auto_arm(d.current_safety_gen()) is None


def test_opt_out_never_moves_the_arm(env):
    d.set_auto_arm(False)
    assert d.auto_arm_if_ready(d.current_safety_gen()) is False
    d.dispatch_intent({"intent": "stand", "params": {}})
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
    d._auto_arm_pending = False                  # e.g. a sit cleared the fold
    assert d.dispatch_intent({"intent": "stand", "params": {}}) is True
    _join_auto()
    assert env.cmd.kinds() == ["mobility", "arm_joints"]
    assert env.mouth.enabled and d.get_arm_pose() == "talk"


def test_stand_gives_up_if_never_standing(env):
    _owe()
    env.state.standing = False
    d.dispatch_intent({"intent": "stand", "params": {}})
    _join_auto()
    assert env.cmd.kinds() == ["mobility"] and not env.mouth.enabled


def test_stop_since_gen_blocks_auto_fold(env):
    _owe()
    gen = d.current_safety_gen()
    d.dispatch_intent({"intent": "stop", "params": {}})
    n = len(env.cmd.sent)
    assert d.auto_arm_if_ready(gen) is False
    assert "arm_joints" not in env.cmd.kinds()[n:] and not env.mouth.enabled


def test_sit_during_stand_wait_blocks_auto_fold(env):
    """stand -> sit race: the sit bumps the safety gen, so no fold follows it."""
    _owe()
    gen = d.current_safety_gen()
    d.dispatch_intent({"intent": "sit", "params": {}})
    assert d.auto_arm_if_ready(gen) is False
    assert "arm_joints" not in env.cmd.kinds()


def test_estop_cancels_pending_auto_fold(env):
    _owe()
    d.dispatch_intent({"intent": "estop", "params": {}})
    assert d._auto_arm_pending is False
    assert d.auto_arm_if_ready(d.current_safety_gen()) is False
    assert "arm_joints" not in env.cmd.kinds()


def test_not_while_moving(env, monkeypatch):
    _owe()
    monkeypatch.setattr(d, "is_navigating", lambda: True)
    assert d.auto_arm_if_ready(d.current_safety_gen()) is False
    assert env.cmd.sent == [] and d._auto_arm_pending is True


def test_halt_resume_owns_the_arm(env, monkeypatch):
    _owe()
    monkeypatch.setattr(d, "_talk_resume", True)
    assert d.auto_arm_if_ready(d.current_safety_gen()) is False
    assert env.cmd.sent == []


def test_disable_mouth_opts_out_until_enable(env):
    _owe()
    assert d.auto_arm_if_ready(d.current_safety_gen()) is True
    d.dispatch_intent({"intent": "disable_mouth", "params": {}})
    assert not env.mouth.enabled and d.get_arm_pose() == "stowed"
    n = len(env.cmd.sent)
    d.dispatch_intent({"intent": "stand", "params": {}})
    _join_auto()
    assert env.cmd.kinds()[n:] == ["mobility"]    # no automatic re-fold
    assert d.dispatch_intent({"intent": "enable_mouth"}) is True
    assert env.mouth.enabled and d._auto_arm is True


def test_enable_mouth_refused_unless_standing(env):
    env.state.standing = False
    assert d.dispatch_intent({"intent": "enable_mouth"}) is False
    assert env.cmd.sent == [] and not env.mouth.enabled


def test_walk_after_auto_fold_keeps_mouth_on(env):
    _owe()
    assert d.auto_arm_if_ready(d.current_safety_gen()) is True
    n = len(env.cmd.sent)
    d.dispatch_intent({"intent": "walk", "params": {}})
    t = d._nav_thread
    if t is not None:
        t.join(timeout=5)
    assert env.cmd.kinds()[n:] == ["mobility"]
    assert env.mouth.enabled and env.mouth.log == ["on"]
