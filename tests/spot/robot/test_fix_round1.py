"""Task 9 fix round 1 (safety review C1, C2, R18). Fakes only, no robot."""
import time

import pytest

import src.session as session_mod
import src.spot.robot as robot_pkg
import src.spot.robot.arm as arm_mod
import src.spot.robot.dispatcher as dispatcher
import src.spot.robot.motion as motion
import src.voice_control.spot_dispatch as d
import src.spot.features.mouth.handlers as mouth_h
import src.voice_control.spot_tts as tts_mod
from src.spot.core.registry import ActionSpec
from src.spot.features import REGISTRY
from src.spot.robot import ARM, CONNECTION, NAV, SAFETY_GEN
from tests.voice_control.test_arm_policy import FakeCmd, FakeState


# ------------------------------------------------- C1: one owner lookup
class _FakeNav:
    _mode = "fake-mode"

    def __init__(self):
        self.calls = []

    def cancel(self):
        self.calls.append("cancel")

    def spawn(self, fn, mode=None, gen=None):
        self.calls.append(("spawn", mode))
        return True

    def is_navigating(self):
        return False


def test_rebinding_nav_reaches_every_path(monkeypatch):
    fake = _FakeNav()
    monkeypatch.setattr(robot_pkg, "NAV", fake)
    cmd = FakeCmd()
    monkeypatch.setattr(CONNECTION, "_session", {"cmd": cmd, "state": FakeState(), "robot": object()})
    monkeypatch.setattr(ARM, "_pose", "stowed")
    monkeypatch.setattr(ARM, "close_mouth", lambda: None)
    monkeypatch.setattr(mouth_h, "_auto_arm_pending", False)

    dispatcher.dispatch_safety({"intent": "stop", "params": {}})   # stop branch cancels nav
    motion.abort_motion(cmd, "walk")                               # abort path cancels nav
    motion.start_restore_watch(cmd, 1, time.time() + 1)            # watch registers on nav
    assert fake.calls == ["cancel", "cancel", ("spawn", "walk")]
    assert d.is_navigating() is False and d._nav_mode == "fake-mode"
    assert NAV is not fake  # the real runner was never touched


# ------------------------------------------------- C2: registry catch-all
@pytest.fixture
def walk_env(monkeypatch):
    class BadState(FakeState):
        def get_robot_state(self):
            raise RuntimeError("state rpc down")
    cmd = FakeCmd()
    monkeypatch.setattr(session_mod, "WALK_POSE", "stow")   # talk != walk: the walk tucks
    monkeypatch.setattr(CONNECTION, "_session", {"cmd": cmd, "state": BadState(), "robot": object()})
    monkeypatch.setattr(CONNECTION, "_context", object())
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(arm_mod, "block_until_arm_arrives", lambda c, i, t: True)
    monkeypatch.setattr(tts_mod, "get_tts", lambda: None)
    monkeypatch.setattr(ARM, "_pose", "talk")
    monkeypatch.setattr(ARM, "_pose_before_walk", None)
    monkeypatch.setattr(ARM, "_pose_during_walk", None)
    monkeypatch.setattr(ARM, "_last_brain", None)
    yield cmd
    NAV.cancel()


def _walk(cmd):
    n = len(cmd.sent)
    ok = d.dispatch_intent({"intent": "walk", "params": {"direction": "forward", "distance": 1}})
    return ok, cmd.kinds()[n:], ARM._pose, ARM._pose_before_walk, ARM._pose_during_walk


def test_raising_registered_feature_returns_false_and_restores_like_legacy(walk_env, monkeypatch):
    legacy = _walk(walk_env)          # base path: the legacy walk branch fails (state rpc down)
    monkeypatch.setattr(ARM, "_pose", "talk")

    def boom(params, ctx):
        raise RuntimeError("feature handler crashed")
    spec = ActionSpec(name="walk", handler=boom, grammar_params=None, prompt_line=None,
                      locomotion=True)
    monkeypatch.setattr(REGISTRY, "_actions", {"walk": spec})
    registered = _walk(walk_env)
    assert registered == legacy
    assert registered[0] is False
    assert registered[2] == "talk" and registered[3] is None   # tucked, then restored


# ------------------------------------------------- R18: close() halts anyway
def test_close_still_stops_when_auto_arm_cancel_raises(monkeypatch):
    import bosdyn.client.robot_command as rc

    def boom():
        raise RuntimeError("feature module broken")

    class Ctx:
        def __exit__(self, *a):
            pass

    class R:
        def power_off(self, **k):
            pass
    cmd = FakeCmd()
    monkeypatch.setattr(d, "_cancel_auto_arm", boom)
    monkeypatch.setattr(rc, "blocking_sit", lambda c, timeout_sec=15: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(ARM, "close_mouth", lambda: None)
    monkeypatch.setattr(ARM, "_pose", "stowed")
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(CONNECTION, "_session", {"cmd": cmd, "state": FakeState(), "robot": R()})
    monkeypatch.setattr(CONNECTION, "_context", Ctx())
    monkeypatch.setattr(SAFETY_GEN, "_gen", SAFETY_GEN.current())
    CONNECTION.close()
    assert cmd.kinds()[0] == "stop"
    assert CONNECTION._session is None
