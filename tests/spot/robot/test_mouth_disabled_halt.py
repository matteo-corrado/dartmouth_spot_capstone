"""Task 19: the core reaches the mouth callbacks (_cancel_auto_arm,
_handle_disable_mouth, _auto_arm_after_stand, start_auto_arm) through
src.spot.robot.mouth_handlers, bound once at import. With "mouth" left out of
ENABLED the module is still bound, so estop / stow / close behave exactly as
with it enabled and never raise. Fakes only, no robot."""
import time

import pytest

import src.spot.features as features_pkg
import src.spot.features.mouth.handlers as mouth_h
import src.spot.robot as robot_pkg
from src.spot.core.registry import Registry
from src.spot.features.enabled import ENABLED
from src.spot.robot import ARM, CONNECTION, SAFETY_GEN
from tests.spot.robot.test_stop_order import EXPECTED, _run
from tests.voice_control.test_arm_policy import FakeCmd, FakeState


@pytest.fixture
def no_mouth(monkeypatch):
    reg = Registry()
    reg.load([n for n in ENABLED if n != "mouth"])
    monkeypatch.setattr(features_pkg, "REGISTRY", reg)
    assert reg.action("disable_mouth") is None and reg.action("enable_mouth") is None
    return reg


def test_core_binds_the_mouth_module_once():
    assert robot_pkg.mouth_handlers is mouth_h
    assert not hasattr(robot_pkg, "legacy_features")


@pytest.mark.parametrize("name", ["stop", "freeze", "estop"])
def test_halt_with_mouth_disabled_is_unchanged(no_mouth, monkeypatch, name):
    out = _run(monkeypatch, "talk_walking_standing", name)
    assert out == EXPECTED[f"talk_walking_standing/{name}"]


def test_stow_with_mouth_disabled(no_mouth, monkeypatch):
    import src.spot.robot.arm as arm_mod
    import src.spot.tts.speaker as tts_mod
    cmd = FakeCmd()
    monkeypatch.setattr(tts_mod, "get_tts", lambda: None)
    monkeypatch.setattr(arm_mod, "block_until_arm_arrives", lambda *a: True)
    monkeypatch.setattr(CONNECTION, "_session", {"cmd": cmd, "state": FakeState(), "robot": object()})
    monkeypatch.setattr(CONNECTION, "_context", object())
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(ARM, "_pose", "talk")
    monkeypatch.setattr(ARM, "_pose_before_walk", None)
    monkeypatch.setattr(ARM, "_pose_during_walk", None)
    monkeypatch.setattr(ARM, "_last_brain", None)
    monkeypatch.setattr(mouth_h, "_auto_arm", True)
    monkeypatch.setattr(mouth_h, "_auto_arm_pending", True)
    assert robot_pkg.dispatcher.dispatch_intent({"intent": "stow", "params": {}}) is True
    assert ARM._pose == "stowed"
    assert mouth_h._auto_arm is False and mouth_h._auto_arm_pending is False


def test_close_with_mouth_disabled(no_mouth, monkeypatch):
    import bosdyn.client.robot_command as rc

    class Ctx:
        def __exit__(self, *a):
            pass

    class R:
        def power_off(self, **k):
            pass
    cmd = FakeCmd()
    monkeypatch.setattr(rc, "blocking_sit", lambda c, timeout_sec=15: None)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(ARM, "close_mouth", lambda: None)
    monkeypatch.setattr(ARM, "_pose", "stowed")
    monkeypatch.setattr(mouth_h, "_auto_arm_pending", True)
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(CONNECTION, "_session", {"cmd": cmd, "state": FakeState(), "robot": R()})
    monkeypatch.setattr(CONNECTION, "_context", Ctx())
    monkeypatch.setattr(SAFETY_GEN, "_gen", SAFETY_GEN.current())
    CONNECTION.close()
    assert cmd.kinds()[0] == "stop"
    assert mouth_h._auto_arm_pending is False
    assert CONNECTION._session is None


# ---- fix round 1: a mouth import failure must not take down src.spot.robot ----

def test_mouth_import_failure_still_imports_robot_and_logs():
    import os
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    code = ("import sys; sys.modules['src.spot.features.mouth.handlers'] = None; "
            "import src.spot.robot as r; "
            "print('BOUND', type(r.mouth_handlers).__name__); "
            "print('CALLS', r.mouth_handlers._cancel_auto_arm(), r.mouth_handlers._auto_arm_after_stand(), "
            "r.mouth_handlers.start_auto_arm(0, wait_s=r.mouth_handlers.AUTO_ARM_STAND_WAIT_S), "
            "r.mouth_handlers._handle_disable_mouth({}, None))")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(root), capture_output=True,
                         text=True, timeout=120, env=dict(os.environ, SPOT_DRY_RUN="1"))
    assert out.returncode == 0, out.stderr[-2000:]
    assert "MOUTH HANDLERS FAILED TO IMPORT" in out.stdout
    assert "BOUND _NoMouth" in out.stdout
    assert "CALLS None None None False" in out.stdout


@pytest.mark.parametrize("name", ["stop", "freeze", "estop"])
def test_halt_with_mouth_stand_in_still_stops(monkeypatch, name):
    monkeypatch.setattr(robot_pkg, "mouth_handlers", robot_pkg._NoMouth("ImportError: boom"))
    out = _run(monkeypatch, "talk_walking_standing", name)
    assert out["calls"] == EXPECTED[f"talk_walking_standing/{name}"]["calls"]
    assert out["final"]["ok"] is True


def test_stow_with_mouth_stand_in_still_stops(monkeypatch, capsys):
    monkeypatch.setattr(robot_pkg, "mouth_handlers", robot_pkg._NoMouth("ImportError: boom"))
    cmd = FakeCmd()
    monkeypatch.setattr(CONNECTION, "_session", {"cmd": cmd, "state": FakeState(), "robot": object()})
    monkeypatch.setattr(CONNECTION, "_context", object())
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(robot_pkg.NAV, "is_navigating", lambda: True)
    monkeypatch.setattr(robot_pkg.NAV, "_mode", "walk")
    monkeypatch.setattr(robot_pkg.NAV, "cancel", lambda *a, **k: None)
    assert robot_pkg.dispatcher.dispatch_intent({"intent": "stow", "params": {}}) is False
    assert "stop" in cmd.kinds()
    assert "mouth handlers unavailable" in capsys.readouterr().out


def test_close_with_mouth_stand_in_still_stops(monkeypatch):
    import bosdyn.client.robot_command as rc
    monkeypatch.setattr(robot_pkg, "mouth_handlers", robot_pkg._NoMouth("ImportError: boom"))

    class Ctx:
        def __exit__(self, *a):
            pass

    class R:
        def power_off(self, **k):
            pass
    cmd = FakeCmd()
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
