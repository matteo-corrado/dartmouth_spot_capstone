"""I2: removing "mouth" from enabled.py really turns the mouth off. Standing
never folds the arm and the jaw never runs; stow still stows the arm."""
import pytest

import src.spot.features as _features
import src.spot.features.mouth.handlers as mouth_h
import src.spot.robot.dispatcher as dispatcher
from src.spot.core.registry import Registry
from src.spot.features.enabled import ENABLED
from tests.voice_control.test_auto_arm import _join_auto, env  # noqa: F401  (fixture)


@pytest.fixture
def no_mouth(monkeypatch):
    reg = Registry()
    assert reg.load([n for n in ENABLED if n != "mouth"]) == []
    assert reg.action("enable_mouth") is None
    monkeypatch.setattr(_features, "REGISTRY", reg)
    return reg


def test_stand_does_not_fold_without_mouth_feature(env, no_mouth):  # noqa: F811
    mouth_h.set_auto_arm(True)  # even if something asks for auto-arm
    assert mouth_h._auto_arm is False and mouth_h._auto_arm_pending is False
    mouth_h._auto_arm = True    # ... or the latch is left on
    dispatcher.dispatch_intent({"intent": "stand", "params": {}})
    _join_auto()
    assert env.cmd.kinds() == ["mobility"]  # the stand only, no arm_joints
    assert not env.mouth.enabled and env.mouth.log == []
    assert mouth_h._auto_arm_pending is False


def test_session_ensure_hook_is_harmless_without_mouth(env, no_mouth):  # noqa: F811
    mouth_h._auto_arm = mouth_h._auto_arm_pending = True
    assert mouth_h.start_auto_arm(0) is None
    assert env.cmd.sent == [] and not env.mouth.enabled


def test_stow_still_stows_without_mouth_feature(env, no_mouth, monkeypatch):  # noqa: F811
    from src.spot.robot import ARM
    monkeypatch.setattr(ARM, "_pose", "talk")
    sent = []
    monkeypatch.setattr(ARM, "send", lambda c, pose, **kw: sent.append(pose) or True)
    monkeypatch.setattr(ARM, "stow_now", lambda c, **kw: sent.append("stow_now"))
    dispatcher.dispatch_intent({"intent": "stow", "params": {}})
    assert "stowed" in sent or "stow_now" in sent


def test_main_gate_off_without_mouth(monkeypatch):
    from types import SimpleNamespace
    import src.spot.app.main as _app
    args, tun = SimpleNamespace(no_arm=False, dry_run=False), SimpleNamespace(auto_arm=True)
    assert _app.auto_arm_wanted(args, tun) is True
    assert _app.auto_arm_on(args, tun) is True       # all features enabled
    reg = Registry()
    reg.load([n for n in ENABLED if n != "mouth"])
    monkeypatch.setattr(_features, "REGISTRY", reg)
    assert _app.auto_arm_on(args, tun) is False      # mouth removed: never fold


def test_deleted_mouth_folder_not_fatal_when_disabled(monkeypatch, capsys):
    """Mouth off in enabled.py and its folder gone (core bound _NoMouth):
    the robot start is not refused. With mouth still enabled it is (C6)."""
    from types import SimpleNamespace
    import src.spot.app.main as _app
    import src.spot.features.enabled as _en
    import src.spot.robot as robot_pkg
    from src.spot.core.events import EventBus
    monkeypatch.setattr(robot_pkg, "mouth_handlers", robot_pkg._NoMouth("ImportError: gone"))
    reg = Registry()
    monkeypatch.setattr(_features, "REGISTRY", reg)
    monkeypatch.setattr(reg, "load", lambda names: [])
    args = SimpleNamespace(dry_run=False)
    monkeypatch.setattr(_en, "ENABLED", [n for n in ENABLED if n != "mouth"])
    assert _app._start_features(args, EventBus(), robot_pkg, _app._tuning) is True
    monkeypatch.setattr(_en, "ENABLED", list(ENABLED))
    assert _app._start_features(args, EventBus(), robot_pkg, _app._tuning) is False
    assert "mouth (core binding)" in capsys.readouterr().out
