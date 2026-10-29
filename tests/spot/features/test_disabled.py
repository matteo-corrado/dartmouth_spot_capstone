"""Plan Review Focus 3 (Task 13): a disabled feature's intent can still
arrive (regex, LLM, spotctl). It must not be dispatched and must not raise:
the same False as an unknown intent. Fakes only, no robot."""
import pytest

import src.spot.features as features_pkg
import src.spot.robot.arm as arm_mod
import src.spot.robot.dispatcher as d
from src.spot.core.registry import Registry
from src.spot.features.enabled import ENABLED
from src.spot.robot import ARM, CONNECTION, NAV
from tests.voice_control.test_arm_policy import FakeCmd, FakeState


@pytest.fixture
def no_walk(monkeypatch):
    cmd = FakeCmd()
    monkeypatch.setattr(CONNECTION, "_session", {"cmd": cmd, "state": FakeState(), "robot": object()})
    monkeypatch.setattr(CONNECTION, "_context", object())
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(arm_mod, "block_until_arm_arrives", lambda c, i, t: True)
    monkeypatch.setattr(ARM, "_pose", "stowed")
    monkeypatch.setattr(ARM, "_pose_before_walk", None)
    monkeypatch.setattr(ARM, "_pose_during_walk", None)
    monkeypatch.setattr(ARM, "_last_brain", None)
    reg = Registry()
    reg.load([n for n in ENABLED if n != "walk"])
    monkeypatch.setattr(features_pkg, "REGISTRY", reg)
    yield cmd, reg
    NAV.cancel()


def test_disabled_feature_intent_is_refused(no_walk, capsys):
    cmd, reg = no_walk
    assert features_pkg.REGISTRY is reg and reg.action("walk") is None
    assert reg.action("follow_me") is not None  # only walk is disabled

    walk = d.dispatch_intent({"intent": "walk", "params": {}})
    unknown = d.dispatch_intent({"intent": "no_such_intent"})
    out = capsys.readouterr().out

    assert walk is False
    assert unknown is False
    assert walk == unknown
    assert "[Spot] Unknown intent: walk" in out
    assert "[Spot] Unknown intent: no_such_intent" in out
    assert cmd.sent == []                    # nothing reached the robot
    assert ARM._pose == "stowed" and ARM._pose_before_walk is None
    assert NAV.is_navigating() is False


def test_disabled_locomotion_feature_is_not_tucked_for_in_talk_pose(no_walk, monkeypatch, capsys):
    """Task 16: the locomotion set is generated, so a disabled walk is no
    locomotion intent either: with the arm out in the talk pose, nothing is
    tucked and no arm command is sent before the refusal."""
    cmd, reg = no_walk
    monkeypatch.setattr(ARM, "_pose", "talk")
    monkeypatch.setattr(ARM, "_gaze_deg", 30.0)
    # a separate walk pose: an enabled locomotion intent would tuck talk -> walk
    monkeypatch.setattr(arm_mod, "_talk_is_walk_pose", lambda: False)
    assert arm_mod.arm_policy_for("follow_me", "talk") == ("walk", "talk")  # still enabled
    assert arm_mod.arm_policy_for("walk", "talk") == ("talk", "talk")       # not locomotion now

    walk = d.dispatch_intent({"intent": "walk", "params": {"direction": "forward", "distance": 1.0}})
    out = capsys.readouterr().out

    assert walk is False
    assert "[Spot] Unknown intent: walk" in out
    assert cmd.sent == []                    # no tuck, no motion
    assert ARM._pose == "talk" and ARM._pose_before_walk is None and ARM._pose_during_walk is None
    assert NAV.is_navigating() is False
