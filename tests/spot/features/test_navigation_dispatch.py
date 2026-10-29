"""Navigation / visual_nav intents reach their registered handlers through
dispatch_intent with the legacy behaviour (session bring-up, catch-all,
nested go_to, describe fall-through). No robot: fakes only."""
import pytest

import src.spot.robot as _r
from src.spot.features import REGISTRY
from src.spot.robot import dispatcher
import src.spot.features.navigation.handlers as nav_h
import src.spot.features.visual_nav.handlers as vn_h

T14 = ("walk_to", "go_to", "tour", "patrol", "come_back", "load_map", "save_location",
       "list_maps", "list_locations", "go_to_object", "check_obstacles", "describe")


@pytest.fixture
def no_legacy(monkeypatch):
    def legacy(*a, **k):
        raise AssertionError("registered intent reached the legacy branches")
    monkeypatch.setattr(dispatcher, "LEGACY_FEATURE_DISPATCH", legacy)


class FakeCmd:
    def __init__(self):
        self.sent = []

    def robot_command(self, cmd, end_time_secs=None, timeout=None):
        self.sent.append(cmd)
        return 7


def test_every_t14_intent_is_registered():
    assert all(REGISTRY.action(n) is not None for n in T14)


def test_describe_gets_a_session_then_is_refused(monkeypatch, no_legacy, capsys):
    calls = []
    monkeypatch.setattr(_r.CONNECTION, "ensure", lambda *a, **k: calls.append(1) or {"cmd": FakeCmd()})
    assert dispatcher.dispatch_intent({"intent": "describe", "params": {}}) is False
    assert calls == [1]
    assert "[Spot] Unknown intent: describe" in capsys.readouterr().out


def test_check_obstacles_without_robot_says_sensors_line(monkeypatch, no_legacy):
    def no_robot(*a, **k):
        raise RuntimeError("robot unreachable")
    monkeypatch.setattr(_r.CONNECTION, "ensure", no_robot)
    said = []
    assert dispatcher.dispatch_intent({"intent": "check_obstacles", "params": {}},
                                      speak=said.append) is False
    assert said == ["I can't reach my sensors right now."]


def test_go_to_without_robot_is_false_and_silent(monkeypatch, no_legacy):
    def no_robot(*a, **k):
        raise RuntimeError("robot unreachable")
    monkeypatch.setattr(_r.CONNECTION, "ensure", no_robot)
    said = []
    assert dispatcher.dispatch_intent({"intent": "go_to", "params": {"location": "x"}},
                                      speak=said.append) is False
    assert said == []


def test_walk_to_sends_capped_trajectory_and_records_it(monkeypatch):
    cmd = FakeCmd()
    monkeypatch.setattr(_r.CONNECTION, "ensure", lambda *a, **k: {"cmd": cmd})
    monkeypatch.setattr(_r.ARM, "_last_traj", None)
    assert nav_h.handle_walk_to({"relative": [0.0, -1.0, 0.0]}, None) is True
    assert len(cmd.sent) == 1 and _r.ARM._last_traj[0] == 7
    assert nav_h.handle_walk_to({"relative": [1.0]}, None) is False
    assert len(cmd.sent) == 1


def test_go_to_object_saved_location_redispatches_go_to(monkeypatch, no_legacy):
    monkeypatch.setattr(_r.CONNECTION, "ensure", lambda *a, **k: {"cmd": FakeCmd(), "robot": object()})
    monkeypatch.setattr(vn_h, "_list_saved_locations", lambda: {"front_door": "w1"})
    seen = []
    spec = REGISTRY.action("go_to")
    monkeypatch.setitem(REGISTRY._actions, "go_to",
                        type(spec)(**{**spec.__dict__, "handler": lambda p, c: seen.append(p) or True}))
    assert vn_h.handle_go_to_object({"description": "Front door"}, None) is True
    assert seen == [{"location": "front_door"}]


def test_go_to_object_yolo_thread_carries_the_dispatch_gen(monkeypatch):
    monkeypatch.setattr(_r.CONNECTION, "ensure", lambda *a, **k: {"cmd": FakeCmd(), "robot": object()})
    monkeypatch.setattr(vn_h, "_list_saved_locations", lambda: {})
    import src.spot.perception.world_objects as wo
    monkeypatch.setattr(wo, "find_navigation_target", lambda robot, desc: None)
    spawned = []
    monkeypatch.setattr(_r.NAV, "cancel", lambda: spawned.append("cancel"))
    monkeypatch.setattr(_r.NAV, "spawn", lambda fn, mode=None, gen=None: spawned.append((mode, gen)) or True)
    monkeypatch.setattr(_r.SAFETY_GEN, "dispatch_current", lambda: 41)
    assert vn_h.handle_go_to_object({"description": "red chair"}, None) is True
    assert spawned == ["cancel", ("visual_nav", 41)]


def test_spot_dispatch_reexports_the_moved_helpers():
    import src.voice_control.spot_dispatch as d
    assert d._spoken_list is nav_h._spoken_list
    assert d._handle_list_maps is nav_h._handle_list_maps
    assert d._handle_list_locations is nav_h._handle_list_locations
    assert d._is_named_location("waypoint_6") is False and d._is_named_location("desk")
