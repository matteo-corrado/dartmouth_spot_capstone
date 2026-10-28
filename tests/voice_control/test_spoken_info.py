"""Task 12 (Lane B): spoken info results via dispatch_intent(..., speak=...)."""
import math
from types import SimpleNamespace

import pytest
from bosdyn.api import robot_state_pb2

import src.voice_control.spot_dispatch as d
import src.voice_control.perception.obstacle_query as oq


class FakeStateClient:
    def get_robot_state(self):
        rs = robot_state_pb2.RobotState()
        rs.power_state.locomotion_charge_percentage.value = 73.4
        rs.power_state.locomotion_estimated_runtime.seconds = 95 * 60
        rs.power_state.motor_power_state = rs.power_state.STATE_ON
        e = rs.estop_states.add()
        e.state = 2
        return rs


@pytest.fixture
def sess(monkeypatch):
    s = {"cmd": object(), "state": FakeStateClient(), "robot": object()}
    monkeypatch.setattr(d, "_spot_session", s)
    return s


def _run(name, params=None):
    said = []
    ok = d.dispatch_intent({"intent": name, "params": params or {}}, speak=said.append)
    return ok, said


def test_battery_spoken(sess):
    ok, said = _run("battery_status")
    assert ok and said == ["My battery is at 73 percent, with about 1 hour and 35 minutes of walking left."]


def test_status_spoken(sess):
    ok, said = _run("status")
    assert ok and "73 percent" in said[0] and "motors are on" in said[0] and "released" in said[0]


def test_obstacles_spoken(sess, monkeypatch):
    monkeypatch.setattr(oq, "nearest_obstacle_in_body_frame",
                        lambda r, max_radius_m=3.0: SimpleNamespace(
                            distance_m=1.24, bearing_rad=math.radians(90), grid_name="no_step"))
    ok, said = _run("check_obstacles")
    assert ok and said == ["The closest thing is about 1.2 meters to the left."]


def test_obstacles_none_spoken(sess, monkeypatch):
    monkeypatch.setattr(oq, "nearest_obstacle_in_body_frame", lambda r, max_radius_m=3.0: None)
    ok, said = _run("check_obstacles")
    assert ok and said == ["I don't see anything close to me right now."]


def test_list_locations_no_robot_needed(monkeypatch):
    monkeypatch.setattr(d, "_spot_session", None)
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("no session"))
    monkeypatch.setattr(d, "_list_saved_locations",
                        lambda: {"poster_table": "w1", "front_door": "w2", "waypoint_3": "w3"})
    ok, said = _run("list_locations")
    assert ok and said == ["I know 2 places: poster table and front door."]


def test_list_locations_empty(monkeypatch):
    monkeypatch.setattr(d, "_list_saved_locations", lambda: {})
    ok, said = _run("list_locations")
    assert ok and said == ["I don't have any saved places yet."]


def test_list_maps_no_map_loaded(monkeypatch):
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("no session"))
    monkeypatch.setattr(d, "list_available_maps", lambda root: [])
    ok, said = _run("list_maps")
    assert ok and said == ["I don't have any maps saved yet."]


def test_list_maps_spoken(monkeypatch):
    monkeypatch.setattr(d, "list_available_maps", lambda root: ["sudi_lab", "hallway"])
    monkeypatch.setattr(d, "read_last_used", lambda root: "sudi_lab")
    ok, said = _run("list_maps")
    assert ok and said == ["I have 2 maps: sudi lab and hallway. The current one is sudi lab."]


def test_graceful_line_without_robot(monkeypatch):
    monkeypatch.setattr(d, "_spot_session", None)

    def no_robot(*a, **k):
        raise RuntimeError("robot unreachable")
    monkeypatch.setattr(d, "ensure_spot_session", no_robot)
    ok, said = _run("battery_status")
    assert ok is False and said == ["I can't reach my sensors right now."]


def test_speak_none_prints(sess, capsys):
    assert d.dispatch_intent({"intent": "battery_status", "params": {}}) is True
    assert "My battery is at 73 percent" in capsys.readouterr().out


def test_state_dict_has_no_volatile_keys(monkeypatch):
    monkeypatch.setattr(d, "_spot_session", None)
    st = d.get_robot_state_dict()
    assert "uptime_s" not in st and "process_pid" not in st
