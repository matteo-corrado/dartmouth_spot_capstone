"""Go-to loop (FakeNavIO simulator) and the SpotNavIO adapter (mocked session)."""
import math
import re
import threading
from types import SimpleNamespace
from unittest import mock

import pytest

from src.voice_control.vision.goto import TEXTS, goto
from src.voice_control.vision.types import Detection

BAD = re.compile(r"\b(stop|halt|freez\w*|e-?stop|estop)\b", re.IGNORECASE)


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += dt


class FakeNavIO:
    """Tiny simulator: a target fixed in odom, a robot that walks at most
    ``step`` metres toward each commanded body-frame goal."""

    def __init__(self, target_xy=(5.0, 0.0), label="chair", step=1.0,
                 visible=lambda io: True, depth=True, blocked=False):
        self.pose = [0.0, 0.0, 0.0]
        self.target = target_xy
        self.label = label
        self.step = step
        self.visible = visible
        self.depth = depth
        self.blocked = blocked
        self.moves = []      # (x, y, yaw) body-frame goals
        self.goals_odom = []  # commanded goal endpoints in odom
        self.vels = []
        self.halts = 0
        self.snaps = []
        self._aborted = False
        self.abort_after_moves = None
        self.spoken = []

    # -- NavIO --------------------------------------------------------------
    def bearing_distance(self):
        x, y, yaw = self.pose
        dx, dy = self.target[0] - x, self.target[1] - y
        b = math.degrees(math.atan2(dy, dx) - yaw)
        b = (b + 180.0) % 360.0 - 180.0
        return b, math.hypot(dx, dy)

    def det(self):
        b, d = self.bearing_distance()
        return Detection(label=self.label, conf=0.9, source="frontleft_fisheye_image",
                         box=(0, 0, 10, 10), mask=None, keypoints=None, bearing_deg=b,
                         distance_m=d if self.depth else None,
                         distance_method="depth" if self.depth else None)

    def snapshot(self, sources, labels, model):
        self.snaps.append((list(sources), labels, model))
        return [self.det()] if self.visible(self) else []

    def odom_pose(self):
        return tuple(self.pose)

    def move_body(self, x, y, yaw):
        self.moves.append((x, y, yaw))
        px, py, pyaw = self.pose
        gx = px + x * math.cos(pyaw) - y * math.sin(pyaw)
        gy = py + x * math.sin(pyaw) + y * math.cos(pyaw)
        self.goals_odom.append((gx, gy))
        if self.abort_after_moves is not None and len(self.moves) >= self.abort_after_moves:
            self._aborted = True
        if self.blocked:
            return True
        dist = math.hypot(x, y)
        f = 1.0 if dist <= self.step else self.step / dist
        self.pose = [px + (gx - px) * f, py + (gy - py) * f, pyaw + yaw]
        return True

    def velocity(self, vx, vy, vrot):
        self.vels.append((vx, vy, vrot))
        return True

    def halt(self):
        self.halts += 1

    def aborted(self):
        return self._aborted

    def speak(self, text):
        self.spoken.append(text)


def _run(io, is_person=False, label="chair", stop_event=None, clock=None):
    clock = clock or FakeClock()
    ev = stop_event or threading.Event()

    def sleep(dt):
        clock.advance(dt)
        return ev.is_set()

    return goto(io, io.det(), label, is_person, ev, now=clock.now, sleep=sleep)


def _left(io):
    gx, gy = io.goals_odom[-1]
    return math.hypot(io.target[0] - gx, io.target[1] - gy)


def test_goto_arrives_object_stops_one_metre():
    io = FakeNavIO()
    r = _run(io)
    assert r.ok is True and r.text == "I'm here."
    assert io.moves, "never walked"
    assert _left(io) == pytest.approx(1.0, abs=0.05)
    _, d = io.bearing_distance()
    assert abs(d - 1.0) <= 0.3
    assert io.spoken == []  # goto never speaks


def test_goto_person_stops_one_and_half():
    io = FakeNavIO(target_xy=(6.0, 0.0), label="person")
    r = _run(io, is_person=True, label="person")
    assert r.ok is True and r.text == "I'm here."
    assert _left(io) == pytest.approx(1.5, abs=0.05)


def test_goto_turns_first_then_walks():
    io = FakeNavIO(target_xy=(0.0, 4.0))  # 90 deg to the left
    r = _run(io)
    assert r.ok is True
    x, y, yaw = io.moves[0]
    assert (x, y) == (0.0, 0.0) and yaw == pytest.approx(math.pi / 2, abs=1e-3)


def test_goto_never_backs_up_when_already_close():
    io = FakeNavIO(target_xy=(0.4, 0.0))
    r = _run(io)
    assert r.ok is True and r.text == "I'm here."
    assert io.moves == []


def test_goto_lost_after_three_seconds():
    clock = FakeClock()
    io = FakeNavIO(target_xy=(8.0, 0.0), visible=lambda io: clock.t < 0.6, step=0.3)
    r = _run(io, clock=clock)
    assert r.ok is False and r.text == "I lost sight of it."
    assert 3.0 <= clock.t <= 4.0
    assert io.halts >= 1


def test_goto_no_progress_says_in_the_way():
    clock = FakeClock()
    io = FakeNavIO(target_xy=(5.0, 0.0), blocked=True)
    r = _run(io, clock=clock)
    assert r.ok is False and r.text == "Something's in the way."
    assert 4.0 <= clock.t <= 5.0
    assert io.halts >= 1


def test_goto_no_command_after_abort():
    io = FakeNavIO(target_xy=(9.0, 0.0), step=0.5)
    io.abort_after_moves = 2
    r = _run(io)
    assert r.ok is False and r.text == "I paused."
    assert len(io.moves) + len(io.vels) == 2
    assert io.halts == 0  # the dispatcher owns the robot after a stop


def test_goto_stop_event_honoured_within_one_tick():
    clock = FakeClock()
    ev = threading.Event()
    io = FakeNavIO(target_xy=(9.0, 0.0), step=0.5)
    orig = io.move_body

    def move(x, y, yaw):
        ok = orig(x, y, yaw)
        if len(io.moves) == 1:
            ev.set()
        return ok
    io.move_body = move
    r = _run(io, stop_event=ev, clock=clock)
    assert r.text == "I paused." and r.ok is False
    assert len(io.moves) == 1 and io.halts == 0


def test_goto_stop_event_set_before_start_never_moves():
    ev = threading.Event()
    ev.set()
    io = FakeNavIO()
    r = _run(io, stop_event=ev)
    assert r.text == "I paused." and io.moves == [] and io.vels == []


def test_goto_no_depth_walks_one_metre_steps_then_gives_up():
    clock = FakeClock()
    io = FakeNavIO(target_xy=(30.0, 0.0), depth=False)
    r = _run(io, clock=clock)
    assert r.ok is False and r.text == TEXTS["no_distance"]
    assert len(io.moves) == 8
    assert all(math.hypot(x, y) == pytest.approx(1.0) for x, y, _ in io.moves)


def test_goto_no_depth_switches_to_measured_when_distance_appears():
    io = FakeNavIO(target_xy=(5.0, 0.0), depth=False)
    orig = io.move_body

    def move(x, y, yaw):
        ok = orig(x, y, yaw)
        io.depth = True  # depth shows up after the first step
        return ok
    io.move_body = move
    r = _run(io)
    assert r.ok is True and r.text == "I'm here."
    assert _left(io) == pytest.approx(1.0, abs=0.05)


def test_goto_snapshot_error_counts_as_unseen_and_halts():
    clock = FakeClock()
    io = FakeNavIO(target_xy=(8.0, 0.0), step=0.3)

    def boom(*a, **k):
        raise RuntimeError("camera")
    io.snapshot = boom
    r = _run(io, clock=clock)
    assert r.text == "I lost sight of it." and io.halts >= 1


def test_goto_exception_in_move_halts_and_reraises_nothing():
    io = FakeNavIO()

    def boom(*a, **k):
        raise RuntimeError("rpc")
    io.move_body = boom
    r = _run(io)
    assert r.ok is False and io.halts >= 1


def test_goto_resnapshots_on_covering_cameras_for_label():
    io = FakeNavIO()
    _run(io)
    assert io.snaps
    for sources, labels, model in io.snaps:
        assert labels == {"chair"} and sources and len(sources) <= 2


def test_goto_texts_no_stop_words():
    for key, text in TEXTS.items():
        assert not BAD.search(text), (key, text)
    assert TEXTS["arrived"] == "I'm here."
    assert TEXTS["lost"] == "I lost sight of it."
    assert TEXTS["blocked"] == "Something's in the way."
    assert TEXTS["paused"] == "I paused."


# ---------------------------------------------------------------------------
# SpotNavIO (mocked session; no robot)
# ---------------------------------------------------------------------------

@pytest.fixture
def sd(monkeypatch):
    from src.voice_control import spot_dispatch
    monkeypatch.setattr(spot_dispatch, "_safety_gen", 7)
    return spot_dispatch


def _session(log):
    cmd = mock.Mock()
    cmd.robot_command.side_effect = lambda *a, **k: log.append(("send", a, k))
    state = mock.Mock()
    state.get_robot_state.side_effect = lambda *a, **k: (log.append(("state",)), mock.Mock())[1]
    return {"robot": mock.Mock(), "cmd": cmd, "state": state}


def _patch_builders(monkeypatch, log):
    from bosdyn.client import robot_command as rc
    import src.graph_nav_utils as gnu
    sentinel = object()
    spy = mock.Mock(return_value=sentinel)
    monkeypatch.setattr(gnu, "capped_mobility_params", spy)
    monkeypatch.setattr(gnu, "walk_speed_cap", lambda: 0.5)

    def traj(**k):
        log.append(("build_traj", k))
        return "TRAJ"

    def vel(**k):
        log.append(("build_vel", k))
        return "VEL"
    monkeypatch.setattr(rc.RobotCommandBuilder, "synchro_trajectory_command_in_body_frame",
                        staticmethod(traj))
    monkeypatch.setattr(rc.RobotCommandBuilder, "synchro_velocity_command", staticmethod(vel))
    monkeypatch.setattr(rc.RobotCommandBuilder, "stop_command", staticmethod(lambda: "STOP"))
    return sentinel, spy


def _wrap_aborted(monkeypatch, sd, log):
    real = sd._aborted

    def rec(gen):
        log.append(("check",))
        return real(gen)
    monkeypatch.setattr(sd, "_aborted", rec)


def test_spotnavio_move_checks_abort_immediately_before_send(sd, monkeypatch):
    from src.voice_control.vision.navio import SpotNavIO
    log = []
    sentinel, spy = _patch_builders(monkeypatch, log)
    _wrap_aborted(monkeypatch, sd, log)
    io = SpotNavIO(_session(log), 7, None)
    assert io.move_body(1.0, 0.0, 0.0) is True
    kinds = [e[0] for e in log]
    i = kinds.index("send")
    assert kinds[i - 1] == "check"
    assert log[i][1][0] == "TRAJ"
    build = next(e for e in log if e[0] == "build_traj")
    assert build[1]["params"] is sentinel and spy.called
    k = log[i][2]
    import time
    assert k["end_time_secs"] - time.time() <= 5.0


def test_spotnavio_move_aborted_sends_nothing(sd, monkeypatch):
    from src.voice_control.vision.navio import SpotNavIO
    log = []
    _patch_builders(monkeypatch, log)
    sess = _session(log)
    io = SpotNavIO(sess, 6, None)  # gen 6 != live 7: a stop arrived
    assert io.aborted() is True
    assert io.move_body(1.0, 0.0, 0.0) is False
    assert io.velocity(0.3, 0.0, 0.0) is False
    sess["cmd"].robot_command.assert_not_called()


def test_spotnavio_velocity_short_end_time_and_caps(sd, monkeypatch):
    import time
    from src.voice_control.vision.navio import SpotNavIO
    log = []
    sentinel, _ = _patch_builders(monkeypatch, log)
    _wrap_aborted(monkeypatch, sd, log)
    io = SpotNavIO(_session(log), 7, None)
    assert io.velocity(3.0, -3.0, 5.0) is True
    kinds = [e[0] for e in log]
    i = kinds.index("send")
    assert kinds[i - 1] == "check" and log[i][1][0] == "VEL"
    assert log[i][2]["end_time_secs"] - time.time() <= 0.6
    b = next(e for e in log if e[0] == "build_vel")[1]
    assert b["params"] is sentinel
    assert b["v_x"] == pytest.approx(0.5) and b["v_y"] == pytest.approx(-0.5)
    assert abs(b["v_rot"]) <= 1.0


def test_spotnavio_abort_racing_the_send_is_followed_by_stop(sd, monkeypatch):
    from src.voice_control.vision.navio import SpotNavIO
    log = []
    _patch_builders(monkeypatch, log)
    sess = _session(log)

    def send(*a, **k):
        log.append(("send", a, k))
        if a[0] == "TRAJ":
            sd._safety_gen = 8  # a stop lands while our command is in flight
    sess["cmd"].robot_command.side_effect = send
    io = SpotNavIO(sess, 7, None)
    assert io.move_body(1.0, 0.0, 0.0) is False
    sent = [e[1][0] for e in log if e[0] == "send"]
    assert sent == ["TRAJ", "STOP"]


def test_spotnavio_send_error_returns_false(sd, monkeypatch):
    from src.voice_control.vision.navio import SpotNavIO
    log = []
    _patch_builders(monkeypatch, log)
    sess = _session(log)
    sess["cmd"].robot_command.side_effect = RuntimeError("rpc")
    io = SpotNavIO(sess, 7, None)
    assert io.move_body(1.0, 0.0, 0.0) is False
    assert io.velocity(0.1, 0.0, 0.0) is False


def test_spotnavio_lock_busy_fails_closed(sd, monkeypatch):
    from src.voice_control.vision import navio
    log = []
    _patch_builders(monkeypatch, log)
    monkeypatch.setattr(navio, "LOCK_TIMEOUT_S", 0.01)
    sess = _session(log)
    io = navio.SpotNavIO(sess, 7, None)
    with sd._arm_lock:
        assert io.move_body(1.0, 0.0, 0.0) is False
    sess["cmd"].robot_command.assert_not_called()


def test_spotnavio_halt_sends_stop_and_never_raises(sd, monkeypatch):
    from src.voice_control.vision.navio import SpotNavIO
    log = []
    _patch_builders(monkeypatch, log)
    sess = _session(log)
    SpotNavIO(sess, 6, None).halt()  # even when aborted
    assert [e[1][0] for e in log if e[0] == "send"] == ["STOP"]
    sess["cmd"].robot_command.side_effect = RuntimeError("rpc")
    SpotNavIO(sess, 7, None).halt()


def test_spotnavio_odom_pose(sd, monkeypatch):
    from src.voice_control.vision import navio
    tf = SimpleNamespace(x=1.5, y=-2.0, rot=SimpleNamespace(to_yaw=lambda: 0.25))
    monkeypatch.setattr(navio, "_odom_tform_body", lambda snap: tf)
    sess = _session([])
    assert navio.SpotNavIO(sess, 7, None).odom_pose() == (1.5, -2.0, 0.25)


def test_spotnavio_speak_never_raises():
    from src.voice_control.vision.navio import SpotNavIO
    said = []
    SpotNavIO({}, 0, said.append).speak("I'm here.")
    assert said == ["I'm here."]
    SpotNavIO({}, 0, None).speak("x")

    def bad(t):
        raise RuntimeError
    SpotNavIO({}, 0, bad).speak("x")
