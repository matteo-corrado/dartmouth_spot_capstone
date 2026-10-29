"""Follow loop against a FakeNavIO simulator with a fake clock (no robot, no GPU)."""
import math
import re
import threading
from types import SimpleNamespace

import importlib

import pytest

# importlib, not ``from vision import follow``: the package re-exports a ``follow`` function.
F = importlib.import_module("src.voice_control.vision.follow")
TEXTS, follow = F.TEXTS, F.follow
from src.voice_control.vision.types import Detection

BAD = re.compile(r"\b(stop|halt|freez\w*|e-?stop|estop)\b", re.IGNORECASE)
RATE = 10.0


@pytest.fixture(autouse=True)
def gpu(monkeypatch):
    monkeypatch.setattr(F, "load_model", lambda name: SimpleNamespace(provider="trt"))


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t


class FakeNavIO:
    """Robot integrates each velocity command for one tick; person at ``person``
    (odom xy, or a callable of time)."""

    def __init__(self, clock, person=(3.0, 0.0), visible=lambda t: True, depth=True,
                 source="frontleft_fisheye_image"):
        self.clock = clock
        self.pose = [0.0, 0.0, 0.0]
        self.person = person
        self.visible = visible
        self.depth = depth
        self.source = source
        self.vels, self.moves, self.snaps = [], [], []
        self.halts = 0
        self._aborted = False
        self.abort_after_vels = None
        self.spoken = []

    def person_xy(self):
        return self.person(self.clock.t) if callable(self.person) else self.person

    def det(self):
        x, y, yaw = self.pose
        px, py = self.person_xy()
        b = math.degrees(math.atan2(py - y, px - x) - yaw)
        b = (b + 180.0) % 360.0 - 180.0
        d = math.hypot(px - x, py - y)
        return Detection(label="person", conf=0.9, source=self.source,
                         box=(100 - b, 50, 160 - b, 250), mask=None, keypoints=None,
                         bearing_deg=b, distance_m=d if self.depth else None,
                         distance_method="depth" if self.depth else None)

    def snapshot(self, sources, labels, model):
        self.snaps.append((self.clock.t, list(sources), labels, model))
        return [self.det()] if self.visible(self.clock.t) else []

    def odom_pose(self):
        return tuple(self.pose)

    def move_body(self, x, y, yaw):
        self.moves.append((x, y, yaw))
        return True

    def velocity(self, vx, vy, vrot):
        if self._aborted:
            return False
        self.vels.append((vx, vy, vrot))
        dt = 1.0 / RATE
        x, y, yaw = self.pose
        self.pose = [x + vx * dt * math.cos(yaw), y + vx * dt * math.sin(yaw), yaw + vrot * dt]
        if self.abort_after_vels is not None and len(self.vels) >= self.abort_after_vels:
            self._aborted = True
        return True

    def halt(self):
        self.halts += 1

    def aborted(self):
        return self._aborted

    def speak(self, text):
        self.spoken.append(text)


def _run(io, clock, stop_event=None, max_ticks=None, **kw):
    ev = stop_event or threading.Event()
    ticks = [0]

    def sleep(dt):
        clock.t += dt
        ticks[0] += 1
        if max_ticks is not None and ticks[0] >= max_ticks:
            ev.set()
        if clock.t > 120:
            raise AssertionError("follow never ended")
        return ev.is_set()

    return follow(io, io.det(), ev, rate_hz=RATE, now=clock.now, sleep=sleep, **kw)


def test_follow_keeps_distance():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.0, 0.0))
    _run(io, clock, max_ticks=1)
    assert io.vels[0][0] > 0.0
    clock = FakeClock()
    io = FakeNavIO(clock, person=(2.0, 0.0))
    _run(io, clock, max_ticks=5)
    assert all(abs(v[0]) < 0.05 for v in io.vels)


def test_follow_converges_to_keep_distance():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(4.0, 0.5))
    r = _run(io, clock, max_ticks=150)
    assert r.text == "I paused."
    d = math.hypot(4.0 - io.pose[0], 0.5 - io.pose[1])
    assert d == pytest.approx(2.0, abs=0.3)


def test_follow_lost_after_three_seconds():
    clock = FakeClock()
    io = FakeNavIO(clock, visible=lambda t: t < 0.5)
    r = _run(io, clock)
    assert r.ok is False and r.text == "I lost you. Say follow me to begin again."
    assert 3.0 <= clock.t <= 3.6
    assert io.halts >= 1


def test_follow_no_command_after_abort():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(5.0, 0.0))
    io.abort_after_vels = 2
    r = _run(io, clock)
    assert r.ok is False and r.text == "I paused."
    assert len(io.vels) == 2 and io.moves == []
    assert io.halts == 0


def test_follow_stop_event_honoured_within_one_tick():
    clock = FakeClock()
    ev = threading.Event()
    io = FakeNavIO(clock, person=(5.0, 0.0))
    orig = io.velocity

    def vel(vx, vy, vrot):
        ok = orig(vx, vy, vrot)
        ev.set()
        return ok
    io.velocity = vel
    r = _run(io, clock, stop_event=ev)
    assert r.text == "I paused." and len(io.vels) == 1 and io.halts == 0


def test_follow_stop_during_snapshot_sends_no_velocity():
    clock = FakeClock()
    ev = threading.Event()
    io = FakeNavIO(clock, person=(5.0, 0.0))
    orig = io.snapshot

    def snap(*a):
        out = orig(*a)
        ev.set()  # the stop lands while detection runs
        return out
    io.snapshot = snap
    r = _run(io, clock, stop_event=ev)
    assert r.text == "I paused." and io.vels == [] and io.halts == 0


def test_follow_stop_before_start_never_moves():
    clock = FakeClock()
    ev = threading.Event()
    ev.set()
    io = FakeNavIO(clock)
    r = _run(io, clock, stop_event=ev)
    assert r.text == "I paused." and io.vels == [] and io.snaps == [] and io.halts == 0


def test_follow_refuses_on_cpu(monkeypatch):
    monkeypatch.setattr(F, "load_model", lambda name: SimpleNamespace(provider="cpu"))
    clock = FakeClock()
    io = FakeNavIO(clock)
    r = _run(io, clock)
    assert r.ok is False
    assert r.text == "My vision processor isn't ready, so I can't follow right now."
    assert io.vels == [] and io.snaps == [] and io.moves == []


def test_follow_refuses_when_model_load_fails(monkeypatch):
    def boom(name):
        raise RuntimeError("no engine")
    monkeypatch.setattr(F, "load_model", boom)
    clock = FakeClock()
    io = FakeNavIO(clock)
    r = _run(io, clock)
    assert r.text == TEXTS["no_gpu"] and io.vels == [] and io.snaps == []


def test_follow_respects_image_budget():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.0, 1.0), source="left_fisheye_image")
    _run(io, clock, max_ticks=50)
    assert io.snaps
    assert max(len(s) for _, s, _, _ in io.snaps) == 3
    for t0, *_ in io.snaps:
        n = sum(len(s) for t, s, _, _ in io.snaps if t0 - 1.0 < t <= t0 + 1e-9)
        assert n <= 20, (t0, n)
    for t0 in range(5):
        n = sum(len(s) for t, s, _, _ in io.snaps if t0 <= t < t0 + 1)
        assert n <= 20


def test_follow_sources_front_pair_plus_last_camera():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.0, 1.0), source="left_fisheye_image")
    _run(io, clock, max_ticks=3)
    _, sources, labels, model = io.snaps[0]
    assert sources == ["frontleft_fisheye_image", "frontright_fisheye_image",
                       "left_fisheye_image"]
    assert labels == {"person"} and model == "yolo26l-seg"


def test_follow_unknown_distance_turns_but_never_drives_forward():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(5.0, 2.0), depth=False)
    _run(io, clock, max_ticks=20)
    assert io.vels and all(v[0] == 0.0 for v in io.vels)
    assert io.vels[0][2] > 0.0  # turns left toward the person


def test_follow_closer_than_keep_never_goes_backward_or_forward():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(1.0, 0.0))
    _run(io, clock, max_ticks=10)
    assert io.vels and all(v[0] == 0.0 for v in io.vels)


def test_follow_caps_and_no_sideways():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(20.0, 0.0))
    _run(io, clock, max_ticks=10)
    assert all(0.0 <= vx <= F.MAX_VX and vy == 0.0 for vx, vy, _ in io.vels)
    assert max(v[0] for v in io.vels) == pytest.approx(F.MAX_VX)
    clock = FakeClock()
    io = FakeNavIO(clock, person=(0.0, 3.0), source="left_fisheye_image")
    _run(io, clock, max_ticks=3)
    assert all(abs(vr) <= F.MAX_VROT for *_, vr in io.vels)
    assert io.vels[0] == (0.0, 0.0, pytest.approx(F.MAX_VROT))  # wide angle: turn only
    assert F.MAX_VX <= 0.5 and F.MAX_VROT <= 0.6


def test_follow_no_forward_on_stale_sighting():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(6.0, 0.0), visible=lambda t: t < 0.05)
    _run(io, clock)
    late = [v for v, t in zip(io.vels, [i / RATE for i in range(len(io.vels))])
            if t > F.STALE_S + 0.15]
    assert late and all(v[0] == 0.0 for v in late)


def test_follow_tracks_moving_person():
    clock = FakeClock()
    io = FakeNavIO(clock, person=lambda t: (3.0 + 0.3 * t, 0.0))
    r = _run(io, clock, max_ticks=100)
    assert r.text == "I paused."
    d = math.hypot(io.person_xy()[0] - io.pose[0], io.person_xy()[1] - io.pose[1])
    assert 1.8 <= d <= 3.5 and io.pose[0] > 2.0


def test_follow_ignores_far_off_person():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.0, 0.0))
    target = io.det()
    io.person = (0.0, -6.0)  # only someone else, far away on the right, is visible
    ev = threading.Event()

    def sleep(dt):
        clock.t += dt
        return ev.is_set()
    r = follow(io, target, ev, rate_hz=RATE, now=clock.now, sleep=sleep)
    assert r.text == TEXTS["lost"]
    assert all(v[2] >= 0.0 for v in io.vels)  # never turned right toward them


def test_follow_snapshot_error_counts_as_unseen():
    clock = FakeClock()
    io = FakeNavIO(clock)

    def boom(*a):
        raise RuntimeError("camera")
    io.snapshot = boom
    r = _run(io, clock)
    assert r.text == TEXTS["lost"] and io.halts >= 1


def test_follow_exception_in_velocity_halts():
    clock = FakeClock()
    io = FakeNavIO(clock)

    def boom(*a):
        raise RuntimeError("rpc")
    io.velocity = boom
    r = _run(io, clock)
    assert r.ok is False and r.text == TEXTS["failed"] and io.halts >= 1


def test_follow_never_speaks_and_texts_have_no_stop_words():
    clock = FakeClock()
    io = FakeNavIO(clock, visible=lambda t: False)
    _run(io, clock)
    assert io.spoken == []
    for key, text in TEXTS.items():
        assert not BAD.search(text), (key, text)
    assert TEXTS["lost"] == "I lost you. Say follow me to begin again."
    assert TEXTS["paused"] == "I paused."
    assert TEXTS["no_gpu"] == "My vision processor isn't ready, so I can't follow right now."


def test_follow_latest_detection_without_distance_never_drives_forward():
    """R19: a remembered 3 m range must not drive once the latest sighting has no distance."""
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.0, 0.3))
    orig = io.snapshot

    def snap(*a):
        io.depth = clock.t < 0.25  # ranged at first, then the same person without range
        return orig(*a)
    io.snapshot = snap
    _run(io, clock, max_ticks=30)
    ranged = [i for i, (t, *_) in enumerate(io.snaps) if t < 0.25]
    first_unranged_t = next(t for t, *_ in io.snaps if t >= 0.25)
    assert io.vels[0][0] > 0.0 and ranged
    after = io.vels[int(round(first_unranged_t * RATE)):]
    assert after and all(v[0] == 0.0 for v in after)


def test_command_law_unknown_or_stale_distance_only_turns():
    assert F.command(10.0, None, 2.0, fresh=True)[0] == 0.0
    assert F.command(10.0, 3.0, 2.0, fresh=False)[0] == 0.0
    vx, vr = F.command(10.0, 3.0, 2.0, fresh=True)
    assert vx > 0.0 and vr > 0.0
