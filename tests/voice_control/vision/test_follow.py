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
                 source="frontleft_fisheye_image", others=(), latency=0.0,
                 on_snapshot=None, camera_by_bearing=False, method="depth"):
        self.clock = clock
        self.others = list(others)      # bystanders (odom xy), always ranged
        self.latency = latency          # snapshot duration (fake clock advances)
        self.on_snapshot = on_snapshot  # hook run after capture, before return
        self.camera_by_bearing = camera_by_bearing
        self.method = method
        self.vel_t = []
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

    def det(self, xy=None, ranged=None):
        x, y, yaw = self.pose
        px, py = xy if xy is not None else self.person_xy()
        ranged = self.depth if ranged is None else ranged
        b = math.degrees(math.atan2(py - y, px - x) - yaw)
        b = (b + 180.0) % 360.0 - 180.0
        d = math.hypot(px - x, py - y)
        src = self.source
        if self.camera_by_bearing:
            src = ("left_fisheye_image" if b > 50 else "right_fisheye_image" if b < -50
                   else "frontleft_fisheye_image")
        return Detection(label="person", conf=0.9, source=src,
                         box=(100 - b, 50, 160 - b, 250), mask=None, keypoints=None,
                         bearing_deg=b, distance_m=d if ranged else None,
                         distance_method=self.method if ranged else None)

    def snapshot(self, sources, labels, model):
        self.snaps.append((self.clock.t, list(sources), labels, model))
        out = [self.det()] if self.visible(self.clock.t) else []
        out += [self.det(xy, ranged=True) for xy in self.others]
        self.clock.t += self.latency
        if self.on_snapshot is not None:
            self.on_snapshot(self)
        return out

    def odom_pose(self):
        return tuple(self.pose)

    def move_body(self, x, y, yaw):
        self.moves.append((x, y, yaw))
        return True

    def velocity(self, vx, vy, vrot):
        if self._aborted:
            return False
        self.vels.append((vx, vy, vrot))
        self.vel_t.append(self.clock.t)
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
        if clock.t > 300:
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


# ---------------------------------------------------------------------------
# Fix round 1 (safety review): timing, pose, id gate, limits, ground ranges
# ---------------------------------------------------------------------------

def test_follow_freshness_measured_after_slow_snapshot():
    """A sighting 0.8 s old before a 0.4 s snapshot is 1.2 s old at send: no forward."""
    clock = FakeClock()
    io = FakeNavIO(clock, person=(6.0, 0.0), visible=lambda t: t < 0.05, latency=0.4)
    _run(io, clock)
    assert any(v[0] > 0 for v in io.vels)
    for (vx, _, _), t in zip(io.vels, io.vel_t):
        if vx > 0.0:
            assert t - 0.0 <= F.STALE_S, t


def test_follow_lost_measured_after_snapshot():
    clock = FakeClock()
    io = FakeNavIO(clock, visible=lambda t: t < 0.05, latency=0.4)
    r = _run(io, clock)
    assert r.text == TEXTS["lost"]
    assert all(t < F.LOST_S for t in io.vel_t)


def test_follow_pose_change_during_snapshot_turns_correct_way():
    """Robot yaws 0.5 rad left while the snapshot runs: the person (straight ahead at
    capture) is now on the right, so the command must turn right."""
    clock = FakeClock()
    done = []

    def turn(io):
        if not done:
            io.pose[2] += 0.5
            done.append(1)
    io = FakeNavIO(clock, person=(3.0, 0.0), on_snapshot=turn)
    _run(io, clock, max_ticks=1)
    assert io.vels[0][2] < 0.0


def test_match_id_swap_keeps_original_person():
    """Tracker swaps ids between two crossing people: target stays on the original."""
    src = "frontleft_fisheye_image"

    def mk(b, d):
        return Detection(label="person", conf=0.9, source=src, box=(0, 0, 1, 1), mask=None,
                         keypoints=None, bearing_deg=b, distance_m=d, distance_method="depth")

    original, bystander = mk(2.0, 3.0), mk(35.0, 3.0)

    class SwapTracker:
        def update(self, ds, t):
            from dataclasses import replace
            return [replace(d, track_id=1 if d is bystander else 2) for d in ds]

    got = F._match([original, bystander], [src], {src: SwapTracker()}, (src, 1),
                   mk(0.0, 3.0), 3.0, 1.0)
    assert got.bearing_deg == 2.0


def test_follow_bystander_25deg_4m_further_not_adopted():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.0, 0.0), visible=lambda t: t < 0.05,
                   others=[(7.0 * math.cos(math.radians(25)), 7.0 * math.sin(math.radians(25)))])
    r = _run(io, clock)
    assert r.text == TEXTS["lost"]
    assert all(v[2] <= 1e-9 for v in io.vels)  # never turned left toward the bystander


def test_follow_two_people_keeps_the_right_one():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.0, 0.0), others=[(3.0, -1.4)])
    r = _run(io, clock, max_ticks=60)
    assert r.text == TEXTS["paused"]
    assert math.hypot(3.0 - io.pose[0], 0.0 - io.pose[1]) == pytest.approx(2.0, abs=0.3)
    assert abs(io.pose[1]) < 0.15 and all(v[2] > -0.05 for v in io.vels)


def test_follow_camera_handoff_inside_loop():
    """Person starts on the left camera; turning brings them into the front pair."""
    clock = FakeClock()
    io = FakeNavIO(clock, person=(0.0, 3.0), camera_by_bearing=True)
    r = _run(io, clock, max_ticks=60)
    assert r.text == TEXTS["paused"]  # never lost across the hand-off
    srcs = [s[-1] for _, s, _, _ in io.snaps]
    assert "left_fisheye_image" in srcs and len(io.snaps[-1][1]) == 2  # back to the front pair
    assert io.pose[2] == pytest.approx(math.pi / 2, abs=0.15)


def test_follow_radius_limit():
    clock = FakeClock()
    io = FakeNavIO(clock, person=lambda t: (3.0 + 0.45 * t, 0.0))
    r = _run(io, clock)
    assert r.ok is False and r.text == "I've gone far enough, so I'll wait here."
    assert TEXTS["limit"] == r.text and r.data["outcome"] == "limit"
    assert F.MAX_RADIUS_M == 15.0 and 15.0 <= io.pose[0] <= 15.2
    assert io.halts >= 1


def test_follow_duration_limit():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.0, 0.0))
    r = _run(io, clock)
    assert r.text == TEXTS["limit"] and io.halts >= 1
    assert F.MAX_DURATION_S == 120.0 and 120.0 <= clock.t <= 120.2


def test_follow_ground_range_drives_only_beyond_four_metres():
    assert F.command(0.0, 3.5, 2.0, fresh=True, method="ground")[0] == 0.0
    assert F.command(0.0, 3.5, 2.0, fresh=True, method="depth")[0] > 0.0
    assert F.command(0.0, 5.0, 2.0, fresh=True, method="ground")[0] > 0.0
    clock = FakeClock()
    io = FakeNavIO(clock, person=(3.5, 0.0), method="ground")
    _run(io, clock, max_ticks=10)
    assert io.vels and all(v[0] == 0.0 for v in io.vels)


def test_follow_unranged_initial_target_is_not_fresh():
    clock = FakeClock()
    io = FakeNavIO(clock, person=(5.0, 0.0))
    t = io.det(ranged=False)
    tgt = F._Target((0.0, 0.0, 0.0), t, 0.0)
    assert tgt.fresh_t == -math.inf and tgt.seen_t == 0.0
    tgt.update((0.0, 0.0, 0.0), io.det(ranged=True), 0.5)
    assert tgt.fresh_t == 0.5 and tgt.last_range == pytest.approx(5.0)
