"""Follow a person at 2 m with velocity commands (spec 6.5).

The loop never speaks; it returns an ``ActionResult`` whose text the caller
speaks. All motion goes through a ``NavIO`` velocity command that lives
<= 0.6 s, so a dead loop leaves the robot standing. After a stop
(``stop_event`` set or ``io.aborted()``) no further motion call is made and no
halt is sent (the dispatcher owns the robot then). Every other ending halts.

Safety rules: never drive forward toward a person whose latest sighting has no
range, or whose last ranged sighting is older than ``STALE_S`` (ages measured
after the snapshot returns); ground-plane ranges drive only beyond
``GROUND_MIN_M``; never back up (vx >= 0); vy = 0. Follow ends within
``MAX_RADIUS_M`` of its start and ``MAX_DURATION_S``.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import replace

from .navio import NavIO
from .runtime import load_model
from .tracker import UNRANGED_MAX_DEG, CameraTracker, gate, handoff
from .types import ActionResult, Detection

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "yolo26l-seg"
FRONT_PAIR = ("frontleft_fisheye_image", "frontright_fisheye_image")
LABEL = "person"
LOST_S = 3.0             # unseen this long -> halt and say so
STALE_S = 1.0            # older sightings never drive forward (detection freshness)
MAX_VX = 0.5             # m/s, on top of SpotNavIO's capped_mobility_params
MAX_VROT = 0.6           # rad/s
K_VX = 0.6               # (m/s) per metre beyond keep_m
K_YAW = 1.2              # (rad/s) per radian of bearing
TURN_FIRST_DEG = 45.0    # beyond this bearing only turn
DEADBAND_M = 0.1         # |d - keep| inside this -> no forward speed
HANDOFF_M = 1.0
HANDOFF_DEG = 30.0
GROUND_MIN_M = 4.0       # ground-plane range may drive only this far out (R23)
MAX_RADIUS_M = 15.0      # travel limit from the start pose (R22)
MAX_DURATION_S = 120.0   # time limit (R22)
MAX_SEND_FAILS = 20      # consecutive non-aborted velocity failures (~2 s) -> give up
GPU_PROVIDERS = ("trt", "cuda")

TEXTS = {
    "lost": "I lost you. Say follow me to begin again.",
    "paused": "I paused.",
    "no_gpu": "My vision processor isn't ready, so I can't follow right now.",
    "failed": "I can't keep following you right now.",
    "limit": "I've gone far enough, so I'll wait here.",
}


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _result(key: str, **data) -> ActionResult:
    return ActionResult(text=TEXTS[key], ok=False, data={"outcome": key, **data})


def _gpu_ready(model: str) -> bool:
    try:
        return load_model(model).provider in GPU_PROVIDERS
    except Exception as e:
        logger.warning("follow: model load failed: %s", e)
        return False


def _finite_dist(d) -> bool:
    return d is not None and math.isfinite(d) and d >= 0


class _Target:
    """Last sighting kept in odom: a point when it had a range, else a heading (an
    old range is never reused to drive). ``seen_t`` feeds the lost timer,
    ``fresh_t`` (time of the last ranged sighting) the forward-motion gate, and
    ``last_range`` only the hand-off gate."""

    def __init__(self, pose, det: Detection, t: float):
        self.fresh_t = -math.inf      # the caller's detection is fresh only if ranged
        self.last_range = None
        self.method = None
        self.update(pose, det, t)

    def update(self, pose, det: Detection, t: float):
        x, y, yaw = pose
        h = yaw + math.radians(det.bearing_deg)
        if _finite_dist(det.distance_m):
            d = det.distance_m
            self.point, self.heading = (x + d * math.cos(h), y + d * math.sin(h)), None
            self.fresh_t, self.last_range, self.method = t, d, det.distance_method
        else:
            self.point, self.heading = None, h
        self.det, self.seen_t = det, t

    def relative(self, pose):
        """(bearing deg +left, distance m or None) from ``pose``."""
        x, y, yaw = pose
        if self.point is not None:
            dx, dy = self.point[0] - x, self.point[1] - y
            return _wrap_deg(math.degrees(math.atan2(dy, dx) - yaw)), math.hypot(dx, dy)
        return _wrap_deg(math.degrees(self.heading - yaw)), None

    def predicted(self, pose) -> Detection:
        b, d = self.relative(pose)
        return replace(self.det, bearing_deg=b, distance_m=d)


def command(bearing_deg: float, distance_m, keep_m: float, fresh: bool, method=None):
    """(vx, vrot) for a target at ``bearing_deg``/``distance_m``. vx is 0 unless the
    distance is known, the ranged sighting fresh, a ground-plane range at least
    ``GROUND_MIN_M``, the target roughly ahead and farther than ``keep_m``; never negative."""
    vrot = max(-MAX_VROT, min(MAX_VROT, K_YAW * math.radians(bearing_deg)))
    vx = 0.0
    if (fresh and _finite_dist(distance_m) and abs(bearing_deg) <= TURN_FIRST_DEG
            and not (method == "ground" and distance_m < GROUND_MIN_M)
            and distance_m - keep_m > DEADBAND_M):
        vx = min(MAX_VX, K_VX * (distance_m - keep_m))
        vx = max(0.0, vx * math.cos(math.radians(bearing_deg)))
    return vx, vrot


def follow(io: NavIO, target: Detection, stop_event: threading.Event,
           rate_hz: float = 10.0, keep_m: float = 2.0, max_images_per_s: int = 20, *,
           now=time.monotonic, sleep=None, model: str = DEFAULT_MODEL) -> ActionResult:
    """Follow ``target`` (a person detection) keeping ``keep_m`` until lost, stopped or at a limit."""
    sleep = sleep if sleep is not None else stop_event.wait

    def halted() -> bool:
        return stop_event.is_set() or bool(io.aborted())

    try:
        if halted():
            return _result("paused")
    except Exception:
        pass
    if not _gpu_ready(model):
        return _result("no_gpu")  # nothing moved, nothing to halt
    try:
        res = _loop(io, target, halted, rate_hz, keep_m, max_images_per_s, now, sleep, model)
    except Exception as e:
        logger.warning("follow failed: %s", e)
        res = _result("failed", error=str(e))
    try:
        stopped = halted()
    except Exception:
        stopped = False
    if stopped:
        return _result("paused")
    try:
        io.halt()  # lost / limit / failed: leave the robot standing still
    except Exception as e:
        logger.warning("follow halt failed: %s", e)
    return res


def _loop(io, target, halted, rate_hz, keep_m, max_images, now, sleep, model) -> ActionResult:
    period = 1.0 / rate_hz if rate_hz and rate_hz > 0 else 0.1
    if halted():
        return _result("paused")
    t0 = now()
    start = io.odom_pose()
    tgt = _Target(start, target, t0)
    track = None                      # (source, track_id) of the target
    trackers: dict[str, CameraTracker] = {}
    window: deque = deque()           # (t, images) snapshots in the last second
    fails = 0
    while True:
        if halted():
            return _result("paused")
        # Capture time/pose: anchors this tick's detections (conservative: the
        # sighting is stamped before the snapshot, so it only ever looks older).
        t_cap = now()
        pose_cap = io.odom_pose()

        sources = list(FRONT_PAIR)
        if tgt.det.source and tgt.det.source not in sources:
            sources.append(tgt.det.source)
        while window and window[0][0] <= t_cap - 1.0:
            window.popleft()
        if sum(n for _, n in window) + len(sources) <= max_images:
            window.append((t_cap, len(sources)))
            try:
                dets = io.snapshot(sources, {LABEL}, model) or []
            except Exception as e:
                logger.warning("follow snapshot failed: %s", e)
                dets = []
            match = _match(dets, sources, trackers, track, tgt.predicted(pose_cap),
                           tgt.last_range, t_cap)
            if match is not None:
                tgt.update(pose_cap, match, t_cap)
                track = (match.source, match.track_id)

        # Control time/pose: read after the snapshot returned, so ages include its latency.
        t = now()
        pose = io.odom_pose()
        if (t - t0 >= MAX_DURATION_S
                or math.hypot(pose[0] - start[0], pose[1] - start[1]) > MAX_RADIUS_M):
            return _result("limit")
        if t - tgt.seen_t >= LOST_S:
            return _result("lost")
        b, d = tgt.relative(pose)
        vx, vrot = command(b, d, keep_m, fresh=(t - tgt.fresh_t) <= STALE_S, method=tgt.method)

        if halted():
            return _result("paused")
        if io.velocity(vx, 0.0, vrot):
            fails = 0
        else:
            if halted():
                return _result("paused")
            fails += 1
            if fails >= MAX_SEND_FAILS:
                return _result("failed", error="velocity commands not accepted")
        sleep(max(0.0, period - (now() - t_cap)))


def _match(dets, sources, trackers, track, predicted: Detection, last_range, t: float):
    """Update the per-camera trackers; return the target's detection or None.

    A track-id match must still pass the hand-off gate (an id swap between two
    crossing people must not move the target onto the bystander). An unranged
    same-id sighting within ``UNRANGED_MAX_DEG`` is accepted too: it makes the
    target heading-only, so it can only turn the robot (R19), never drive it."""
    by_src: dict[str, list] = {s: [] for s in sources}
    for d in dets:
        if d.label == LABEL:
            by_src.setdefault(d.source, []).append(d)
    tracked = []
    for src, ds in by_src.items():
        tracked += trackers.setdefault(src, CameraTracker()).update(ds, t)
    if track is not None:
        for d in tracked:
            if (d.source, d.track_id) != track:
                continue
            if gate(predicted, d, HANDOFF_M, HANDOFF_DEG, last_range) is not None:
                return d
            if (not _finite_dist(d.distance_m)
                    and abs(_wrap_deg(d.bearing_deg - predicted.bearing_deg)) <= UNRANGED_MAX_DEG):
                return d
    return handoff(predicted, tracked, max_dist=HANDOFF_M, max_bearing=HANDOFF_DEG,
                   last_range=last_range)
