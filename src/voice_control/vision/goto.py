"""Go to a detected object or person and halt short of it (spec 6.4).

The loop never speaks; it returns an ``ActionResult`` whose text the caller
speaks. All motion goes through a ``NavIO``; after a stop (``stop_event`` set
or ``io.aborted()``) no further motion call is made and no halt is sent (the
dispatcher owns the robot then; a halt could cancel e.g. its sit).
"""
from __future__ import annotations

import logging
import math
import threading
import time

from .cameras import BODY_SOURCES
from .navio import NavIO
from .types import ActionResult, Detection
from .world import cameras_covering

logger = logging.getLogger(__name__)

STOP_PERSON_M = 1.5
STOP_OBJECT_M = 1.0
ARRIVE_TOL_M = 0.3
LOST_S = 3.0
STUCK_S = 4.0
PROGRESS_M = 0.1          # odom displacement that counts as progress ...
PROGRESS_YAW_RAD = 0.2    # ... or turning (a 180 deg turn takes ~3 s at 1 rad/s)
MATCH_DEG = 25.0          # re-detection must be this close to the predicted bearing
TURN_FIRST_DEG = 20.0     # beyond this, a tick only turns in place
STEP_M = 1.0              # no distance: walk this far, then re-measure
STEP_DONE_M = 0.9
MAX_STEPS = 8
DEFAULT_MODEL = "yolo26l-seg"

TEXTS = {
    "arrived": "I'm here.",
    "lost": "I lost sight of it.",
    "blocked": "Something's in the way.",
    "paused": "I paused.",
    "no_distance": "I couldn't tell how far away it is.",
    "failed": "I couldn't get there.",
}


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _result(key: str, **data) -> ActionResult:
    return ActionResult(text=TEXTS[key], ok=(key == "arrived"), data={"outcome": key, **data})


class _Target:
    """Target kept in odom: a point when the range is known, else a heading."""

    def __init__(self, pose, det: Detection):
        self.point = None
        self.heading = None
        self.update(pose, det)

    def update(self, pose, det: Detection):
        x, y, yaw = pose
        h = yaw + math.radians(det.bearing_deg)
        d = det.distance_m
        if d is None and self.point is not None:
            d = math.hypot(self.point[0] - x, self.point[1] - y)  # keep the old range
        if d is not None and math.isfinite(d) and d >= 0:
            self.point = (x + d * math.cos(h), y + d * math.sin(h))
            self.heading = None
        else:
            self.point, self.heading = None, h

    def relative(self, pose):
        """(bearing deg +left, distance m or None) from ``pose``."""
        x, y, yaw = pose
        if self.point is not None:
            dx, dy = self.point[0] - x, self.point[1] - y
            return _wrap_deg(math.degrees(math.atan2(dy, dx) - yaw)), math.hypot(dx, dy)
        return _wrap_deg(math.degrees(self.heading - yaw)), None


def goto(io: NavIO, target: Detection, label: str, is_person: bool,
         stop_event: threading.Event, rate_hz: float = 2.0, *,
         now=time.monotonic, sleep=None, model: str = DEFAULT_MODEL) -> ActionResult:
    """Walk toward ``target`` and halt 1.5 m (person) / 1.0 m (object) short."""
    sleep = sleep if sleep is not None else stop_event.wait

    def halted() -> bool:
        return stop_event.is_set() or bool(io.aborted())

    try:
        res = _loop(io, target, label, is_person, halted, rate_hz, now, sleep, model)
    except Exception as e:
        logger.warning("goto failed: %s", e)
        res = _result("failed", error=str(e))
    try:
        stopped = halted()
    except Exception:
        stopped = False
    if stopped:
        return _result("paused")
    try:
        io.halt()  # every other ending leaves the robot standing still
    except Exception as e:
        logger.warning("goto halt failed: %s", e)
    return res


def _loop(io, target, label, is_person, halted, rate_hz, now, sleep, model) -> ActionResult:
    period = 1.0 / rate_hz if rate_hz and rate_hz > 0 else 0.5
    stop_m = STOP_PERSON_M if is_person else STOP_OBJECT_M
    if halted():
        return _result("paused")
    pose = io.odom_pose()
    tgt = _Target(pose, target)
    t0 = now()
    last_seen = t0
    anchor_pose, anchor_t = pose, t0
    steps, step_origin = 0, None
    first = True
    while True:
        if halted():
            return _result("paused")
        t = now()
        pose = io.odom_pose()
        if not first:  # the caller's detection is fresh on the first tick
            pred_b, _ = tgt.relative(pose)
            sources = cameras_covering(pred_b) or list(BODY_SOURCES)
            try:
                dets = io.snapshot(sources, {label}, model)
            except Exception as e:
                logger.warning("goto snapshot failed: %s", e)
                dets = []
            best = None
            for d in dets or []:
                if d.label != label:
                    continue
                off = abs(_wrap_deg(d.bearing_deg - pred_b))
                if off <= MATCH_DEG and (best is None or off < best[0]):
                    best = (off, d)
            if best is not None:
                last_seen = t
                tgt.update(pose, best[1])
            elif t - last_seen >= LOST_S:
                return _result("lost")
        first = False

        moved = math.hypot(pose[0] - anchor_pose[0], pose[1] - anchor_pose[1])
        turned = abs(math.radians(_wrap_deg(math.degrees(pose[2] - anchor_pose[2]))))
        if moved >= PROGRESS_M or turned >= PROGRESS_YAW_RAD:
            anchor_pose, anchor_t = pose, t

        b, d = tgt.relative(pose)
        if d is not None and d <= stop_m + ARRIVE_TOL_M:
            return _result("arrived", distance_m=d)  # never back up toward the goal
        if t - anchor_t >= STUCK_S:
            return _result("blocked")

        if halted():
            return _result("paused")
        yaw = math.radians(b)
        if abs(b) > TURN_FIRST_DEG:
            io.move_body(0.0, 0.0, yaw)
        elif d is not None:
            rem = d - stop_m
            io.move_body(rem * math.cos(yaw), rem * math.sin(yaw), yaw)
        elif step_origin is None or math.hypot(pose[0] - step_origin[0],
                                               pose[1] - step_origin[1]) >= STEP_DONE_M:
            if steps >= MAX_STEPS:
                return _result("no_distance")
            steps += 1
            step_origin = pose
            io.move_body(STEP_M * math.cos(yaw), STEP_M * math.sin(yaw), yaw)
        sleep(period)
