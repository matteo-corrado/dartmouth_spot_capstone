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
from dataclasses import replace

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
FRESH_DIST_S = 1.0        # a person goal needs a distance measured this recently
PROGRESS_M = 0.1          # odom displacement that counts as progress ...
PROGRESS_YAW_RAD = 0.2    # ... or turning (a 180 deg turn takes ~3 s at 1 rad/s)
MATCH_DEG = 25.0          # re-detection must be this close to the predicted bearing
TURN_FIRST_DEG = 20.0     # beyond this, a tick only turns in place
STEP_M = 1.0              # object without distance: walk this far, then re-measure
STEP_DONE_M = 0.9
MAX_STEPS = 8
GROUND_MIN_M = 4.0        # R23: a person's ground-plane range counts only this far out
                          # (same value as follow.GROUND_MIN_M)
DEFAULT_MODEL = "yolo26l-seg"

TEXTS = {
    "arrived": "I'm here.",
    "lost": "I lost sight of it.",
    "blocked": "Something's in the way.",
    "paused": "I paused.",
    "no_distance": "I couldn't tell how far away you are.",       # person
    "no_distance_object": "I couldn't tell how far away it is.",
    "failed": "I couldn't get there.",
}


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _person_range(det: Detection) -> Detection:
    """R23 for a person: a ground-plane range under GROUND_MIN_M (feet hidden ->
    overestimate) is no range at all."""
    d = det.distance_m
    if det.distance_method == "ground" and d is not None and d < GROUND_MIN_M:
        return replace(det, distance_m=None, distance_method=None)
    return det


def _result(key: str, **data) -> ActionResult:
    return ActionResult(text=TEXTS[key], ok=(key == "arrived"), data={"outcome": key, **data})


class _Target:
    """Target kept in odom: a point when the range is known, else a heading.

    Objects keep a remembered range along a new bearing; people never do (a
    person's goal must come from a fresh measured distance), so a distance-less
    person detection leaves the last measured point untouched."""

    def __init__(self, pose, det: Detection, keep_range: bool):
        self.point = None
        self.heading = None
        self.keep_range = keep_range
        self.update(pose, det)

    def update(self, pose, det: Detection):
        x, y, yaw = pose
        h = yaw + math.radians(det.bearing_deg)
        d = det.distance_m
        if d is not None and not (math.isfinite(d) and d >= 0):
            d = None
        if d is None and self.point is not None:
            if not self.keep_range:
                return
            d = math.hypot(self.point[0] - x, self.point[1] - y)
        if d is not None:
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
    st = {"moved": False}

    def halted() -> bool:
        return stop_event.is_set() or bool(io.aborted())

    try:
        res = _loop(io, target, label, is_person, halted, rate_hz, now, sleep, model, st)
    except Exception as e:
        logger.warning("goto failed: %s", e)
        res = _result("failed", error=str(e))
    try:
        stopped = halted()
    except Exception:
        stopped = False
    if stopped:
        return _result("paused")  # no halt: the dispatcher owns the robot now
    if st["moved"]:
        try:
            io.halt()  # leave the robot standing still
        except Exception as e:
            logger.warning("goto halt failed: %s", e)
    return res


def _loop(io, target, label, is_person, halted, rate_hz, now, sleep, model, st) -> ActionResult:
    period = 1.0 / rate_hz if rate_hz and rate_hz > 0 else 0.5
    stop_m = STOP_PERSON_M if is_person else STOP_OBJECT_M

    def move(x, y, yaw):
        st["moved"] = True
        return io.move_odom(x, y, yaw)

    if halted():
        return _result("paused")
    if is_person:
        target = _person_range(target)
    pose = io.odom_pose()
    t0 = now()
    tgt = _Target(pose, target, keep_range=not is_person)
    last_seen = t0
    dist_t = t0 if tgt.point is not None else None  # last measured distance
    anchor_pose, anchor_t = pose, t0
    steps, step_origin = 0, None
    first = True
    while True:
        if halted():
            return _result("paused")
        t = now()
        if not first:  # the caller's detection is fresh on the first tick
            seen_pose = io.odom_pose()
            pred_b, _ = tgt.relative(seen_pose)
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
                if is_person:
                    d = _person_range(d)
                off = abs(_wrap_deg(d.bearing_deg - pred_b))
                if off <= MATCH_DEG and (best is None or off < best[0]):
                    best = (off, d)
            if best is not None:
                last_seen = t
                tgt.update(seen_pose, best[1])
                if best[1].distance_m is not None and tgt.point is not None:
                    dist_t = t
            elif t - last_seen >= LOST_S:
                return _result("lost")
            pose = io.odom_pose()  # fresh pose for progress, arrival and goals
        first = False
        if is_person and dist_t is not None and t - dist_t >= LOST_S:
            return _result("lost")  # no fresh distance for 3 s

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
        h = pose[2] + math.radians(b)  # odom heading toward the target
        if abs(b) > TURN_FIRST_DEG:
            move(pose[0], pose[1], h)  # turn in place
        elif d is not None:
            if not is_person or t - dist_t <= FRESH_DIST_S:
                tx, ty = tgt.point
                move(tx - stop_m * math.cos(h), ty - stop_m * math.sin(h), h)
            else:
                # A person without a fresh distance: replace the last walking
                # goal with a turn-only one at the current position, so the
                # old goal does not keep walking for the trajectory lifetime.
                move(pose[0], pose[1], h)
        elif is_person:
            return _result("no_distance")  # never walk blind toward a person
        else:
            if step_origin is None or math.hypot(pose[0] - step_origin[0],
                                                 pose[1] - step_origin[1]) >= STEP_DONE_M:
                if steps >= MAX_STEPS:
                    return _result("no_distance_object")
                steps += 1
                step_origin = pose
            # re-issue the rest of the current 1 m step every tick
            move(step_origin[0] + STEP_M * math.cos(h), step_origin[1] + STEP_M * math.sin(h), h)
        sleep(period)
