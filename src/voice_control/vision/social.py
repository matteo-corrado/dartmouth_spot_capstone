"""Friendly Spot social mode on the capstone vision stack.

Port of friendly_spot's perception -> comfort model -> behaviour loop:

- People come from YOLO26-pose on the surround fisheyes plus the gripper
  camera, with range from each camera's registered depth (``mask_distance``).
- The gripper camera replaces the Spot CAM PTZ: the folded arm is yawed toward
  the target (``aim``), and when the gripper sees the target its higher-res
  view is used for pose, face emotion and hand gestures.
- ``ComfortModel`` (unchanged weights) picks GO_CLOSE / GO_CLOSE_SLOWLY / STAY /
  SIT / BACK_AWAY_SLOWLY / BACK_AWAY. SIT is a crouch (the arm stays up so the
  gripper keeps watching); BACK_AWAY is a short slow reverse.

Safety: every motion goes through ``NavIO`` (safety generation re-checked
under the arm lock). After a stop (``stop_event`` / ``io.aborted()``) no more
motion is sent and no halt either; the dispatcher owns the robot. Forward
moves only with a fresh depth range (torso keypoints, else box), minus a body
margin, never leaving anyone ahead closer than ``MIN_GAP_M``. Reverse only
when the back camera's depth is clear to ``REAR_CLEAR_M`` and it sees nobody.
A stop mid-crouch leaves the robot crouched; the next stand/walk restores height. The mode stays
within ``MAX_RADIUS_M`` of its start and ends after ``MAX_DURATION_S``.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

from .cameras import BODY_SOURCES, HAND_SOURCE, upright_to_raw
from .comfort import BehaviorLabel, ComfortModel, PerceptionInput
from .navio import NavIO
from .social_pose import PoseClassifier, normalise
from .types import ActionResult, Detection

logger = logging.getLogger(__name__)

MODEL = "yolo26l-pose"
SOURCES = (*BODY_SOURCES, HAND_SOURCE)
REAR_SOURCE = "back_fisheye_image"
RATE_HZ = 2.0
MAX_DURATION_S = 600.0
MAX_RADIUS_M = 3.0        # social mode stays near where it started
TRACK_DEG = 30.0          # keep the same person if they are within this of the last bearing
HAND_MATCH_DEG = 15.0     # a gripper detection this close to the target is the target
STABLE_CYCLES = 2         # a behaviour must win this many cycles in a row
MOVE_REPEAT_S = 3.0       # an unchanged move behaviour may repeat after this
SETTLE_S = 1.5            # after a move: no gaze, pose history cleared
TURN_DEG = 50.0           # target beyond this bearing -> turn the body to face them
MIN_GAP_M = 0.8           # never step closer than this to a person
MAX_STEP_M = 0.5
STALE_S = 1.0
REAR_CLEAR_M = 1.5
BODY_MARGIN_M = 0.2       # person surface is nearer than their torso depth
STEP_CLEAR_DEG = 30.0     # every ranged person this close to the step direction must clear the gap
CROUCH_M = -0.2
TORSO_KPS = (5, 6, 11, 12)  # COCO shoulders + hips
KP_MIN_CONF = 0.3

# behaviour -> (stop distance m, max step m)   (original: 1.0 m @ 0.8 m/s, 1.5 m @ 0.5 m/s)
APPROACH = {BehaviorLabel.GO_CLOSE: (1.0, 0.5), BehaviorLabel.GO_CLOSE_SLOWLY: (1.5, 0.3)}
# behaviour -> (distance m, speed m/s)   (original 0.5 m / 0.3 m; operator: short and slow)
REVERSE = {BehaviorLabel.BACK_AWAY: (0.3, 0.2), BehaviorLabel.BACK_AWAY_SLOWLY: (0.2, 0.12)}
MOVES = set(APPROACH) | set(REVERSE)

TEXTS = {
    "paused": "I paused.",
    "done": "That was nice. I'll stop being social now.",
    "limit": "I've wandered far enough, so I'll stop being social now.",
    "failed": "I can't keep my social mode going right now.",
}


def _wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _ranged(d: Detection) -> bool:
    return d.distance_m is not None and math.isfinite(d.distance_m) and d.distance_m > 0


class SocialState:
    """What the loop knows about its current person between cycles."""

    def __init__(self):
        self.bearing: float | None = None
        self.source: str | None = None
        self.last_seen = -math.inf
        self.pose = PoseClassifier()
        self.history: deque = deque(maxlen=STABLE_CYCLES)
        self.last_behavior: BehaviorLabel | None = None
        self.last_move_t = -math.inf
        self.crouched = False

    def moved(self, t: float) -> None:
        self.last_move_t = t
        self.pose.reset()


def pick_target(dets: list[Detection], last_bearing: float | None) -> Detection | None:
    """The person to attend to: whoever is nearest the last bearing (within
    TRACK_DEG), else the nearest ranged person, else the most confident one."""
    people = [d for d in dets if d.label == "person"]
    if not people:
        return None
    if last_bearing is not None:
        near = [d for d in people if abs(_wrap(d.bearing_deg - last_bearing)) <= TRACK_DEG]
        if near:
            return min(near, key=lambda d: abs(_wrap(d.bearing_deg - last_bearing)))
    ranged = [d for d in people if _ranged(d)]
    if ranged:
        return min(ranged, key=lambda d: d.distance_m)
    return max(people, key=lambda d: d.conf)


def best_view(dets: list[Detection], target: Detection) -> Detection:
    """Gripper view of the target if the gripper sees them, else the target."""
    if target.source == HAND_SOURCE:
        return target
    hand = [d for d in dets if d.label == "person" and d.source == HAND_SOURCE
            and abs(_wrap(d.bearing_deg - target.bearing_deg)) <= HAND_MATCH_DEG]
    return max(hand, key=lambda d: d.conf) if hand else target


def keypoint_range(frame, kps) -> float | None:
    """Depth at the torso keypoints (shoulders, hips): the nearest of the 5x5
    medians. pose models give no mask, and a box median can read the wall
    between someone's arms and legs."""
    if frame is None or frame.depth is None or kps is None:
        return None
    vals = []
    rows, cols = frame.depth.shape[:2]
    for i in TORSO_KPS:
        x, y, c = kps[i]
        if c < KP_MIN_CONF:
            continue
        u, v = upright_to_raw(float(x), float(y), frame.raw_rows, frame.raw_cols, frame.rot_deg)
        u, v = int(round(u)), int(round(v))
        if not (0 <= u < cols and 0 <= v < rows):
            continue
        patch = frame.depth[max(v - 2, 0):v + 3, max(u - 2, 0):u + 3]
        patch = patch[np.isfinite(patch) & (patch > 0)]
        if patch.size >= 5:
            vals.append(float(np.median(patch)))
    return min(vals) if vals else None


def best_range(dets: list[Detection], target: Detection,
               frames: dict | None = None) -> tuple[float | None, str | None]:
    """Range to the target: torso-keypoint depth from any camera seeing them,
    else their box depth range, else a ground-plane range."""
    same = [d for d in dets if d.label == "person"
            and abs(_wrap(d.bearing_deg - target.bearing_deg)) <= HAND_MATCH_DEG]
    kp = [r for r in (keypoint_range((frames or {}).get(d.source), d.keypoints) for d in same)
          if r is not None]
    if kp:
        return min(kp), "keypoints"
    ranged = [d for d in same if _ranged(d)]
    depth = [d for d in ranged if d.distance_method == "depth"]
    pool = depth or ranged
    if not pool:
        return None, None
    d = min(pool, key=lambda d: d.distance_m)
    return d.distance_m, d.distance_method


def step_gap_ok(dets: list[Detection], bearing: float, step: float) -> bool:
    """After a ``step`` toward ``bearing`` every ranged person near that direction
    is still at least MIN_GAP_M away (bystanders included)."""
    for d in dets:
        if d.label != "person" or not _ranged(d):
            continue
        if abs(_wrap(d.bearing_deg - bearing)) <= STEP_CLEAR_DEG:
            if d.distance_m - BODY_MARGIN_M - step < MIN_GAP_M:
                return False
    return True


def rear_depth_clear(frame) -> bool:
    """Back camera depth (any surface, not only people): the lower-central half of
    the image has nothing nearer than REAR_CLEAR_M. Missing depth -> not clear."""
    if frame is None or frame.depth is None:
        return False
    dep = frame.depth
    rows, cols = dep.shape[:2]
    region = dep[rows // 4:, cols // 4: 3 * cols // 4]
    vals = region[np.isfinite(region) & (region > 0)]
    if vals.size < 50:
        return False  # can't see: don't back up blind
    return float(np.percentile(vals, 5)) >= REAR_CLEAR_M


def rear_clear(dets: list[Detection], frames: dict | None = None) -> bool:
    """Safe to back up: the back camera returned this cycle with clear depth,
    no ranged detection behind closer than REAR_CLEAR_M, and no unranged person
    behind (could be right behind us)."""
    if not rear_depth_clear((frames or {}).get(REAR_SOURCE)):
        return False
    for d in dets:
        if d.source != REAR_SOURCE:
            continue
        if _ranged(d) and d.distance_m < REAR_CLEAR_M:
            return False
        if d.label == "person" and not _ranged(d):
            return False
    return True


def stable_behavior(state: SocialState, behavior: BehaviorLabel, now: float) -> BehaviorLabel | None:
    """Debounce: the behaviour to execute now, or None."""
    state.history.append(behavior)
    if len(state.history) < STABLE_CYCLES or len(set(state.history)) != 1:
        return None
    if behavior != state.last_behavior:
        return behavior
    if behavior in MOVES and now - state.last_move_t >= MOVE_REPEAT_S:
        return behavior
    return None


class Perceiver:
    """Fills a ``PerceptionInput`` for one target from one camera crop."""

    def __init__(self, emotion=None, gesture=None):
        # emotion(img, box) -> (label, scores, face_box); gesture(img, box) -> label
        self.emotion = emotion
        self.gesture = gesture

    def perceive(self, frames: dict, view: Detection, distance_m, current_action: str,
                 pose: PoseClassifier) -> PerceptionInput:
        p = PerceptionInput(current_action=current_action, distance_m=distance_m)
        if view.keypoints is not None:
            p.pose_landmarks = normalise(view.keypoints, view.box)
            p.pose_label = pose.classify(p.pose_landmarks)
        else:
            p.pose_label = "unknown"
        frame = frames.get(view.source)
        img = frame.image if frame is not None else None
        if img is not None and self.emotion is not None:
            try:
                label, scores, face_box = self.emotion(img, view.box)
                if label:
                    p.emotion_label, p.emotion_scores, p.face_bbox = label, scores, face_box
            except Exception as e:
                logger.debug("social: emotion failed: %s", e)
        if img is not None and self.gesture is not None:
            try:
                p.gesture_label = self.gesture(img, view.box) or "none"
            except Exception as e:
                logger.debug("social: gesture failed: %s", e)
        p.frame = img
        return p


REVERSE_TICK_S = 0.2


def _reverse(io: NavIO, dist: float, speed: float, halted: Callable[[], bool],
             wait: Callable[[float], object]) -> bool:
    """Back up about ``dist`` m at ``speed`` m/s with short-lived velocity commands.
    Ends on odometry (leaving room for the last command's coast), or after the
    commands that cover ``dist`` minus that coast, then sends zero velocity."""
    from .navio import VEL_END_S
    coast = speed * VEL_END_S
    ticks = max(1, math.floor((dist - coast) / speed / REVERSE_TICK_S))
    try:
        x0, y0, _ = io.odom_pose()
    except Exception:
        return False  # no odometry: don't back up
    for _ in range(ticks):
        if halted():
            return False
        try:
            x, y, _ = io.odom_pose()
        except Exception:
            break
        if math.hypot(x - x0, y - y0) >= dist - coast:
            break
        if not io.velocity(-speed, 0.0, 0.0):
            return False
        wait(REVERSE_TICK_S)
    if halted():
        return False
    io.velocity(0.0, 0.0, 0.0)
    return True


def social(io: NavIO, sense: Callable[[], tuple[dict, list[Detection]]],
           stop_event: threading.Event, *, aim: Callable[[float | None], object] = None,
           perceiver: Perceiver | None = None, model: ComfortModel | None = None,
           now=time.monotonic, wait=None, on_perception=None,
           max_duration_s: float = MAX_DURATION_S) -> ActionResult:
    """Run the comfort loop until stopped, out of range, or out of time.

    ``sense()`` -> (frames by source, person detections incl. keypoints).
    ``aim(bearing)`` yaws the gripper toward a body bearing (None = centre).
    """
    wait = wait if wait is not None else stop_event.wait
    aim = aim or (lambda b: None)
    perceiver = perceiver or Perceiver()
    model = model or ComfortModel()
    state = SocialState()

    def halted() -> bool:
        return stop_event.is_set() or bool(io.aborted())

    try:
        start = io.odom_pose()
    except Exception as e:
        logger.warning("social: no odometry: %s", e)
        return ActionResult(TEXTS["failed"], False, {"outcome": "failed"})
    t0 = now()
    period = 1.0 / RATE_HZ

    while True:
        if halted():
            return ActionResult(TEXTS["paused"], False, {"outcome": "paused"})
        t = now()
        if t - t0 > max_duration_s:
            break
        try:
            x, y, _ = io.odom_pose()
            odom_ok = True
        except Exception:
            x, y, odom_ok = start[0], start[1], False  # no motion this cycle
        if math.hypot(x - start[0], y - start[1]) > MAX_RADIUS_M - MAX_STEP_M:
            _finish(io, state, aim, halted)
            return ActionResult(TEXTS["limit"], True, {"outcome": "limit"})

        try:
            frames, dets = sense()
        except Exception as e:
            logger.warning("social: sense failed: %s", e)
            wait(period)
            continue
        seen_t = now()
        if halted():
            return ActionResult(TEXTS["paused"], False, {"outcome": "paused"})

        target = pick_target(dets, state.bearing)
        if target is None:
            if seen_t - state.last_seen > 10.0 and state.bearing is not None:
                state.bearing = None
                state.pose.reset()
                aim(None)
            wait(period)
            continue

        view = best_view(dets, target)
        if view.source != state.source:
            state.pose.reset()
        state.source, state.bearing, state.last_seen = view.source, target.bearing_deg, seen_t
        dist, method = best_range(dets, target, frames)
        # Freshness from when the images were taken (capture + inference count).
        shot_t = min((f.t for f in frames.values() if getattr(f, "t", None)), default=None)
        age0 = (time.time() - shot_t) if shot_t is not None else 0.0

        settling = seen_t - state.last_move_t < SETTLE_S
        action = "moving" if settling else "idle"
        perception = perceiver.perceive(frames, view, dist, action, state.pose)
        comfort, behavior = model.predict_behavior(perception)
        if on_perception is not None:
            on_perception(perception, comfort, behavior)
        logger.info("social: %s %.0f deg %s m pose=%s emo=%s gest=%s -> %.2f %s",
                    view.source, target.bearing_deg,
                    "?" if dist is None else f"{dist:.2f}", perception.pose_label,
                    perception.emotion_label, perception.gesture_label, comfort, behavior.value)

        todo = stable_behavior(state, behavior, seen_t)
        if settling or not odom_ok:
            todo = None
        if todo is None or not _moves_body(todo, target.bearing_deg, state):
            if not settling:
                aim(target.bearing_deg)  # never while the body is about to move
        if todo is not None:
            fresh = age0 + (now() - seen_t) <= STALE_S
            if _execute(io, state, todo, target.bearing_deg, dist, method, fresh, dets,
                        halted, wait, now, frames):
                state.last_behavior = todo
        wait(period)

    _finish(io, state, aim, halted)
    return ActionResult(TEXTS["done"], True, {"outcome": "done"})


def _moves_body(behavior, bearing, state) -> bool:
    """True if executing ``behavior`` may move the body (no gaze this cycle)."""
    if behavior in MOVES:
        return True
    return (behavior == BehaviorLabel.SIT) != state.crouched or abs(bearing) > TURN_DEG


def _execute(io, state, behavior, bearing, dist, method, fresh, dets, halted, wait, now,
             frames=None) -> bool:
    """Carry out one behaviour. True if it was done (or nothing was needed)."""
    if halted():
        return False
    if behavior in (BehaviorLabel.STAY, BehaviorLabel.SIT):
        want_crouch = behavior == BehaviorLabel.SIT
        ok = True
        if want_crouch != state.crouched:
            ok = io.posture(CROUCH_M if want_crouch else 0.0)
            state.moved(now())
            if ok:
                state.crouched = want_crouch
        if ok and abs(bearing) > TURN_DEG:
            ok = io.move_body(0.0, 0.0, math.radians(bearing))
            state.moved(now())
            state.crouched = False  # move_body walks at nominal height
        return ok

    if state.crouched:
        if not io.posture(0.0):
            return False
        state.crouched = False
        state.moved(now())

    if behavior in APPROACH:
        stop_m, max_step = APPROACH[behavior]
        b = math.radians(bearing)
        step = 0.0
        if fresh and method in ("keypoints", "depth") and dist is not None and abs(bearing) <= 45.0:
            step = min(max_step, MAX_STEP_M, dist - BODY_MARGIN_M - max(stop_m, MIN_GAP_M))
            if step > 0.05 and not step_gap_ok(dets, bearing, step):
                step = 0.0
        if step <= 0.05:
            if abs(bearing) <= TURN_DEG:
                return True  # already close enough (or no trusted range): stay
            ok = io.move_body(0.0, 0.0, b)
        else:
            ok = io.move_body(step * math.cos(b), step * math.sin(b), b)
        state.moved(now())
        return ok

    if behavior in REVERSE:
        if not fresh or not rear_clear(dets, frames):
            logger.info("social: someone/something behind me, not backing away")
            return False
        d, v = REVERSE[behavior]
        ok = _reverse(io, d, v, halted, wait)
        state.moved(now())
        return ok
    return False


def _finish(io, state, aim, halted) -> None:
    """Natural end: stand back up and centre the gripper (never after a stop)."""
    if halted():
        return
    if state.crouched:
        io.posture(0.0)
    aim(None)
