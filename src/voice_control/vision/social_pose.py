"""Friendly Spot rule-based pose labels on YOLO26-pose COCO-17 keypoints.

Port of friendly_spot "Facial Recognition/streamlinedRuleBasedEstimation.py"
(MediaPipe 33 landmarks). Same rules, same order, same thresholds. Changes:

- COCO-17 indices instead of MediaPipe ones (no heels/feet/fingers are used).
- Coordinates are 2-D and normalised by the person box height instead of the
  frame size, so a person far away in a fisheye reads like one close in the
  gripper camera. Low-confidence keypoints are NaN and fail the rules that
  need them.
- Motion compares the new pose with the previous one (the original appended
  first and compared the pose with itself, so walking/running never fired).
- Torso angle measured from upright (the original read ~180 deg when upright).
- ``reset()`` clears the history; call it when the camera or the robot moved.
"""
from __future__ import annotations

from collections import deque

import numpy as np

# COCO-17 keypoint indices
NOSE = 0
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

KP_MIN_CONF = 0.3

# Original thresholds
MIN_BUFFER_MOTION = 3
MIN_WAVING = 8
STRAIGHT_KNEE = 120
UPRIGHT_TORSO = 30
MIN_WALK_SPEED = 0.02
MAX_WALK_SPEED = 0.15
SIMILAR_MEASUREMENT = 0.1
MIN_WAVE_MOTION = 0.1
MIN_KNEE_ANGLE_DIFFERENCE_WALKING = 30
MIN_HEAD_DIP_FOR_BEND = 0.25


def normalise(keypoints: np.ndarray, box) -> np.ndarray:
    """17x3 (x, y, conf) px -> 17x2 in person-box-height units; NaN where conf is low."""
    kps = np.asarray(keypoints, dtype=float)
    h = max(float(box[3]) - float(box[1]), 1.0)
    pts = kps[:, :2] / h
    pts[kps[:, 2] < KP_MIN_CONF] = np.nan
    return pts


def _ok(*pts) -> bool:
    return all(p is not None and np.all(np.isfinite(p)) for p in pts)


def _dist(a, b) -> float:
    if not _ok(a, b):
        return float("inf")
    return float(np.linalg.norm(a - b))


def _angle(p1, p2, p3) -> float:
    """Angle at p2 (degrees); 0 when a point is missing (as the original)."""
    if not _ok(p1, p2, p3):
        return 0.0
    v1, v2 = p1 - p2, p3 - p2
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


class PoseClassifier:
    """Feed one person's normalised keypoints per frame; returns a pose label."""

    def __init__(self, buffer_size: int = 30):
        self.buffer: deque = deque(maxlen=buffer_size)

    def reset(self) -> None:
        self.buffer.clear()

    # -- helpers --------------------------------------------------------------
    def _knees(self, lm):
        return (_angle(lm[LEFT_HIP], lm[LEFT_KNEE], lm[LEFT_ANKLE]),
                _angle(lm[RIGHT_HIP], lm[RIGHT_KNEE], lm[RIGHT_ANKLE]))

    def _torso_angle(self, lm) -> float:
        sc = (lm[LEFT_SHOULDER] + lm[RIGHT_SHOULDER]) / 2
        hc = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2
        if not _ok(sc, hc):
            return 0.0
        # Image y grows downward, so an upright torso has shoulders above hips:
        # use hip_y - shoulder_y (the original used shoulder_y - hip_y, which
        # reads ~180 deg for an upright person).
        return abs(float(np.degrees(np.arctan2(sc[0] - hc[0], hc[1] - sc[1]))))

    def _speed(self, lm) -> float:
        if len(self.buffer) < MIN_BUFFER_MOTION:
            return 0.0
        d = lm - self.buffer[-1]
        d = d[np.all(np.isfinite(d), axis=1)]
        return float(np.linalg.norm(d)) if d.size else 0.0

    # -- rules (original order) -------------------------------------------------
    def is_standing(self, lm) -> bool:
        lk, rk = self._knees(lm)
        return (lk > STRAIGHT_KNEE and rk > STRAIGHT_KNEE
                and self._torso_angle(lm) < UPRIGHT_TORSO
                and _dist(lm[LEFT_ANKLE], lm[RIGHT_ANKLE]) != float("inf")
                and _dist(lm[LEFT_ANKLE], lm[RIGHT_ANKLE]) > SIMILAR_MEASUREMENT)

    def is_walking(self, lm, speed) -> bool:
        if len(self.buffer) < MIN_BUFFER_MOTION:
            return False
        lk, rk = self._knees(lm)
        if not _ok(lm[LEFT_KNEE], lm[RIGHT_KNEE]):
            return False
        return (abs(lk - rk) > MIN_KNEE_ANGLE_DIFFERENCE_WALKING
                and MIN_WALK_SPEED < speed < MAX_WALK_SPEED
                and self._torso_angle(lm) < UPRIGHT_TORSO)

    def is_running(self, lm, speed) -> bool:
        if len(self.buffer) < MIN_BUFFER_MOTION:
            return False
        if not _ok(lm[LEFT_KNEE], lm[RIGHT_KNEE], lm[LEFT_ANKLE], lm[RIGHT_ANKLE]):
            return False
        lk, rk = self._knees(lm)
        return lk < STRAIGHT_KNEE and rk < STRAIGHT_KNEE and speed > MAX_WALK_SPEED

    def is_waving(self, lm) -> bool:
        if len(self.buffer) < MIN_BUFFER_MOTION:
            return False
        lw, rw, ls, rs = lm[LEFT_WRIST], lm[RIGHT_WRIST], lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        recent = list(self.buffer)[-MIN_WAVING:] + [lm]

        def x_range(idx):
            xs = [p[idx][0] for p in recent if np.isfinite(p[idx][0])]
            return (max(xs) - min(xs)) if len(xs) >= 2 else 0.0

        left = _ok(lw, ls) and lw[1] < ls[1] and x_range(LEFT_WRIST) > MIN_WAVE_MOTION
        right = _ok(rw, rs) and rw[1] < rs[1] and x_range(RIGHT_WRIST) > MIN_WAVE_MOTION
        return bool(left or right)

    def is_raising_hand(self, lm) -> bool:
        nose = lm[NOSE]
        if not _ok(nose):
            # Face turned away: fall back to the shoulders, a bit stricter.
            sh = [p for p in (lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]) if _ok(p)]
            if not sh:
                return False
            top = min(p[1] for p in sh) - 0.15
        else:
            top = nose[1]
        return any(_ok(w) and w[1] < top for w in (lm[LEFT_WRIST], lm[RIGHT_WRIST]))

    def is_clapping(self, lm) -> bool:
        if len(self.buffer) < MIN_BUFFER_MOTION:
            return False
        lw, rw, ls, rs = lm[LEFT_WRIST], lm[RIGHT_WRIST], lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER]
        if not _ok(lw, rw, ls, rs):
            return False
        return _dist(lw, rw) < SIMILAR_MEASUREMENT and lw[1] < ls[1] and rw[1] < rs[1]

    def is_bending(self, lm) -> bool:
        nose = lm[NOSE]
        hc = (lm[LEFT_HIP] + lm[RIGHT_HIP]) / 2
        if not _ok(nose, hc):
            return False
        return (self._torso_angle(lm) > UPRIGHT_TORSO
                and abs(nose[1] - hc[1]) < MIN_HEAD_DIP_FOR_BEND)

    def is_arms_crossed(self, lm) -> bool:
        ls, rs, lw, rw = lm[LEFT_SHOULDER], lm[RIGHT_SHOULDER], lm[LEFT_WRIST], lm[RIGHT_WRIST]
        if not _ok(ls, rs, lw, rw):
            return False
        # A wrist across the body sits nearer the opposite shoulder (in x). The
        # original compared raw x against one shoulder, which ignores which way
        # the person faces and fired whenever the hands were level.
        left_across = abs(lw[0] - rs[0]) < abs(lw[0] - ls[0])
        right_across = abs(rw[0] - ls[0]) < abs(rw[0] - rs[0])
        return (left_across or right_across) and abs(lw[1] - rw[1]) < SIMILAR_MEASUREMENT

    def classify(self, lm: np.ndarray) -> str:
        """Label for this frame, then remember it for the motion rules."""
        speed = self._speed(lm)
        try:
            if self.is_running(lm, speed):
                return "running"
            if self.is_walking(lm, speed):
                return "walking"
            if self.is_clapping(lm):
                return "clapping"
            if self.is_waving(lm):
                return "waving"
            if self.is_raising_hand(lm):
                return "raising_hand"
            if self.is_bending(lm):
                return "bending"
            if self.is_arms_crossed(lm):
                return "arms_crossed"
            if self.is_standing(lm):
                return "standing"
            return "unknown"
        finally:
            self.buffer.append(lm)
