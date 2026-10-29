"""Per-camera person tracker and cross-camera hand-off (spec 6.5).

A small IoU + constant-velocity Kalman tracker (Ultralytics' BYTETracker needs
torch, which spot-env must not import). One ``CameraTracker`` per camera; boxes
are upright-image pixels.
"""
from __future__ import annotations

import dataclasses
import math

from .types import Detection

IOU_MIN = 0.3
MAX_AGE_S = 1.0          # a track unseen this long dies
Q_ACCEL = 400.0          # process noise (px/s^2)^2 scale for the centre
R_MEAS = 25.0            # measurement noise, px^2


class _Axis:
    """1-D constant-velocity Kalman filter (position, velocity)."""

    def __init__(self, p: float):
        self.x, self.v = p, 0.0
        self.P = [[R_MEAS, 0.0], [0.0, 1e4]]

    def predict_at(self, dt: float) -> float:
        return self.x + self.v * dt

    def step(self, dt: float, z: float) -> None:
        (a, b), (c, d) = self.P
        # predict: F = [[1, dt], [0, 1]], Q from a white-noise acceleration model
        x = self.x + self.v * dt
        q = Q_ACCEL
        a2 = a + dt * (b + c) + dt * dt * d + q * dt ** 4 / 4
        b2 = b + dt * d + q * dt ** 3 / 2
        c2 = c + dt * d + q * dt ** 3 / 2
        d2 = d + q * dt * dt
        # update with z (H = [1, 0])
        s = a2 + R_MEAS
        k0, k1 = a2 / s, c2 / s
        y = z - x
        self.x = x + k0 * y
        self.v = self.v + k1 * y
        self.P = [[(1 - k0) * a2, (1 - k0) * b2], [c2 - k1 * a2, d2 - k1 * b2]]


class _Track:
    def __init__(self, tid: int, box, t: float):
        x1, y1, x2, y2 = box
        self.id = tid
        self.cx, self.cy = _Axis((x1 + x2) / 2), _Axis((y1 + y2) / 2)
        self.w, self.h = x2 - x1, y2 - y1
        self.t = t

    def predicted_box(self, t: float):
        dt = t - self.t
        cx, cy = self.cx.predict_at(dt), self.cy.predict_at(dt)
        return (cx - self.w / 2, cy - self.h / 2, cx + self.w / 2, cy + self.h / 2)

    def correct(self, box, t: float) -> None:
        x1, y1, x2, y2 = box
        dt = max(t - self.t, 0.0)
        self.cx.step(dt, (x1 + x2) / 2)
        self.cy.step(dt, (y1 + y2) / 2)
        self.w, self.h = x2 - x1, y2 - y1
        self.t = t


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class CameraTracker:
    """Assigns stable ``track_id``s to one camera's detections over time."""

    def __init__(self):
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, dets: list[Detection], t: float) -> list[Detection]:
        """Match ``dets`` (taken at time ``t``) to live tracks; returns copies with ``track_id`` set."""
        self._tracks = [tr for tr in self._tracks if t - tr.t <= MAX_AGE_S]
        pairs = []
        for ti, tr in enumerate(self._tracks):
            pb = tr.predicted_box(t)
            for di, d in enumerate(dets):
                s = iou(pb, d.box)
                if s >= IOU_MIN:
                    pairs.append((s, ti, di))
        pairs.sort(reverse=True)
        used_t, ids = set(), {}
        for s, ti, di in pairs:  # greedy, highest IoU first
            if ti in used_t or di in ids:
                continue
            used_t.add(ti)
            tr = self._tracks[ti]
            tr.correct(dets[di].box, t)
            ids[di] = tr.id
        out = []
        for di, d in enumerate(dets):
            if di not in ids:
                tr = _Track(self._next_id, d.box, t)
                self._next_id += 1
                self._tracks.append(tr)
                ids[di] = tr.id
            out.append(dataclasses.replace(d, track_id=ids[di]))
        return out


def _wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def _xy(d: Detection):
    b = math.radians(d.bearing_deg)
    return d.distance_m * math.cos(b), d.distance_m * math.sin(b)


def _has_dist(d: Detection) -> bool:
    return d.distance_m is not None and math.isfinite(d.distance_m) and d.distance_m >= 0


UNRANGED_MAX_DEG = 15.0   # hand-off from an unranged sighting: bearing limit ...
UNRANGED_MAX_DM = 1.5     # ... and the candidate's range within this of the last known range


def gate(last: Detection, c: Detection, max_dist: float = 1.0, max_bearing: float = 30.0,
         last_range: float | None = None) -> float | None:
    """Score (lower = closer) if ``c`` may be the same person as ``last``, else None.

    Ranged ``last``: ``c`` needs a range, within ``max_bearing`` deg and ``max_dist`` m
    (body-frame xy). Unranged ``last``: ``c`` needs a range within ``UNRANGED_MAX_DM``
    of ``last_range`` (the last measured range) and a bearing within
    ``min(max_bearing, UNRANGED_MAX_DEG)``; with no ``last_range`` nothing passes."""
    if not _has_dist(c):
        return None
    db = abs(_wrap_deg(c.bearing_deg - last.bearing_deg))
    if _has_dist(last):
        if db > max_bearing:
            return None
        (lx, ly), (cx, cy) = _xy(last), _xy(c)
        dist = math.hypot(cx - lx, cy - ly)
        return dist if dist <= max_dist else None
    if last_range is None or not math.isfinite(last_range):
        return None
    if db > min(max_bearing, UNRANGED_MAX_DEG) or abs(c.distance_m - last_range) > UNRANGED_MAX_DM:
        return None
    return db


def handoff(last: Detection, candidates: list[Detection], max_dist: float = 1.0,
            max_bearing: float = 30.0, last_range: float | None = None) -> Detection | None:
    """The candidate nearest ``last`` that passes ``gate`` (body-frame position from
    bearing + distance), else None."""
    best = None
    for c in candidates:
        score = gate(last, c, max_dist, max_bearing, last_range)
        if score is not None and (best is None or score < best[0]):
            best = (score, c)
    return best[1] if best else None
