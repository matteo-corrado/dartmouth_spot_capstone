"""World snapshot: camera frames -> labelled detections with bearing and distance."""
from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .cameras import BODY_SOURCES, capture, mask_distance, pixel_bearing
from .runtime import load_model
from .types import CameraFrame, Detection
from .vocab import COCO_NAMES, YOLOE_NAMES

logger = logging.getLogger(__name__)

# Static camera table: (centre yaw, half-FOV) in degrees, +left.
_STATIC_COVERAGE: dict[str, tuple[float, float]] = {
    "frontleft_fisheye_image": (0.0, 60.0),
    "frontright_fisheye_image": (0.0, 60.0),
    "left_fisheye_image": (90.0, 50.0),
    "right_fisheye_image": (-90.0, 50.0),
    "back_fisheye_image": (180.0, 50.0),
}


def _angle_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two bearings, degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


def cameras_covering(bearing_deg: float,
                     frames: dict[str, CameraFrame] | None = None) -> list[str]:
    """Body cameras whose horizontal FOV contains ``bearing_deg`` (+left), nearest axis first, max 2."""
    hits = []
    if frames is None:
        # The front cameras are cross-mounted (frontleft looks front-right), so for
        # a tie prefer frontright on the left side and frontleft on the right side.
        prefer = ("frontright_fisheye_image" if bearing_deg > 0
                  else "frontleft_fisheye_image" if bearing_deg < 0 else None)
        for order, (src, (centre, half)) in enumerate(_STATIC_COVERAGE.items()):
            off = _angle_diff(bearing_deg, centre)
            if off <= half:
                hits.append((off, 0 if src == prefer else 1, order, src))
    else:
        for order, src in enumerate(BODY_SOURCES):
            f = frames.get(src)
            if f is None:
                continue
            centre, lo, hi = _interval(f)
            rel = _signed_diff(bearing_deg, centre)
            if lo <= rel <= hi:
                hits.append((abs(rel), 0, order, src))
    hits.sort()
    return [h[-1] for h in hits[:2]]


def _signed_diff(a: float, b: float) -> float:
    """a - b wrapped to (-180, 180]."""
    d = (a - b) % 360.0
    return d - 360.0 if d > 180.0 else d


def _interval(f: CameraFrame) -> tuple[float, float, float]:
    """(centre bearing, lo, hi): horizontal coverage of the upright image, relative to centre."""
    h, w = f.image.shape[:2]
    centre = pixel_bearing(f, (w - 1) / 2.0, (h - 1) / 2.0)
    edges = [_signed_diff(pixel_bearing(f, x, (h - 1) / 2.0), centre) for x in (0.0, w - 1.0)]
    return centre, min(edges), max(edges)


def _names_for(model: str) -> list[str]:
    return YOLOE_NAMES if "yoloe" in model else COCO_NAMES


def detect_frames(frames: dict[str, CameraFrame], model: str = "yolo26l-seg",
                  labels: set[str] | None = None, conf: float = 0.35) -> list[Detection]:
    """Run ``model`` on each frame's upright image; sorted by confidence descending."""
    names = _names_for(model)
    net = load_model(model)
    out: list[Detection] = []
    for src, frame in frames.items():
        img = frame.image
        if img.ndim == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2 frames are BGR, model wants RGB
        try:
            raw = net.detect(img, conf)
        except Exception as e:
            logger.warning("detect: skipping %s (%s: %s)", src, type(e).__name__, e)
            continue
        for r in raw:
            if not 0 <= r.cls < len(names):
                continue
            label = names[r.cls]
            if labels is not None and label not in labels:
                continue
            x0, y0, x1, y1 = (float(v) for v in r.box)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            if r.mask is not None:
                ys, xs = np.nonzero(r.mask)
                if xs.size:
                    cx, cy = float(xs.mean()), float(ys.mean())
            dist, method = mask_distance(frame, r.mask, (x0, y0, x1, y1))
            out.append(Detection(
                label=label, conf=float(r.conf), source=src, box=(x0, y0, x1, y1),
                mask=r.mask, keypoints=r.keypoints,
                bearing_deg=pixel_bearing(frame, cx, cy),
                distance_m=dist, distance_method=method))
    out.sort(key=lambda d: d.conf, reverse=True)
    return out


def snapshot(robot, sources: list[str] = BODY_SOURCES, model: str = "yolo26l-seg",
             labels: set[str] | None = None, with_depth: bool = True) -> list[Detection]:
    """Capture ``sources`` and detect on them."""
    frames = capture(robot, list(sources), with_depth=with_depth)
    return detect_frames(frames, model=model, labels=labels)


def dedupe(dets: list[Detection], bearing_tol: float = 10.0,
           dist_tol: float = 0.7) -> list[Detection]:
    """Collapse the same label seen by overlapping cameras to the highest-conf one."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.conf, reverse=True):
        for k in kept:
            if k.label != d.label or _angle_diff(k.bearing_deg, d.bearing_deg) > bearing_tol:
                continue
            if (k.distance_m is not None and d.distance_m is not None
                    and abs(k.distance_m - d.distance_m) > dist_tol):
                continue
            break
        else:
            kept.append(d)
    return kept
