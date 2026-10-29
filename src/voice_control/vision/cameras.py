"""Camera geometry: upright rotation, pixel mapping, bearings, spoken wording."""
from __future__ import annotations

import math

import cv2
import numpy as np

from .types import CameraFrame

# Degrees CCW to make each image upright (BD rotation table).
ROTATION_DEG: dict[str, float] = {
    "back_fisheye_image": 0.0,
    "frontleft_fisheye_image": -78.0,
    "frontright_fisheye_image": -102.0,
    "left_fisheye_image": 0.0,
    "right_fisheye_image": 180.0,
    "hand_color_image": 0.0,
}
BODY_SOURCES = (
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
)
HAND_SOURCE = "hand_color_image"


def _canvas_size(rows: int, cols: int, rot_deg: float) -> tuple[int, int]:
    """Expanded (W', H'), computed like PIL rotate(expand=True)."""
    a = math.radians(rot_deg)
    c, s = math.cos(a), math.sin(a)
    xs, ys = [], []
    for x, y in ((-cols / 2, -rows / 2), (cols / 2, -rows / 2),
                 (cols / 2, rows / 2), (-cols / 2, rows / 2)):
        xs.append(x * c - y * s)
        ys.append(x * s + y * c)
    w = math.ceil(round(max(xs), 9)) - math.floor(round(min(xs), 9))
    h = math.ceil(round(max(ys), 9)) - math.floor(round(min(ys), 9))
    return int(w), int(h)


def _rot_matrix(rows: int, cols: int, rot_deg: float) -> tuple[np.ndarray, tuple[int, int]]:
    """2x3 raw->upright affine (rotate CCW about centre, into expanded canvas)."""
    w, h = _canvas_size(rows, cols, rot_deg)
    m = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), rot_deg, 1.0)
    m[0, 2] += (w - 1) / 2.0 - (cols - 1) / 2.0
    m[1, 2] += (h - 1) / 2.0 - (rows - 1) / 2.0
    return m, (w, h)


def upright(img: np.ndarray, source: str) -> np.ndarray:
    rows, cols = img.shape[:2]
    m, size = _rot_matrix(rows, cols, ROTATION_DEG[source])
    return cv2.warpAffine(img, m, size, flags=cv2.INTER_LINEAR)


def raw_to_upright(x: float, y: float, raw_rows: int, raw_cols: int,
                   rot_deg: float) -> tuple[float, float]:
    m, _ = _rot_matrix(raw_rows, raw_cols, rot_deg)
    return (float(m[0, 0] * x + m[0, 1] * y + m[0, 2]),
            float(m[1, 0] * x + m[1, 1] * y + m[1, 2]))


def upright_to_raw(x: float, y: float, raw_rows: int, raw_cols: int,
                   rot_deg: float) -> tuple[float, float]:
    m, _ = _rot_matrix(raw_rows, raw_cols, rot_deg)
    inv = cv2.invertAffineTransform(m)
    return (float(inv[0, 0] * x + inv[0, 1] * y + inv[0, 2]),
            float(inv[1, 0] * x + inv[1, 1] * y + inv[1, 2]))


def pixel_ray_body(frame: CameraFrame, x_up: float, y_up: float) -> tuple[np.ndarray, np.ndarray]:
    """Ray through an upright pixel: (origin, unit direction) in the body frame."""
    u, v = upright_to_raw(x_up, y_up, frame.raw_rows, frame.raw_cols, frame.rot_deg)
    ray = np.array([(u - frame.cx) / frame.fx, (v - frame.cy) / frame.fy, 1.0])
    T = np.asarray(frame.body_T_sensor, dtype=float)
    d = T[:3, :3] @ ray
    return T[:3, 3].copy(), d / np.linalg.norm(d)


def pixel_bearing(frame: CameraFrame, x_up: float, y_up: float) -> float:
    """Body-frame bearing in degrees, +left (CCW), 0 = straight ahead."""
    _, d = pixel_ray_body(frame, x_up, y_up)
    return math.degrees(math.atan2(d[1], d[0]))


_SECTORS = ("in front", "front-left", "on my left", "behind-left",
            "behind me", "behind-right", "on my right", "front-right")


def sector(bearing_deg: float) -> str:
    return _SECTORS[int(math.floor((bearing_deg + 22.5) / 45.0)) % 8]


_NUMBER_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven",
                 "eight", "nine", "ten")


def distance_words(d: float | None) -> str:
    if d is None:
        return ""
    if d < 1.0:
        return "right next to me"
    n = int(math.floor(d + 0.5))
    if n == 1:
        return "about one metre"
    word = _NUMBER_WORDS[n] if n < len(_NUMBER_WORDS) else str(n)
    return f"about {word} metres"
