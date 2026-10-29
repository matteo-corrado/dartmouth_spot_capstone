"""Plain data types shared across the vision package."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CameraFrame:
    source: str
    image: np.ndarray  # upright, HxW or HxWx3 uint8
    raw_rows: int
    raw_cols: int
    rot_deg: float
    fx: float
    fy: float
    cx: float
    cy: float
    body_T_sensor: np.ndarray  # 4x4
    depth: np.ndarray | None  # raw-frame metres
    jpeg: bytes
    t: float
    raw: np.ndarray | None = None  # raw-orientation image, used by the stitch


@dataclass
class Detection:
    label: str
    conf: float
    source: str
    box: tuple[float, float, float, float]  # upright px
    mask: np.ndarray | None  # upright, bool
    keypoints: np.ndarray | None  # 17x3 upright x, y, conf
    bearing_deg: float
    distance_m: float | None
    distance_method: str | None  # "depth" | "ground" | None
    track_id: int | None = None


@dataclass
class ActionResult:
    text: str
    ok: bool
    data: dict = field(default_factory=dict)
