"""Camera geometry: upright rotation, pixel mapping, bearings, spoken wording."""
from __future__ import annotations

import math
import time

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

# Depth source registered to each visual source (verified on the robot, Task 2 step 5).
DEPTH_SOURCE: dict[str, str] = {
    "frontleft_fisheye_image": "frontleft_depth_in_visual_frame",
    "frontright_fisheye_image": "frontright_depth_in_visual_frame",
    "left_fisheye_image": "left_depth_in_visual_frame",
    "right_fisheye_image": "right_depth_in_visual_frame",
    "back_fisheye_image": "back_depth_in_visual_frame",
    "hand_color_image": "hand_depth_in_hand_color_frame",
}
GROUND_Z_DEFAULT = -0.5  # body-frame floor height when the snapshot has no ground plane
MAX_GROUND_RANGE_M = 15.0
MIN_DEPTH_PIXELS = 20


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


def frame_from_response(resp, depth_resp=None) -> CameraFrame:
    """Build a CameraFrame from one ImageResponse (and its optional depth response)."""
    from bosdyn.client.frame_helpers import (BODY_FRAME_NAME, GROUND_PLANE_FRAME_NAME,
                                             get_a_tform_b)
    name = resp.source.name
    jpeg = bytes(resp.shot.image.data)
    raw = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise ValueError(f"could not decode image for {name}")
    rows, cols = raw.shape[:2]
    intr = resp.source.pinhole.intrinsics
    snap = resp.shot.transforms_snapshot
    T = get_a_tform_b(snap, BODY_FRAME_NAME, resp.shot.frame_name_image_sensor).to_matrix()
    depth = None
    if depth_resp is not None:
        dimg = depth_resp.shot.image
        arr = np.frombuffer(dimg.data, np.uint16).reshape(dimg.rows, dimg.cols)
        scale = depth_resp.source.depth_scale or 1.0
        depth = arr.astype(np.float32) / scale
        depth[arr == 0] = np.nan
        if depth.shape != (rows, cols):
            depth = cv2.resize(depth, (cols, rows), interpolation=cv2.INTER_NEAREST)
    frame = CameraFrame(
        source=name, image=upright(raw, name), raw_rows=rows, raw_cols=cols,
        rot_deg=ROTATION_DEG[name], fx=intr.focal_length.x, fy=intr.focal_length.y,
        cx=intr.principal_point.x, cy=intr.principal_point.y,
        body_T_sensor=np.asarray(T, dtype=float), depth=depth, jpeg=jpeg,
        t=time.time(), raw=raw,
    )
    # Floor height in the body frame; not a CameraFrame field, read with getattr.
    ground = get_a_tform_b(snap, BODY_FRAME_NAME, GROUND_PLANE_FRAME_NAME)
    frame.ground_z = float(ground.z) if ground is not None else GROUND_Z_DEFAULT
    return frame


def capture(robot, sources: list[str], with_depth: bool = False) -> dict[str, CameraFrame]:
    """One get_image call for all sources (plus their depth sources)."""
    from bosdyn.api import image_pb2
    from bosdyn.client.image import ImageClient, build_image_request
    client = robot.ensure_client(ImageClient.default_service_name)
    reqs = [build_image_request(s, image_format=image_pb2.Image.FORMAT_JPEG) for s in sources]
    if with_depth:
        reqs += [build_image_request(DEPTH_SOURCE[s],
                                     pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
                                     image_format=image_pb2.Image.FORMAT_RAW)
                 for s in sources if s in DEPTH_SOURCE]
    by_name = {r.source.name: r for r in client.get_image(reqs)}
    out = {}
    for s in sources:
        if s in by_name:
            out[s] = frame_from_response(by_name[s], by_name.get(DEPTH_SOURCE.get(s)))
    return out


def _ground_distance(frame: CameraFrame, box_up) -> float | None:
    x0, y0, x1, y1 = box_up
    origin, d = pixel_ray_body(frame, (x0 + x1) / 2.0, y1)
    gz = getattr(frame, "ground_z", GROUND_Z_DEFAULT)
    if d[2] >= -1e-6:
        return None
    t = (gz - origin[2]) / d[2]
    if t <= 0 or t > MAX_GROUND_RANGE_M:
        return None
    hit = origin + t * d
    return float(math.hypot(hit[0], hit[1]))


def mask_distance(frame: CameraFrame, mask_up: np.ndarray | None,
                  box_up) -> tuple[float | None, str | None]:
    """Distance to a detection: median depth inside the mask, else ground plane."""
    if frame.depth is not None:
        if mask_up is None:
            x0, y0, x1, y1 = (int(round(v)) for v in box_up)
            mask_up = np.zeros(frame.image.shape[:2], bool)
            mask_up[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
        m = cv2.erode(mask_up.astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
        M, _ = _rot_matrix(frame.raw_rows, frame.raw_cols, frame.rot_deg)
        raw_mask = cv2.warpAffine(m, M, (frame.raw_cols, frame.raw_rows),
                                  flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP) > 0
        vals = frame.depth[raw_mask]
        vals = vals[np.isfinite(vals)]
        if vals.size >= MIN_DEPTH_PIXELS:
            return float(np.median(vals)), "depth"
    g = _ground_distance(frame, box_up)
    return (g, "ground") if g is not None else (None, None)
