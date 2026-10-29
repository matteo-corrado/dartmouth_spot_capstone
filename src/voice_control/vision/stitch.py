"""OpenCV re-implementation of Boston Dynamics' stitch_front_images (no OpenGL).

Virtual eye from BD's sum of the two camera positions, a plane 2 m ahead along
the summed camera axes, gluPerspective(110), and BD's blend score (pow 10).
Verified pixel-close to BD's OpenGL output. Maps depend only on calibration, so
they are cached.
"""
from __future__ import annotations

import cv2
import numpy as np

from .types import CameraFrame

FOVY, DIST = 110.0, 2.0
_MAPS: dict[tuple, list] = {}


def _score(u, v, cols, rows):
    # BD frag shader calculateScore, on 0..1 image coords
    lx = 1 - np.abs(u / cols * 2 - 1)
    ly = 1 - np.abs(v / rows * 2 - 1)
    s = lx * ly * (1 - 0.001) + 0.001
    s = np.where((lx < 0) | (ly < 0), 0.0, s)
    return s ** 10


def _grey(frame: CameraFrame) -> np.ndarray:
    img = frame.raw if frame.raw is not None else frame.image
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _key(*frames: CameraFrame) -> tuple:
    out = []
    for f in frames:
        T = np.asarray(f.body_T_sensor, dtype=float)
        out.append((f.source, f.raw_rows, f.raw_cols,
                    round(f.fx, 2), round(f.fy, 2), round(f.cx, 2), round(f.cy, 2),
                    tuple(np.round(T[:3, 3] * 1000).astype(int)),          # 1 mm
                    tuple(np.round(T[:3, :3].ravel() * 573).astype(int))))  # ~0.1 deg
    return tuple(out)


def _build_maps(fr: CameraFrame, fl: CameraFrame, size: tuple[int, int]):
    """Per-camera remap grids and weights for the virtual front view."""
    W, H = size
    Ts = [np.asarray(f.body_T_sensor, dtype=float) for f in (fr, fl)]
    eye = Ts[0][:3, 3] + Ts[1][:3, 3]  # BD sums positions (not average) - kept as-is
    n = Ts[0][:3, :3] @ [0, 0, 1.0] + Ts[1][:3, :3] @ [0, 0, 1.0]
    eye[1] = 0
    n[1] = 0
    n /= np.linalg.norm(n)
    plane = eye + n * DIST
    s = np.cross(n, [0, 0, 1])
    s /= np.linalg.norm(s)
    u = np.cross(s, n)
    fy = 1 / np.tan(np.radians(FOVY) / 2)
    xs = (2 * (np.arange(W) + 0.5) / W - 1) * (W / H) / fy
    ys = (1 - 2 * (np.arange(H) + 0.5) / H) / fy
    X, Y = np.meshgrid(xs, ys)
    d = n[None, None] + X[..., None] * s + Y[..., None] * u  # ray dirs, body frame
    t = ((plane - eye) @ n) / (d @ n)
    P = eye + t[..., None] * d  # points on plane
    Ph = np.concatenate([P, np.ones(P.shape[:2] + (1,))], -1)
    out = []
    for f, T in zip((fr, fl), Ts):
        q = Ph @ np.linalg.inv(T).T
        z = q[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = (f.fx * q[..., 0] / z + f.cx).astype(np.float32)
            mv = (f.fy * q[..., 1] / z + f.cy).astype(np.float32)
        rows, cols = _grey(f).shape
        w = np.where(z > 0, _score(mu, mv, cols, rows), 0.0)
        out.append((mu, mv, w))
    return out


def stitch_front(fr: CameraFrame, fl: CameraFrame, size=(1080, 720)) -> np.ndarray:
    """Stitch frontright + frontleft fisheyes into one grey front view (uint8, H x W)."""
    W, H = size
    key = (tuple(size),) + _key(fr, fl)
    maps = _MAPS.get(key)
    if maps is None:
        if len(_MAPS) >= 4:
            _MAPS.clear()
        maps = _MAPS[key] = _build_maps(fr, fl, size)
    acc = np.zeros((H, W))
    tot = np.zeros((H, W))
    for f, (mu, mv, w) in zip((fr, fl), maps):
        # GL texel centres sit at (i+0.5)/N, so pixel index = coord - 0.5; GL also
        # never samples black outside the texture, so replicate the edge.
        acc += cv2.remap(_grey(f), mu - 0.5, mv - 0.5, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REPLICATE) * w
        tot += w
    out = np.zeros((H, W))
    np.divide(acc, tot, out=out, where=tot > 0)  # edge weights reach ~1e-30; don't clamp
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)
