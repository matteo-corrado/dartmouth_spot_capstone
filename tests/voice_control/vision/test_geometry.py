import numpy as np
import pytest

from src.voice_control.vision.cameras import (
    ROTATION_DEG,
    distance_words,
    pixel_bearing,
    raw_to_upright,
    sector,
    upright,
    upright_to_raw,
)
from src.voice_control.vision.types import CameraFrame


@pytest.mark.parametrize("source", sorted(ROTATION_DEG))
@pytest.mark.parametrize("p", [(0, 0), (639, 479), (320, 240), (100, 400)])
def test_upright_to_raw_roundtrip_tilted(source, p):
    rot = ROTATION_DEG[source]
    up = raw_to_upright(p[0], p[1], 480, 640, rot)
    back = upright_to_raw(up[0], up[1], 480, 640, rot)
    assert back[0] == pytest.approx(p[0], abs=1e-6)
    assert back[1] == pytest.approx(p[1], abs=1e-6)


def test_upright_shape():
    assert upright(np.zeros((480, 640)), "frontleft_fisheye_image").shape == (726, 604)
    raw = np.zeros((480, 640), np.uint8)
    raw[0, 0] = 255
    out = upright(raw, "right_fisheye_image")
    assert out.shape == (480, 640)
    assert out[479, 639] == 255


def _frame(R):
    T = np.eye(4)
    T[:3, :3] = R
    return CameraFrame(
        source="frontleft_fisheye_image",
        image=np.zeros((726, 604), np.uint8),
        raw_rows=480, raw_cols=640, rot_deg=-78.0,
        fx=300.0, fy=300.0, cx=320.0, cy=240.0,
        body_T_sensor=T, depth=None, jpeg=b"", t=0.0,
    )


def test_bearing_uses_transform_not_name():
    # sensor x -> body -x, y -> body -z, z (optical axis) -> body -y (right)
    right = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], float)
    f = _frame(right)
    h, w = f.image.shape[:2]
    assert pixel_bearing(f, (w - 1) / 2, (h - 1) / 2) == pytest.approx(-90, abs=0.5)
    # axis to body +x (ahead): sensor x -> -y_b (right), y -> -z_b, z -> +x_b
    ahead = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
    f = _frame(ahead)
    assert pixel_bearing(f, (w - 1) / 2, (h - 1) / 2) == pytest.approx(0, abs=0.5)


@pytest.mark.parametrize("b,word", [
    (0, "in front"), (30, "front-left"), (90, "on my left"),
    (-135, "behind-right"), (180, "behind me"), (-179, "behind me"),
    (22.4, "in front"), (22.6, "front-left"),
])
def test_sector_boundaries(b, word):
    assert sector(b) == word


def test_distance_words():
    assert distance_words(None) == ""
    assert distance_words(0.6) == "right next to me"
    assert distance_words(1.2) == "about one metre"
    assert distance_words(2.4) == "about two metres"


@pytest.mark.parametrize("rot", [-78.0, -102.0])
def test_upright_content_alignment(rot):
    import cv2
    from src.voice_control.vision.cameras import ROTATION_DEG
    source = next(s for s, r in ROTATION_DEG.items() if r == rot)
    raw = np.zeros((480, 640), np.uint8)
    cv2.circle(raw, (123, 345), 4, 255, -1)
    up = upright(raw, source)
    ys, xs = np.nonzero(up > 127)
    ex, ey = raw_to_upright(123, 345, 480, 640, rot)
    assert abs(xs.mean() - ex) < 1.0 and abs(ys.mean() - ey) < 1.0
