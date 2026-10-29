from pathlib import Path

import cv2
import numpy as np
import pytest

from src.voice_control.vision.cameras import frame_from_response, mask_distance
from src.voice_control.vision.types import CameraFrame

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures/vision/private/front_pair.bin"


def _cam(pitch_deg, depth=None, rot_deg=0.0, rows=480, cols=640, ground_z=None):
    """Camera at the body origin, optical axis pitched down by pitch_deg (neg = up)."""
    c, s = np.cos(np.radians(pitch_deg)), np.sin(np.radians(pitch_deg))
    z = np.array([c, 0.0, -s])
    x = np.array([0.0, -1.0, 0.0])
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, :3] = np.column_stack([x, y, z])
    img = np.zeros((rows, cols), np.uint8)
    f = CameraFrame(
        source="right_fisheye_image" if rot_deg == 180 else "back_fisheye_image",
        image=img, raw_rows=rows, raw_cols=cols, rot_deg=rot_deg,
        fx=300.0, fy=300.0, cx=cols / 2, cy=rows / 2,
        body_T_sensor=T, depth=depth, jpeg=b"", t=0.0,
    )
    if ground_z is not None:
        f.ground_z = ground_z
    return f


def test_mask_distance_uses_depth_median():
    depth = np.full((480, 640), 5.0, np.float32)
    mask = np.zeros((480, 640), bool)
    mask[100:300, 200:400] = True
    depth[mask] = 2.0
    f = _cam(30, depth=depth)
    assert mask_distance(f, mask, (200, 100, 400, 300)) == (pytest.approx(2.0), "depth")


def test_mask_distance_depth_maps_through_rotation():
    # 180 deg camera: upright mask in the top-left is the raw bottom-right.
    depth = np.full((480, 640), 5.0, np.float32)
    depth[300:420, 400:600] = 2.0
    f = _cam(30, depth=depth, rot_deg=180.0)
    mask = np.zeros((480, 640), bool)
    mask[60:180, 40:240] = True
    d, m = mask_distance(f, mask, (40, 60, 240, 180))
    assert m == "depth" and d == pytest.approx(2.0)


def test_distance_zero_depth_falls_back_to_ground():
    depth = np.full((480, 640), np.nan, np.float32)
    f = _cam(30, depth=depth, ground_z=-0.5)
    mask = np.zeros((480, 640), bool)
    mask[100:240, 280:360] = True
    d, m = mask_distance(f, mask, (280, 100, 360, 240))
    assert m == "ground"
    assert d == pytest.approx(0.5 / np.tan(np.radians(30)), abs=0.05)


def test_ground_without_depth_and_default_floor():
    f = _cam(30, depth=None)  # default floor is -0.5 m
    d, m = mask_distance(f, None, (280, 100, 360, 240))
    assert m == "ground"
    assert d == pytest.approx(0.866, abs=0.05)


def test_ground_ray_above_horizon_returns_none():
    f = _cam(-30, depth=None)
    assert mask_distance(f, None, (280, 100, 360, 240)) == (None, None)


def test_ground_ray_too_far_returns_none():
    f = _cam(1, depth=None)  # hits floor at ~28 m
    assert mask_distance(f, None, (280, 100, 360, 240)) == (None, None)


@pytest.mark.skipif(not FIXTURE.exists(), reason="private front_pair fixture missing")
def test_frame_from_saved_response():
    from bosdyn.api import image_pb2
    msg = image_pb2.GetImageResponse()
    msg.ParseFromString(FIXTURE.read_bytes())
    resp = next(r for r in msg.image_responses if r.source.name == "frontleft_fisheye_image")
    f = frame_from_response(resp)
    assert f.image.shape[:2] == (726, 604)
    assert f.raw is not None and f.raw.shape[:2] == (f.raw_rows, f.raw_cols)
    assert f.fx > 0 and f.depth is None
    assert f.body_T_sensor.shape == (4, 4)


@pytest.mark.skipif(not FIXTURE.exists(), reason="private front_pair fixture missing")
def test_capture_tolerates_one_bad_camera(caplog):
    from bosdyn.api import image_pb2
    from src.voice_control.vision.cameras import capture
    msg = image_pb2.GetImageResponse()
    msg.ParseFromString(FIXTURE.read_bytes())
    bad = next(r for r in msg.image_responses if r.source.name == "frontleft_fisheye_image")
    bad.shot.image.data = b"not a jpeg"

    class _Client:
        def get_image(self, reqs):
            return list(msg.image_responses)

    class _Robot:
        def ensure_client(self, name):
            return _Client()

    srcs = ["frontright_fisheye_image", "frontleft_fisheye_image"]
    with caplog.at_level("WARNING"):
        out = capture(_Robot(), srcs)
    assert list(out) == ["frontright_fisheye_image"]
    assert "frontleft_fisheye_image" in caplog.text


def test_mask_distance_ground_z_field_default_none():
    assert _cam(30).ground_z is None
