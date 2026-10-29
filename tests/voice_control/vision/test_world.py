import numpy as np
import pytest

from src.voice_control.vision import world
from src.voice_control.vision.cameras import BODY_SOURCES, pixel_bearing
from src.voice_control.vision.runtime import RawDet
from src.voice_control.vision.types import CameraFrame, Detection
from src.voice_control.vision.vocab import COCO_NAMES, YOLOE_NAMES


def _frame(source="frontleft_fisheye_image", yaw_deg=0.0, depth=None, h=480, w=640):
    a = np.radians(yaw_deg)
    # optical axis (sensor z) -> body (cos yaw, sin yaw, 0); sensor y -> body -z
    z = np.array([np.cos(a), np.sin(a), 0.0])
    y = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    return CameraFrame(
        source=source, image=np.zeros((h, w, 3), np.uint8), raw_rows=h, raw_cols=w,
        rot_deg=0.0, fx=300.0, fy=300.0, cx=w / 2, cy=h / 2, body_T_sensor=T,
        depth=depth, jpeg=b"", t=0.0)


class FakeModel:
    provider = "cpu"

    def __init__(self, dets):
        self.dets = dets
        self.seen = []

    def detect(self, img, conf=0.35):
        self.seen.append(img)
        return list(self.dets)


def _patch(monkeypatch, dets, names=None):
    m = FakeModel(dets)
    asked = []
    monkeypatch.setattr(world, "load_model", lambda n: (asked.append(n), m)[1])
    return m, asked


def test_detect_frames_fills_bearing_and_distance(monkeypatch):
    f = _frame(depth=np.full((480, 640), 3.0, np.float32))
    mask = np.zeros((480, 640), bool)
    mask[200:280, 400:480] = True  # centroid (439.5, 239.5)
    m, asked = _patch(monkeypatch, [RawDet(0, 0.9, (400, 200, 480, 280), mask=mask)])
    out = world.detect_frames({f.source: f})
    assert asked == ["yolo26l-seg"]
    assert len(out) == 1
    d = out[0]
    assert d.label == COCO_NAMES[0] and d.source == f.source
    assert d.bearing_deg == pytest.approx(pixel_bearing(f, 439.5, 239.5), abs=0.05)
    assert d.bearing_deg < 0  # right of axis -> negative (+left)
    assert d.distance_m == pytest.approx(3.0) and d.distance_method == "depth"


def test_box_centre_used_without_mask_and_rgb_conversion(monkeypatch):
    f = _frame()
    f.image[..., 0] = 10  # B channel
    f.image[..., 2] = 200  # R channel
    m, _ = _patch(monkeypatch, [RawDet(0, 0.8, (300, 200, 340, 280))])
    out = world.detect_frames({f.source: f})
    assert out[0].bearing_deg == pytest.approx(pixel_bearing(f, 320, 240), abs=1e-6)
    assert m.seen[0][0, 0, 0] == 200  # RGB handed to the model


def test_yoloe_names_by_model(monkeypatch):
    f = _frame()
    _patch(monkeypatch, [RawDet(5, 0.9, (300, 200, 340, 280))])
    out = world.detect_frames({f.source: f}, model="yoloe-26l-seg")
    assert out[0].label == YOLOE_NAMES[5]


def test_labels_filter_and_sorted(monkeypatch):
    f = _frame()
    _patch(monkeypatch, [RawDet(0, 0.5, (0, 0, 40, 40)), RawDet(56, 0.7, (100, 100, 140, 140)),
                         RawDet(0, 0.9, (200, 200, 240, 240))])
    out = world.detect_frames({f.source: f})
    assert [d.conf for d in out] == [0.9, 0.7, 0.5]
    out = world.detect_frames({f.source: f}, labels={"person"})
    assert {d.label for d in out} == {"person"} and len(out) == 2


def test_inference_error_skips_frame(monkeypatch):
    f1, f2 = _frame("a"), _frame("b")

    class Boom(FakeModel):
        def detect(self, img, conf=0.35):
            if len(self.seen) == 0:
                self.seen.append(img)
                raise RuntimeError("CUDA error")
            return super().detect(img, conf)

    m = Boom([RawDet(0, 0.9, (300, 200, 340, 280))])
    monkeypatch.setattr(world, "load_model", lambda n: m)
    out = world.detect_frames({"a": f1, "b": f2})
    assert [d.source for d in out] == ["b"]


def _det(label="person", conf=0.9, bearing=2.0, dist=2.0, source="frontleft_fisheye_image"):
    return Detection(label, conf, source, (0, 0, 1, 1), None, None, bearing, dist,
                     "depth" if dist is not None else None)


def test_dedupe_overlapping_front_cameras():
    a = _det(conf=0.8, bearing=2.0, source="frontleft_fisheye_image")
    b = _det(conf=0.9, bearing=3.5, dist=2.3, source="frontright_fisheye_image")
    out = world.dedupe([a, b])
    assert out == [b]


def test_dedupe_keeps_distinct():
    base = _det()
    assert len(world.dedupe([base, _det(label="chair")])) == 2
    assert len(world.dedupe([base, _det(bearing=20.0)])) == 2
    assert len(world.dedupe([base, _det(dist=3.0)])) == 2
    # missing distance -> bearing only
    assert len(world.dedupe([base, _det(dist=None, conf=0.5)])) == 1


def test_dedupe_wraparound():
    assert len(world.dedupe([_det(bearing=179.0), _det(bearing=-178.0, conf=0.5)])) == 1


def test_snapshot_captures_then_detects(monkeypatch):
    f = _frame()
    calls = {}
    monkeypatch.setattr(world, "capture", lambda r, s, with_depth: (calls.update(s=s, d=with_depth), {f.source: f})[1])
    _patch(monkeypatch, [RawDet(0, 0.9, (300, 200, 340, 280))])
    out = world.snapshot("robot")
    assert calls["s"] == list(BODY_SOURCES) and calls["d"] is True
    assert len(out) == 1


@pytest.mark.parametrize("bearing,expected", [
    (0, ["frontleft_fisheye_image", "frontright_fisheye_image"]),
    (90, ["left_fisheye_image"]),
    (-90, ["right_fisheye_image"]),
    (175, ["back_fisheye_image"]),
    (-175, ["back_fisheye_image"]),
    (180, ["back_fisheye_image"]),
])
def test_cameras_covering_static(bearing, expected):
    assert world.cameras_covering(bearing) == expected


def test_cameras_covering_static_overlap_nearest_first_max_two():
    out = world.cameras_covering(50)
    assert len(out) == 2 and out[0] == "left_fisheye_image"
    assert world.cameras_covering(45.0 + 360) == world.cameras_covering(45.0)


def test_cameras_covering_with_frames_uses_transform():
    # cross-mounted: "frontleft" actually looks right
    frames = {"frontleft_fisheye_image": _frame("frontleft_fisheye_image", yaw_deg=-20),
              "frontright_fisheye_image": _frame("frontright_fisheye_image", yaw_deg=20),
              "back_fisheye_image": _frame("back_fisheye_image", yaw_deg=180)}
    # half-FOV = atan(320/300) = 46.8 deg
    assert world.cameras_covering(25, frames)[0] == "frontright_fisheye_image"
    assert world.cameras_covering(-25, frames)[0] == "frontleft_fisheye_image"
    assert world.cameras_covering(0, frames) == ["frontleft_fisheye_image", "frontright_fisheye_image"] or \
        world.cameras_covering(0, frames) == ["frontright_fisheye_image", "frontleft_fisheye_image"]
    assert world.cameras_covering(180, frames) == ["back_fisheye_image"]
    assert world.cameras_covering(90, frames) == []
