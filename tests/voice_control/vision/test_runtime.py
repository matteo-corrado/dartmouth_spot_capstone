"""Tests for the onnxruntime detector runtime (letterbox, decode, provider fallback)."""
from pathlib import Path

import numpy as np
import pytest

from src.voice_control.vision import runtime

REPO = Path(__file__).resolve().parents[3]
MODELS = ("yolo26l-seg", "yolo26l-pose", "yoloe-26l-seg")
HAND_IMG = REPO / "tests" / "fixtures" / "vision" / "private" / "hand_color_image.jpg"


@pytest.fixture(autouse=True)
def _clear_model_cache():
    runtime._MODELS.clear()
    yield
    runtime._MODELS.clear()


def test_letterbox_grey_to_rgb():
    img = np.full((480, 640), 200, np.uint8)
    x, scale, pad = runtime.letterbox(img)
    assert x.shape == (1, 3, 640, 640)
    assert x.dtype == np.float32
    assert scale == 1.0
    assert pad == (0, 80)
    assert np.allclose(x[0, :, :80, :], 114 / 255)       # top padding
    assert np.allclose(x[0, :, 560:, :], 114 / 255)      # bottom padding
    assert np.allclose(x[0, :, 80:560, :], 200 / 255)    # image, all three channels


def test_letterbox_scales_large_rgb():
    img = np.zeros((960, 1280, 3), np.uint8)
    img[..., 0] = 255                                    # red channel (RGB in)
    x, scale, pad = runtime.letterbox(img)
    assert scale == 0.5 and pad == (0, 80)
    assert np.allclose(x[0, 0, 80:560], 1.0) and np.allclose(x[0, 2, 80:560], 0.0)


def _row(box, conf, cls, extra=()):
    return np.array([*box, conf, cls, *extra], np.float32)


def test_decode_det_rows():
    out0 = np.zeros((1, 300, 6), np.float32)
    out0[0, 0] = _row((100, 180, 300, 380), 0.9, 2)
    out0[0, 1] = _row((10, 90, 50, 130), 0.2, 5)
    dets = runtime.decode([out0], scale=0.5, pad=(0, 80), img_hw=(960, 1280), conf=0.35)
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == 2 and d.conf == pytest.approx(0.9)
    assert d.box == pytest.approx((200, 200, 600, 600))
    assert d.mask is None and d.keypoints is None


def test_decode_seg_mask():
    out0 = np.zeros((1, 300, 6 + 32), np.float32)
    coeffs = np.zeros(32, np.float32)
    coeffs[0] = 1.0
    out0[0, 0] = _row((200, 280, 400, 480), 0.8, 0, coeffs)   # letterbox px
    protos = np.zeros((1, 32, 160, 160), np.float32)
    protos[0, 0] = 10.0                                      # sigmoid(10) ~ 1 everywhere
    dets = runtime.decode([out0, protos], scale=1.0, pad=(0, 80), img_hw=(480, 640), conf=0.35)
    assert len(dets) == 1
    m = dets[0].mask
    assert m.shape == (480, 640) and m.dtype == bool
    assert dets[0].box == pytest.approx((200, 200, 400, 400))
    assert m[205:395, 205:395].all()                         # inside box (image px)
    assert not m[:195].any() and not m[405:].any()           # outside box
    assert not m[:, :195].any() and not m[:, 405:].any()


def test_decode_seg_mask_negative_protos_empty():
    out0 = np.zeros((1, 300, 6 + 32), np.float32)
    coeffs = np.zeros(32, np.float32)
    coeffs[0] = 1.0
    out0[0, 0] = _row((200, 280, 400, 480), 0.8, 0, coeffs)
    protos = np.full((1, 32, 160, 160), -10.0, np.float32)
    dets = runtime.decode([out0, protos], scale=1.0, pad=(0, 80), img_hw=(480, 640), conf=0.35)
    assert not dets[0].mask.any()


def test_decode_pose_keypoints():
    out0 = np.zeros((1, 300, 6 + 51), np.float32)
    kpts = np.zeros((17, 3), np.float32)
    kpts[:, 0] = 300.0
    kpts[:, 1] = 280.0
    kpts[:, 2] = 0.7
    kpts[5] = (100.0, 180.0, 0.9)
    out0[0, 0] = _row((100, 180, 300, 380), 0.9, 0, kpts.ravel())
    dets = runtime.decode([out0], scale=0.5, pad=(0, 80), img_hw=(960, 1280), conf=0.35)
    assert len(dets) == 1
    k = dets[0].keypoints
    assert k.shape == (17, 3)
    assert k[5] == pytest.approx((200.0, 200.0, 0.9))
    assert k[0] == pytest.approx((600.0, 400.0, 0.7))
    assert dets[0].mask is None


class _FakeIO:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape


E2E_SEG = ([1, 300, 38], [1, 32, 160, 160])


def _fake_session_cls(output_shapes, fail_gpu=False, tried=None):
    class FakeSession:
        def __init__(self, path, sess_options=None, providers=None):
            names = [p[0] if isinstance(p, tuple) else p for p in providers]
            if tried is not None:
                tried.append(names[0])
            if fail_gpu and ("TensorrtExecutionProvider" in names or "CUDAExecutionProvider" in names):
                raise RuntimeError("no GPU for you")
            self._providers = names

        def get_providers(self):
            return self._providers

        def get_inputs(self):
            return [_FakeIO("images", [1, 3, 640, 640])]

        def get_outputs(self):
            return [_FakeIO(f"output{i}", s) for i, s in enumerate(output_shapes)]

    return FakeSession


def test_runtime_falls_back_to_cpu(tmp_path, monkeypatch):
    (tmp_path / "yolo26l-seg.onnx").write_bytes(b"")
    tried = []
    FakeSession = _fake_session_cls(E2E_SEG, fail_gpu=True, tried=tried)

    monkeypatch.setattr(runtime.ort, "InferenceSession", FakeSession)
    m = runtime.load_model("yolo26l-seg", model_dir=tmp_path)
    assert m.provider == "cpu"
    assert m.name == "yolo26l-seg"
    assert tried == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
    assert runtime.gpu_available() is False
    assert runtime.load_model("yolo26l-seg", model_dir=tmp_path) is m   # cached
    assert len(tried) == 3


@pytest.mark.parametrize("shapes", [
    ([1, 116, 8400], [1, 32, 160, 160]),   # raw yolo26l-seg head (end2end=False)
    ([1, 56, 8400],),                      # raw yolo26l-pose head
    ([1, 300, 38],),                       # seg rows without the protos output
])
def test_raw_head_layout_raises(tmp_path, monkeypatch, shapes):
    (tmp_path / "yolo26l-seg.onnx").write_bytes(b"")
    monkeypatch.setattr(runtime.ort, "InferenceSession", _fake_session_cls(shapes))
    with pytest.raises(ValueError, match="end2end=True"):
        runtime.load_model("yolo26l-seg", model_dir=tmp_path)
    assert "yolo26l-seg" not in runtime._MODELS


@pytest.mark.parametrize("shapes", [E2E_SEG, ([1, 300, 57],), ([1, 300, 6],)])
def test_end2end_layouts_accepted(tmp_path, monkeypatch, shapes):
    (tmp_path / "m.onnx").write_bytes(b"")
    monkeypatch.setattr(runtime.ort, "InferenceSession", _fake_session_cls(shapes))
    assert runtime.load_model("m", model_dir=tmp_path).provider == "trt"


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        runtime.load_model("yolo26l-seg", model_dir=tmp_path)


def _gpu_ready():
    if not all((runtime.MODEL_DIR / f"{m}.onnx").exists() for m in MODELS):
        return "models/vision/*.onnx missing"
    if "TensorrtExecutionProvider" not in runtime.ort.get_available_providers():
        return "TensorRT provider not available"
    if not HAND_IMG.exists():
        return "private hand_color_image fixture missing"
    return None


@pytest.mark.skipif(_gpu_ready() is not None, reason=str(_gpu_ready()))
@pytest.mark.parametrize("name", MODELS)
def test_gpu_models_agree(name):
    import cv2
    img = cv2.cvtColor(cv2.imread(str(HAND_IMG)), cv2.COLOR_BGR2RGB)
    gpu = runtime.load_model(name)
    assert gpu.provider in ("trt", "cuda")
    assert runtime.gpu_available()
    cpu = runtime._open(name, runtime.MODEL_DIR, kinds=("cpu",))
    assert cpu.provider == "cpu"

    # Detect at 0.35 but only compare confident (> 0.6) detections both ways, so a
    # borderline fp16-vs-fp32 confidence near a threshold cannot flake the test.
    g = gpu.detect(img, conf=0.35)
    c = cpu.detect(img, conf=0.35)

    def box_diff(a, b):
        return max(abs(x - y) for x, y in zip(a.box, b.box))

    confident_g = [d for d in g if d.conf > 0.6]
    assert confident_g, "expected detections (image contains a person)"
    for src, other in ((confident_g, c), ([d for d in c if d.conf > 0.6], g)):
        for d in src:
            same = [x for x in other if x.cls == d.cls]
            assert same, f"class {d.cls} missing on the other provider"
            assert min(box_diff(d, x) for x in same) <= 1.0, d
    for d in confident_g:
        if name != "yolo26l-pose":
            assert d.mask is not None and d.mask.shape == img.shape[:2]
        else:
            assert d.keypoints.shape == (17, 3)
