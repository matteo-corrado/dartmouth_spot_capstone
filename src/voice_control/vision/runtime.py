"""GPU detector runtime: onnxruntime sessions (TensorRT -> CUDA -> CPU), letterbox, decode.

Ultralytics is never imported here (it auto-installs packages into the venv and
cannot use the GPU in spot-env). Models are exported off-env with
``end2end=True`` (NMS-free head), so decoding is a threshold plus un-letterbox.

Confirmed output layout (2026-10-29, Ultralytics 8.4.192 export, imgsz 640,
checked row-by-row against Ultralytics' own prediction on hand_color_image.jpg):

  yolo26l-seg    output0 [1, 300, 38]  output1 [1, 32, 160, 160] (mask protos)
  yoloe-26l-seg  output0 [1, 300, 38]  output1 [1, 32, 160, 160] (150 classes)
  yolo26l-pose   output0 [1, 300, 57]

  output0 rows: [x1, y1, x2, y2, conf, cls, *extra], boxes in letterbox px
  (corner format), conf already post-sigmoid, rows sorted by conf descending.
  seg extra  = 32 mask coefficients -> sigmoid(coeffs @ protos) at 160x160.
  pose extra = 17 x (x, y, conf) keypoints, x/y in letterbox px.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

MODEL_DIR = Path(os.environ.get("SPOT_VISION_MODEL_DIR")
                 or Path(__file__).resolve().parents[3] / "models" / "vision")

INPUT_SIZE = 640
PAD_VALUE = 114
N_MASK_COEFFS = 32
N_KPT_VALUES = 17 * 3

_PROVIDER = {"trt": "TensorrtExecutionProvider", "cuda": "CUDAExecutionProvider",
             "cpu": "CPUExecutionProvider"}


@dataclass
class RawDet:
    cls: int
    conf: float
    box: tuple                       # (x1, y1, x2, y2) in input-image px
    mask: np.ndarray | None = None   # bool HxW of the input image (seg models)
    keypoints: np.ndarray | None = None  # (17, 3) x, y in input-image px, conf


def letterbox(img: np.ndarray, size: int = INPUT_SIZE) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize + centre-pad to ``size`` square, as Ultralytics' LetterBox does.

    ``img`` is HxW grey or HxWx3 **RGB** uint8. Returns (NCHW float32 0-1 RGB
    tensor, scale, pad) where pad = (left, top) in letterbox px and
    input px = (letterbox px - pad) / scale.
    """
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nw, nh = round(w * scale), round(h * scale)
    if (nw, nh) != (w, h):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    left = round((size - nw) / 2 - 0.1)
    top = round((size - nh) / 2 - 0.1)
    canvas = np.full((size, size, 3), PAD_VALUE, np.uint8)
    canvas[top:top + nh, left:left + nw] = img
    x = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return np.ascontiguousarray(x), scale, (left, top)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _decode_mask(coeffs: np.ndarray, protos: np.ndarray, box_lb: np.ndarray,
                 scale: float, pad: tuple[int, int], img_hw: tuple[int, int]) -> np.ndarray:
    """One instance mask: sigmoid(coeffs @ protos) -> 640, crop to box, un-letterbox, > 0.5."""
    c, ph, pw = protos.shape
    # einsum, not @: spot-env's BLAS gemv is ~14 ms here (thread contention) vs 0.3 ms.
    logits = np.einsum("k,kn->n", coeffs.astype(np.float32), protos.reshape(c, -1))
    prob = _sigmoid(logits).reshape(ph, pw).astype(np.float32)
    prob = cv2.resize(prob, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    x1, y1, x2, y2 = np.clip(np.round(box_lb), 0, INPUT_SIZE).astype(int)
    cropped = np.zeros_like(prob)
    cropped[y1:y2, x1:x2] = prob[y1:y2, x1:x2]
    h, w = img_hw
    left, top = pad
    nw, nh = round(w * scale), round(h * scale)
    region = cropped[top:top + nh, left:left + nw]
    if region.shape[:2] != (h, w):
        region = cv2.resize(region, (w, h), interpolation=cv2.INTER_LINEAR)
    return region > 0.5


def decode(outputs: list[np.ndarray], scale: float, pad: tuple[int, int],
           img_hw: tuple[int, int], conf: float = 0.35) -> list[RawDet]:
    """Decode end-to-end YOLO26 outputs into RawDets in input-image px.

    The model kind follows from the row width: 6 = boxes only, 6 + 32 with a
    protos output = segmentation, 6 + 51 = pose.
    """
    rows = outputs[0][0]
    protos = outputs[1][0] if len(outputs) > 1 else None
    extra = rows.shape[1] - 6
    is_seg = protos is not None and extra == N_MASK_COEFFS
    is_pose = extra == N_KPT_VALUES
    h, w = img_hw
    left, top = pad
    offset = np.array([left, top, left, top], np.float32)
    dets = []
    for row in rows[rows[:, 4] >= conf]:
        box_lb = row[:4]
        box = (box_lb - offset) / scale
        box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
        mask = kpts = None
        if is_seg:
            mask = _decode_mask(row[6:], protos, box_lb, scale, pad, img_hw)
        elif is_pose:
            kpts = row[6:].reshape(17, 3).astype(np.float32).copy()
            kpts[:, 0] = (kpts[:, 0] - left) / scale
            kpts[:, 1] = (kpts[:, 1] - top) / scale
        dets.append(RawDet(cls=int(row[5]), conf=float(row[4]),
                           box=tuple(float(v) for v in box), mask=mask, keypoints=kpts))
    return dets


class Model:
    """One ONNX model on one execution provider; inference serialised by a lock."""

    def __init__(self, name: str, session, provider: str):
        self.name = name
        self.provider = provider
        self._session = session
        self._input = session.get_inputs()[0].name
        self._lock = threading.Lock()

    def detect(self, img: np.ndarray, conf: float = 0.35) -> list[RawDet]:
        """Detect on ``img``: HxW grey or HxWx3 **RGB** uint8.

        cv2-decoded images (e.g. the hand colour camera JPEG) are BGR; callers
        must convert with ``cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`` first.
        Returned boxes, masks and keypoints are in ``img`` pixels.
        """
        x, scale, pad = letterbox(img)
        with self._lock:
            outputs = self._session.run(None, {self._input: x})
        return decode(outputs, scale, pad, img.shape[:2], conf)


def _providers(kind: str, model_dir: Path) -> list:
    if kind == "trt":
        cache = str(model_dir / "trt_cache")
        return [("TensorrtExecutionProvider", {
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": cache,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": cache}),
                "CUDAExecutionProvider", "CPUExecutionProvider"]
    if kind == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


_E2E_ROWS = 300
_E2E_WIDTHS = (6, 6 + N_MASK_COEFFS, 6 + N_KPT_VALUES)


def _check_layout(name: str, session) -> None:
    """Refuse a non-end2end export: decoding a raw head would give silent garbage."""
    outs = session.get_outputs()
    shape = list(outs[0].shape)
    ok = (len(shape) == 3 and shape[1] == _E2E_ROWS and shape[2] in _E2E_WIDTHS
          and (shape[2] != 6 + N_MASK_COEFFS or len(outs) >= 2))
    if not ok:
        raise ValueError(
            f"{name}: output0 shape {shape} is not the end-to-end YOLO26 layout "
            f"[1, {_E2E_ROWS}, 6|38|57]; re-export the model with end2end=True")


def _open(name: str, model_dir: Path, kinds: tuple[str, ...] = ("trt", "cuda", "cpu")) -> Model:
    """Create an uncached Model, trying providers in ``kinds`` order."""
    path = Path(model_dir) / f"{name}.onnx"
    if not path.exists():
        raise FileNotFoundError(path)
    if "trt" in kinds:
        (Path(model_dir) / "trt_cache").mkdir(parents=True, exist_ok=True)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    err: Exception | None = None
    for kind in kinds:
        try:
            session = ort.InferenceSession(str(path), sess_options=opts,
                                           providers=_providers(kind, Path(model_dir)))
        except Exception as e:  # provider init / engine build failure -> next provider
            err = e
            continue
        _check_layout(name, session)
        active = session.get_providers()[0]   # ORT may silently drop a failing provider
        provider = next((k for k, v in _PROVIDER.items() if v == active), "cpu")
        return Model(name, session, provider)
    raise RuntimeError(f"could not create a session for {name}") from err


_MODELS: dict[str, Model] = {}
_FAILED: dict[tuple[str, str], Exception] = {}  # (name, model dir) -> load error
_MODELS_LOCK = threading.Lock()


class ModelUnavailable(RuntimeError):
    """A cached earlier load failure: raised at once, no new session build."""


def load_model(name: str, model_dir: Path = MODEL_DIR) -> Model:
    """Load (once) and return ``name``; TensorRT fp16 -> CUDA -> CPU.

    A failed load is cached too: later calls raise ``ModelUnavailable`` at once
    instead of retrying the (possibly minutes-long) build under ``_MODELS_LOCK``
    while a caller holds the dispatcher lock. ``reset_models()`` forgets both."""
    key = (name, str(Path(model_dir).resolve()))
    with _MODELS_LOCK:
        model = _MODELS.get(name)
        if model is not None:
            return model
        err = _FAILED.get(key)
        if err is not None:
            raise ModelUnavailable(f"{name} failed to load earlier: {err}") from err
        try:
            model = _MODELS[name] = _open(name, Path(model_dir))
        except Exception as e:
            _FAILED[key] = e
            raise
        return model


def reset_models() -> None:
    """Forget loaded models and cached load failures (tests, setup scripts)."""
    with _MODELS_LOCK:
        _MODELS.clear()
        _FAILED.clear()


def gpu_available() -> bool:
    """True if any loaded model runs on TensorRT or CUDA."""
    return any(m.provider in ("trt", "cuda") for m in list(_MODELS.values()))
