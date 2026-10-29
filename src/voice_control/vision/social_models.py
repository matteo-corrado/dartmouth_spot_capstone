"""Face emotion and hand gesture models for social mode (replace DeepFace and
MediaPipe gestures from friendly_spot).

- Face: OpenCV YuNet (face_detection_yunet_2023mar.onnx, MIT), CPU, ~10 ms.
- Emotion: HSEmotion enet_b0_8_va_mtl.onnx (AffectNet 8 classes, Apache-2.0),
  labels mapped to the DeepFace names the comfort model's weights use.
- Gesture: HaGRIDv2 YOLOv10n (hagridv2_yolov10n_gestures.onnx, NMS-free,
  34 classes), labels mapped to the MediaPipe names the comfort model uses.

Both callables take an upright camera image (grey or BGR) and a person box
(upright px) and look only inside that person. Models live in
``models/vision/social/`` (gitignored; see scripts/setup_vision_models.py).
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .runtime import MODEL_DIR, _providers

logger = logging.getLogger(__name__)

SOCIAL_DIR = MODEL_DIR / "social"
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
EMOTION_MODEL = "enet_b0_8_va_mtl.onnx"
GESTURE_MODEL = "hagridv2_yolov10n_gestures.onnx"

FACE_SCORE = 0.6
FACE_MARGIN = 0.1
GESTURE_CONF = 0.4
GESTURE_SIZE = 640

AFFECTNET = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")
TO_DEEPFACE = {"Anger": "angry", "Contempt": "disgust", "Disgust": "disgust", "Fear": "fear",
               "Happiness": "happy", "Neutral": "neutral", "Sadness": "sad", "Surprise": "surprise"}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

HAGRID = ("grabbing", "grip", "holy", "point", "call", "three3", "timeout", "xsign",
          "hand_heart", "hand_heart2", "little_finger", "middle_finger", "take_picture",
          "dislike", "fist", "four", "like", "mute", "ok", "one", "palm", "peace",
          "peace_inverted", "rock", "stop", "stop_inverted", "three", "three2", "two_up",
          "two_up_inverted", "three_gun", "thumb_index", "thumb_index2", "no_gesture")
TO_MEDIAPIPE = {"like": "thumb_up", "dislike": "thumb_down", "peace": "victory",
                "peace_inverted": "victory", "palm": "open_palm", "stop": "open_palm",
                "stop_inverted": "open_palm", "fist": "closed_fist", "one": "pointing_up",
                "rock": "iloveyou"}


def _session(path: Path) -> ort.InferenceSession:
    if not path.exists():
        raise FileNotFoundError(path)
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    err = None
    for kind in ("trt", "cuda", "cpu"):
        try:
            return ort.InferenceSession(str(path), sess_options=opts,
                                        providers=_providers(kind, MODEL_DIR))
        except Exception as e:
            err = e
    raise RuntimeError(f"could not load {path.name}") from err


def _bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img


def _crop(img, box, pad: float = 0.0, top_frac: float = 1.0):
    """Crop ``box`` (x0, y0, x1, y1) grown by ``pad`` and cut to its top ``top_frac``.
    Returns (crop, x offset, y offset) or None."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = (float(v) for v in box)
    bw, bh = x1 - x0, y1 - y0
    x0, x1 = x0 - pad * bw, x1 + pad * bw
    y0, y1 = y0 - pad * bh, y0 + (bh * top_frac) + pad * bh
    x0, y0 = int(max(0, x0)), int(max(0, y0))
    x1, y1 = int(min(w, x1)), int(min(h, y1))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img[y0:y1, x0:x1], x0, y0


class FaceEmotion:
    """(label, scores, face_box) for the largest face in a person's upper body."""

    def __init__(self, model_dir: Path = SOCIAL_DIR):
        self.face = cv2.FaceDetectorYN.create(str(model_dir / FACE_MODEL), "", (320, 320),
                                              FACE_SCORE, 0.3, 50)
        self.net = _session(model_dir / EMOTION_MODEL)
        self.inp = self.net.get_inputs()[0].name

    def detect_face(self, img, person_box):
        c = _crop(_bgr(img), person_box, pad=0.1, top_frac=0.55)
        if c is None:
            return None
        crop, ox, oy = c
        self.face.setInputSize((crop.shape[1], crop.shape[0]))
        _, faces = self.face.detect(crop)
        if faces is None or len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])[:4]
        return (float(x + ox), float(y + oy), float(w), float(h))

    def classify(self, face_bgr) -> dict[str, float]:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)
        logits = self.net.run(None, {self.inp: x})[0][0, :8]
        e = np.exp(logits - logits.max())
        p = e / e.sum()
        scores: dict[str, float] = {}
        for name, v in zip(AFFECTNET, p):
            key = TO_DEEPFACE[name]
            scores[key] = scores.get(key, 0.0) + float(v)
        return scores

    def __call__(self, img, person_box):
        fb = self.detect_face(img, person_box)
        if fb is None:
            return None, None, None
        x, y, w, h = fb
        c = _crop(_bgr(img), (x, y, x + w, y + h), pad=FACE_MARGIN)
        if c is None:
            return None, None, None
        scores = self.classify(c[0])
        return max(scores, key=scores.get), scores, (int(x), int(y), int(w), int(h))


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def decode_gestures(out: np.ndarray, conf: float = GESTURE_CONF):
    """[1, 300, 6] (x1, y1, x2, y2, score, cls) -> [(name, score, box)], deduped, no 'no_gesture'."""
    rows = [r for r in out[0] if r[4] > conf and 0 <= int(r[5]) < len(HAGRID)
            and HAGRID[int(r[5])] != "no_gesture"]
    rows.sort(key=lambda r: -r[4])
    kept = []
    for r in rows:
        if all(_iou(r[:4], k[:4]) <= 0.7 for k in kept):
            kept.append(r)
    return [(HAGRID[int(r[5])], float(r[4]), tuple(float(v) for v in r[:4])) for r in kept]


def format_gestures(hands, person_cx: float) -> str:
    """Comfort-model gesture label: 'none', 'Hand: <g>' or 'Left: <g>, Right: <g>'
    (image left/right of the person's centre; the model only compares the two)."""
    mapped = [(TO_MEDIAPIPE.get(n, "unknown"), (b[0] + b[2]) / 2) for n, _, b in hands[:2]]
    if not mapped:
        return "none"
    if len(mapped) == 1:
        return f"Hand: {mapped[0][0]}"
    mapped.sort(key=lambda m: m[1])
    return f"Left: {mapped[0][0]}, Right: {mapped[1][0]}"


class HandGesture:
    """Gesture label for the hands inside a person box."""

    def __init__(self, model_dir: Path = SOCIAL_DIR):
        self.net = _session(model_dir / GESTURE_MODEL)
        self.inp = self.net.get_inputs()[0].name

    def detect(self, img_bgr):
        h, w = img_bgr.shape[:2]
        r = min(GESTURE_SIZE / h, GESTURE_SIZE / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        px, py = (GESTURE_SIZE - nw) / 2, (GESTURE_SIZE - nh) / 2
        left, top = int(round(px - 0.1)), int(round(py - 0.1))
        canvas = np.full((GESTURE_SIZE, GESTURE_SIZE, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(img_bgr, (nw, nh),
                                                          interpolation=cv2.INTER_LINEAR)
        x = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        out = self.net.run(None, {self.inp: x.transpose(2, 0, 1)[None]})[0]
        hands = []
        for name, score, (x0, y0, x1, y1) in decode_gestures(out):
            hands.append((name, score, ((x0 - left) / r, (y0 - top) / r,
                                        (x1 - left) / r, (y1 - top) / r)))
        return hands

    def __call__(self, img, person_box) -> str:
        c = _crop(_bgr(img), person_box, pad=0.15)
        if c is None:
            return "none"
        crop, _, _ = c
        return format_gestures(self.detect(crop), crop.shape[1] / 2)


_CACHE: dict[str, object] = {}


def _load(key, cls):
    """Load once; a failure is cached too (no retry of a slow build mid-dispatch)."""
    if key not in _CACHE:
        try:
            _CACHE[key] = cls()
        except Exception as e:
            _CACHE[key] = e
    v = _CACHE[key]
    if isinstance(v, Exception):
        raise v
    return v


def load_emotion() -> FaceEmotion:
    return _load("emotion", FaceEmotion)


def load_gesture() -> HandGesture:
    return _load("gesture", HandGesture)


def warm() -> None:
    """Startup: load both and run one dummy pass (builds TensorRT engines)."""
    blank = np.zeros((480, 640, 3), np.uint8)
    for name, loader in (("emotion", load_emotion), ("gesture", load_gesture)):
        try:
            m = loader()
            if name == "emotion":
                m.classify(blank[:64, :64])  # no face in a blank frame: run the net directly
            else:
                m(blank, (0, 0, 640, 480))
            print(f"[Vision] social {name} model ready")
        except Exception as e:  # noqa: BLE001 - social mode runs without it
            print(f"[Vision] social {name} model unavailable: {type(e).__name__}: {e}")


def loaded(key: str):
    """The model if warm-up loaded it, else None (never loads)."""
    v = _CACHE.get(key)
    return None if v is None or isinstance(v, Exception) else v
