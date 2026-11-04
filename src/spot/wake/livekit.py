"""LiveKit-Wakeword inference for the Spot voice loop (Stage 2F P3).

spot-env is Python 3.10; the livekit-wakeword pip package needs 3.11 (StrEnum),
so we do NOT import it at runtime. We run its 3-stage ONNX chain directly through
onnxruntime: melspectrogram.onnx -> embedding_model.onnx -> <wake>.onnx (the
trained classifier head). The two front-end models are frozen and shared across
all wake words; only the classifier is custom-trained.

Pinned to livekit-wakeword commit 1ec7f680. API mirrors the sherpa-onnx path so
client_mic.py swaps backends via SPOT_WAKE_BACKEND with no other change:
    is_available() -> bool
    process_frame(pcm16_bytes) -> bool
    reset() -> None
"""
from __future__ import annotations
import json
import os
from pathlib import Path

import numpy as np

WAKE_DIR = Path(os.environ.get("LIVEKIT_WAKE_DIR", "/mnt/ssd/wake-models"))
MEL_MODEL = WAKE_DIR / "melspectrogram.onnx"
EMB_MODEL = WAKE_DIR / "embedding_model.onnx"
# Default = the deploy symlink (active.onnx -> winning candidate) so swapping
# candidates needs no env change (spec Component 1 deploy path).
CLF_MODEL = Path(os.environ.get("LIVEKIT_WAKE_MODEL", str(WAKE_DIR / "active.onnx")))
THRESHOLD_FILE = WAKE_DIR / "active_threshold.json"

SAMPLE_RATE = 16000
# 2 s + one extra 10 ms hop guarantees >= 200 mel frames -> >= 16 embedding
# windows (floor((T-76)/8)+1 >= 16 needs T >= 196), avoiding the boundary case
# where a 2.000 s window emits 199 frames -> 15 windows -> a silent 0.0 score.
WINDOW_SAMPLES = SAMPLE_RATE * 2 + 160
STEP_SAMPLES = int(SAMPLE_RATE * 0.08)    # re-score every 80 ms
EMBEDDING_WINDOW = 76                      # mel frames per embedding
EMBEDDING_STRIDE = 8                       # mel-frame hop
MIN_EMBEDDINGS = 16                        # classifier consumes the last 16


def _load_threshold() -> float:
    """Precedence: env var > deployed threshold file > 0.5 (logged loud).

    Falling back to the default means the Phase C DET sweep was not completed
    for this model — we warn rather than silently ship a guessed threshold.
    """
    env = os.environ.get("LIVEKIT_WAKE_THRESHOLD")
    if env is not None:
        return float(env)
    if THRESHOLD_FILE.exists():
        return float(json.loads(THRESHOLD_FILE.read_text())["threshold"])
    print("[wake livekit] WARNING: no LIVEKIT_WAKE_THRESHOLD and no "
          f"{THRESHOLD_FILE.name}; using default 0.5 (run the Phase C DET sweep)")
    return 0.5


def embedding_windows(mel: np.ndarray) -> np.ndarray:
    """Slide a 76-frame / stride-8 window over mel frames.

    mel: (T, 32). Returns (N, 76, 32); N == 0 when T < 76.
    """
    n_frames = mel.shape[0]
    starts = range(0, n_frames - EMBEDDING_WINDOW + 1, EMBEDDING_STRIDE)
    wins = [mel[s:s + EMBEDDING_WINDOW] for s in starts]
    if not wins:
        return np.empty((0, EMBEDDING_WINDOW, mel.shape[1]), dtype=mel.dtype)
    return np.stack(wins, axis=0)


class LiveKitWakeWord:
    name = "livekit"

    def __init__(self) -> None:
        self._available = False
        self.last_score = 0.0
        for p in (MEL_MODEL, EMB_MODEL, CLF_MODEL):
            if not p.exists():
                print(f"[wake livekit] model missing: {p}")
                print("[wake livekit] train on the cluster + scp the 3 ONNX to /mnt/ssd/wake-models")
                return
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._mel = ort.InferenceSession(str(MEL_MODEL), providers=providers)
        self._emb = ort.InferenceSession(str(EMB_MODEL), providers=providers)
        self._clf = ort.InferenceSession(str(CLF_MODEL), providers=providers)
        self._mel_in = self._mel.get_inputs()[0].name
        self._emb_in = self._emb.get_inputs()[0].name
        self._clf_in = self._clf.get_inputs()[0].name
        self.threshold = _load_threshold()
        self.buffer = np.zeros(0, dtype=np.int16)
        self._pending = 0
        self._available = True
        print(f"[wake livekit] 3-stage chain loaded, threshold={self.threshold}")

    def is_available(self) -> bool:
        return self._available

    def reset(self) -> None:
        if self._available:
            self.buffer = np.zeros(0, dtype=np.int16)
            self._pending = 0
            self.last_score = 0.0

    def _score(self, window_i16: np.ndarray) -> float:
        audio = window_i16.astype(np.float32) / 32768.0
        mel = self._mel.run(None, {self._mel_in: audio[np.newaxis, :]})[0]
        if mel.ndim == 4:
            mel = mel[:, 0, :, :]                         # (1,1,T,32) -> (1,T,32)
        mel = mel[0]                                      # (T, 32)
        assert mel.ndim == 2 and mel.shape[1] == 32, f"unexpected mel shape {mel.shape}"
        mel = mel / 10.0 + 2.0                            # Google front-end calibration
        wins = embedding_windows(mel)
        if wins.shape[0] < MIN_EMBEDDINGS:
            return 0.0
        batch = wins[..., np.newaxis].astype(np.float32)  # (N, 76, 32, 1)
        emb = self._emb.run(None, {self._emb_in: batch})[0].squeeze(axis=(1, 2))  # (N, 96)
        seq = emb[-MIN_EMBEDDINGS:][np.newaxis, :, :].astype(np.float32)          # (1, 16, 96)
        return float(self._clf.run(None, {self._clf_in: seq})[0][0, 0])

    def process_frame(self, pcm16_bytes: bytes) -> bool:
        if not self._available:
            return False
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        self.buffer = np.concatenate([self.buffer, samples])[-WINDOW_SAMPLES:]
        self._pending += len(samples)
        if len(self.buffer) < WINDOW_SAMPLES or self._pending < STEP_SAMPLES:
            return False
        self._pending = 0
        self.last_score = self._score(self.buffer)
        return self.last_score >= self.threshold
