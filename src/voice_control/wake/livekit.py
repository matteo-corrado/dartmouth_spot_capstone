"""LiveKit-Wakeword inference wrapper for the Spot voice loop.

API mirrors WakeWordDetector (sherpa-onnx path) so client_mic.py can swap
backends via SPOT_WAKE_BACKEND without further changes:
    is_available() -> bool
    process_frame(pcm16_bytes) -> bool
    reset() -> None
"""
from __future__ import annotations
import os
from pathlib import Path

import numpy as np

LIVEKIT_MODEL = Path(
    os.environ.get("LIVEKIT_WAKE_MODEL", "/mnt/ssd/wake-models/hey_spot.onnx")
)
SAMPLE_RATE = 16000
WAKE_THRESHOLD = float(os.environ.get("LIVEKIT_WAKE_THRESHOLD", "0.55"))
WINDOW_DURATION_S = 2.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_DURATION_S)
STEP_SAMPLES = int(SAMPLE_RATE * 0.08)


class LiveKitWakeWord:
    name = "livekit"

    def __init__(self) -> None:
        self._available = False
        self.model = None
        if not LIVEKIT_MODEL.exists():
            print(f"[wake livekit] model missing: {LIVEKIT_MODEL}")
            print(f"[wake livekit] run scripts/training/train_wake_hey_spot.sh to produce it")
            return
        try:
            from livekit.wakeword import WakeWordModel
        except ImportError:
            print("[wake livekit] livekit-wakeword not installed; run scripts/training/setup_livekit_wakeword.sh")
            return
        self.model = WakeWordModel(models=[str(LIVEKIT_MODEL)])
        self.model_key = LIVEKIT_MODEL.stem
        self.buffer = np.zeros(0, dtype=np.int16)
        self._available = True
        print(f"[wake livekit] loaded {LIVEKIT_MODEL.name}, threshold={WAKE_THRESHOLD}")

    def is_available(self) -> bool:
        return self._available

    def reset(self) -> None:
        if self._available:
            self.buffer = np.zeros(0, dtype=np.int16)

    def process_frame(self, pcm16_bytes: bytes) -> bool:
        if not self._available:
            return False
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16)
        self.buffer = np.concatenate([self.buffer, samples])
        detected = False
        while len(self.buffer) >= WINDOW_SAMPLES:
            window = self.buffer[-WINDOW_SAMPLES:]
            scores = self.model.predict(window)
            score = float(scores.get(self.model_key, 0.0))
            if score >= WAKE_THRESHOLD:
                detected = True
            self.buffer = self.buffer[STEP_SAMPLES:]
        return detected
