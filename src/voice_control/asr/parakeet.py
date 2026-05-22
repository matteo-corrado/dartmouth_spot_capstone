"""Parakeet TDT 0.6B v3 backend via onnx-asr (CUDA EP). Rollback for Nemotron."""
import os
from collections import Counter

import numpy as np
import onnx_asr

SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.15
PARAKEET_MODEL_ID = os.environ.get("PARAKEET_MODEL_ID", "nemo-parakeet-tdt-0.6b-v3")
PARAKEET_HF_HOME = os.environ.get("PARAKEET_HF_HOME", "/mnt/ssd/parakeet-models")

HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thanks for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "see you next time",
    "bye", "goodbye", "you", "the end", ".", "...",
}


def _is_repetitive(text: str, threshold: float = 0.7) -> bool:
    words = text.strip().lower().split()
    if len(words) < 4:
        return False
    return Counter(words).most_common(1)[0][1] / len(words) > threshold


class ParakeetBackend:
    name = "parakeet"

    def __init__(self) -> None:
        os.environ["HF_HOME"] = PARAKEET_HF_HOME
        print(f"[Parakeet] loading {PARAKEET_MODEL_ID} from {PARAKEET_HF_HOME}")
        self.model = onnx_asr.load_model(
            PARAKEET_MODEL_ID,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("[Parakeet] warming up")
        # onnx-asr requires float32 (or path); int16 raises WrongDataTypeError.
        self.model.recognize(np.zeros(SAMPLE_RATE, dtype=np.float32), sample_rate=SAMPLE_RATE)
        print("[Parakeet] ready")

    def transcribe(self, pcm_bytes: bytes) -> str:
        if len(pcm_bytes) / 2 / SAMPLE_RATE < MIN_AUDIO_DURATION:
            return ""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        text = self.model.recognize(audio, sample_rate=SAMPLE_RATE).strip()
        if not text:
            return ""
        low = text.lower().strip(".,!?")
        if low in HALLUCINATION_PHRASES or _is_repetitive(text):
            return ""
        return text
