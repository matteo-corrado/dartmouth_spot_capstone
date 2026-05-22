"""Parakeet TDT 0.6B v3 backend via onnx-asr (CUDA EP). Rollback for Nemotron."""
import os
import re
import unicodedata
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

# Phonetic-equivalent rewrites for v3 multilingual-head leaks. onnx-asr <=0.11
# exposes `language=` only on Canary AED (see nemo.py:232); Parakeet TDT
# inherits transducer decoding with no language token, so cross-language
# misfires on short utterances are unfixable at the model layer.
# Grow empirically — only add entries verified against the lab corpus.
PHONETIC_REWRITES = {
    "стоп": "stop",  # ru "stop" — observed in T12 on stop_lab.wav
    "стой": "stop",  # ru "halt"
}
_NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")


def _is_repetitive(text: str, threshold: float = 0.7) -> bool:
    words = text.strip().lower().split()
    if len(words) < 4:
        return False
    return Counter(words).most_common(1)[0][1] / len(words) > threshold


def _english_post(text: str) -> str:
    """Rewrite known phonetic equivalents; drop unrecoverable non-Latin."""
    key = "".join(c for c in text.lower() if not unicodedata.category(c).startswith("P")).strip()
    if key in PHONETIC_REWRITES:
        return PHONETIC_REWRITES[key]
    if _NON_LATIN_RE.search(text):
        print(f"[Parakeet] dropping non-Latin transcript: {text!r}")
        return ""
    return text


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
        text = _english_post(text)
        if not text:
            return ""
        low = text.lower().strip(".,!?")
        if low in HALLUCINATION_PHRASES or _is_repetitive(text):
            return ""
        return text
