"""Nemotron Speech Streaming 0.6B backend via sherpa-onnx (CUDA EP).

Exposes both:
- transcribe(pcm_bytes) -> str          (offline, for VAD-endpointed turns)
- stream() -> NemotronStream context     (online, words emerge during speech)

For 2A T10 (wake-gated pipeline), the dispatch uses transcribe() once the VAD
segment ends. Streaming is reserved for a Stage 2B/C upgrade where partial
transcripts feed the brain mid-utterance.
"""
from __future__ import annotations
import os
from collections import Counter
from pathlib import Path

import numpy as np
import sherpa_onnx

SAMPLE_RATE = 16000
MIN_AUDIO_DURATION = 0.15

NEMO_DIR = Path(
    os.environ.get(
        "NEMO_DIR",
        "/mnt/ssd/nemotron-models/sherpa-onnx-nemotron-speech-streaming-en-0.6b-560ms-int8-2026-04-25",
    )
)

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


class NemotronBackend:
    name = "nemotron"

    def __init__(self) -> None:
        encoder = str(NEMO_DIR / "encoder.int8.onnx")
        decoder = str(NEMO_DIR / "decoder.int8.onnx")
        joiner = str(NEMO_DIR / "joiner.int8.onnx")
        tokens = str(NEMO_DIR / "tokens.txt")
        for p in (encoder, decoder, joiner, tokens):
            if not Path(p).exists():
                raise FileNotFoundError(f"Nemotron asset missing: {p}")
        print(f"[Nemotron] loading from {NEMO_DIR}")
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            tokens=tokens,
            sample_rate=SAMPLE_RATE,
            feature_dim=80,
            provider="cuda",
            num_threads=2,
            decoding_method="greedy_search",
        )
        print("[Nemotron] warming up (1 s silence)")
        warmup = np.zeros(SAMPLE_RATE, dtype=np.float32)
        s = self.recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, warmup)
        while self.recognizer.is_ready(s):
            self.recognizer.decode_stream(s)
        print("[Nemotron] ready")

    def transcribe(self, pcm_bytes: bytes) -> str:
        if len(pcm_bytes) / 2 / SAMPLE_RATE < MIN_AUDIO_DURATION:
            return ""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        s = self.recognizer.create_stream()
        s.accept_waveform(SAMPLE_RATE, audio)
        s.input_finished()
        while self.recognizer.is_ready(s):
            self.recognizer.decode_stream(s)
        text = self.recognizer.get_result(s).strip()
        if not text:
            return ""
        low = text.lower().strip(".,!?")
        if low in HALLUCINATION_PHRASES or _is_repetitive(text):
            return ""
        return text
