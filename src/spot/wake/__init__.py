"""Wake-word backends. Selected via SPOT_WAKE_BACKEND env var.

Default = sherpa_onnx (production now). livekit becomes default once the
custom-trained hey_spot.onnx is in place (see scripts/training/).

Backends expose: is_available(), process_frame(pcm16_bytes) -> bool, reset().
"""
import os

_DEFAULT = "sherpa_onnx"


def make_wake_detector():
    name = os.environ.get("SPOT_WAKE_BACKEND", _DEFAULT).lower()
    if name == "sherpa_onnx":
        from .sherpa_onnx import WakeWordDetector
        return WakeWordDetector()
    if name == "livekit":
        from .livekit import LiveKitWakeWord
        return LiveKitWakeWord()
    raise ValueError(f"Unknown SPOT_WAKE_BACKEND={name!r}; valid: sherpa_onnx|livekit")
