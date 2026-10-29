"""ASR backends. Selected via SPOT_ASR_BACKEND env var (default: parakeet).

Plan deviation: plan set Nemotron default with Parakeet as rollback. On Jetson
AGX Orin the sherpa-onnx Python wheel has no aarch64+GPU build (CPU only,
~700ms / 2s utterance), while Parakeet via onnx-asr uses the existing CUDA-
capable onnxruntime-gpu 1.23.0 wheel and hits ~190ms warm (matches plan's
200-300ms GPU latency target). Nemotron stays available via SPOT_ASR_BACKEND.
"""
import os

_DEFAULT = "parakeet"


def get_backend():
    name = os.environ.get("SPOT_ASR_BACKEND", _DEFAULT).lower()
    if name == "nemotron":
        from .nemotron import NemotronBackend
        return NemotronBackend()
    if name == "parakeet":
        from .parakeet import ParakeetBackend
        return ParakeetBackend()
    raise ValueError(f"Unknown SPOT_ASR_BACKEND={name!r}; valid: nemotron|parakeet")
