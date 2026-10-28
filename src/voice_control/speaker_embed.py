"""WeSpeaker voiceprint embedding + combined direction/voice accept rule (Task 18).

fbank() matches WeSpeaker's training frontend (Kaldi fbank on int16-scale
samples, 80 mel, 25 ms / 10 ms, no dither, per-utterance mean normalisation).
SpeakerEmbedder runs voxceleb_resnet34_LM.onnx (see scripts/setup_wespeaker.py).

Voiceprints are kept in memory only; nothing here writes to disk.
"""
import numpy as np


def fbank(pcm16: bytes, sr=16000) -> np.ndarray:
    """(T, 80) float32 Kaldi fbank of 16-bit mono PCM, mean-normalised per utterance."""
    import torch
    import torchaudio.compliance.kaldi as kaldi

    # WeSpeaker feeds int16-range floats (not scaled to [-1, 1]).
    wav = torch.from_numpy(np.frombuffer(pcm16, dtype=np.int16).astype(np.float32)).unsqueeze(0)
    feats = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        energy_floor=0.0,
        sample_frequency=sr,
    )
    feats = feats - feats.mean(dim=0, keepdim=True)
    return feats.numpy().astype(np.float32)


class SpeakerEmbedder:
    """256-d L2-normalised speaker embedding via onnxruntime (CUDA, CPU fallback)."""

    def __init__(self, model_path):
        import onnxruntime as ort

        # Skip TensorRT (first in the list on Jetson) to avoid engine builds.
        wanted = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        providers = [p for p in wanted if p in ort.get_available_providers()]
        self._sess = ort.InferenceSession(str(model_path), providers=providers)
        self._input = self._sess.get_inputs()[0].name

    def embed(self, pcm16: bytes) -> np.ndarray:
        feats = fbank(pcm16)[np.newaxis, :, :]
        emb = np.asarray(self._sess.run(None, {self._input: feats})[0], dtype=np.float32).reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-12)


def cosine(a, b) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def accept_combined(dir_ok: bool | None, sim: float | None, threshold: float) -> bool:
    """Combine direction-lock and voiceprint verdicts; None = signal unavailable."""
    if dir_ok is None and sim is None:
        return True
    if sim is None:
        return bool(dir_ok)
    voice_ok = sim >= threshold
    if dir_ok is None:
        return voice_ok
    return bool(dir_ok) and voice_ok
