"""WeSpeaker voiceprint embedding + combined direction/voice accept rule (Task 18).

fbank() matches WeSpeaker's frontend (Kaldi fbank on int16-scale samples,
80 mel, 25 ms / 10 ms, hamming window, no energy, no dither at inference,
per-utterance mean normalisation).
SpeakerEmbedder runs voxceleb_resnet34_LM.onnx (see scripts/setup_wespeaker.py).

Voiceprints are kept in memory only; nothing here writes to disk.
"""
import threading
import time
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = (Path(__file__).resolve().parents[2] / "models" / "speaker"
                      / "voxceleb_resnet34_LM.onnx")


def fbank(pcm16: bytes, sr=16000) -> np.ndarray:
    """(T, 80) float32 Kaldi fbank of 16-bit mono PCM, mean-normalised per utterance.

    Raises ValueError if the audio is shorter than one 25 ms frame.
    """
    samples = np.frombuffer(pcm16, dtype=np.int16)
    if len(samples) < int(sr * 0.025):
        raise ValueError(f"audio too short for fbank: {len(samples)} samples < one 25 ms frame")

    import torch
    import torchaudio.compliance.kaldi as kaldi

    # Mirrors wespeaker/dataset/processor.py compute_fbank (inference: dither 0).
    # WeSpeaker feeds int16-range floats (not scaled to [-1, 1]).
    wav = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)
    feats = kaldi.fbank(
        wav,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=sr,
        window_type="hamming",
        use_energy=False,
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


class VoiceprintGate:
    """Per-conversation voiceprint state for the speaker lock (memory only).

    Lifecycle: set_wake(pcm) at the wake (after clear()), on_accepted(pcm)
    for the first accepted utterance -> enrolment = L2-normalised mean of the
    two embeddings; decide()/score() for later utterances; clear() on lock
    release / takeover. Nothing is ever written to disk.

    `embedder` (anything with .embed(pcm16) -> vector) may be None until the
    background warm-up assigns it, or forever if the model is missing; every
    embedding failure returns None (fail open to direction-only) and is
    logged once.
    """

    def __init__(self, embedder=None, log_fn=print):
        self.embedder = embedder
        self._log = log_fn
        self._warned = set()
        self.wake_emb = None
        self.enrol = None
        self.last_sim = None

    @property
    def enrolled(self) -> bool:
        return self.enrol is not None

    def _warn_once(self, key, msg):
        if key not in self._warned:
            self._warned.add(key)
            self._log(msg)

    def _embed(self, pcm16):
        emb = self.embedder
        if emb is None:
            self._warn_once("no-model", "[Voice] voiceprint model not loaded — direction-only")
            return None
        if not pcm16:
            return None
        try:
            v = np.asarray(emb.embed(pcm16), dtype=np.float32).reshape(-1)
        except Exception as e:
            self._warn_once("embed-error", f"[Voice] embedding failed ({type(e).__name__}: {e}) "
                                           "— direction-only for that utterance (logged once)")
            return None
        n = float(np.linalg.norm(v))
        if not np.isfinite(n) or n <= 0:
            return None
        return v / n

    def clear(self):
        """Forget the voiceprint. last_sim (a score, not a voiceprint) is kept
        for `spotctl status`."""
        self.wake_emb = None
        self.enrol = None

    def set_wake(self, pcm16):
        """Wake fired: start a fresh enrolment from the wake audio."""
        self.clear()
        self.wake_emb = self._embed(pcm16)

    def on_accepted(self, pcm16):
        """First accepted utterance completes the enrolment (no-op once enrolled)."""
        if not self.enrolled:
            self._complete(self._embed(pcm16))

    def _complete(self, emb):
        if self.enrolled:
            return
        parts = [v for v in (self.wake_emb, emb) if v is not None]
        if not parts:
            return
        m = np.mean(parts, axis=0)
        n = float(np.linalg.norm(m))
        if n > 0:
            self.enrol = (m / n).astype(np.float32)

    def score(self, pcm16):
        """(sim, emb) for an utterance: emb is its embedding (None on failure);
        sim is the cosine to the enrolment, or to the wake embedding while not
        yet enrolled (informational), or None if neither exists."""
        ref = self.enrol if self.enrolled else self.wake_emb
        if ref is None and self.embedder is None:
            return None, None
        emb = self._embed(pcm16)
        sim = None if (emb is None or ref is None) else cosine(ref, emb)
        self.last_sim = sim
        return sim, emb

    def decide(self, pcm16, dir_ok, enabled: bool, threshold: float):
        """Accept verdict for one utterance during a conversation.

        dir_ok: speaker-lock verdict, None when direction is unavailable.
        Disabled: direction-only, exactly as before (sim still computed and
        returned so the operator can watch it). Enabled: accept_combined once
        enrolled; the first utterance after the wake is direction-only.
        An accepted utterance completes the enrolment if it isn't yet.
        Returns (accepted, sim).
        """
        sim, emb = self.score(pcm16)
        was_enrolled = self.enrolled
        if enabled:
            accepted = accept_combined(dir_ok, sim if was_enrolled else None, threshold)
        else:
            accepted = dir_ok is not False
        if accepted and not was_enrolled:
            self._complete(emb)
        return accepted, sim


def warm_up_async(gate: VoiceprintGate, model_path=DEFAULT_MODEL_PATH, log_fn=print,
                  factory=SpeakerEmbedder):
    """Load the embedder on a daemon thread and hand it to `gate` when ready.

    Pays the torch/torchaudio import, ORT session build and first-run CUDA
    init off the main thread; the gate sees None (fail open) until the very
    last step assigns the warmed embedder. Missing model -> logged, stays
    None, returns None (no thread). Returns the started thread otherwise.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        log_fn(f"[Voice] voiceprint model missing ({model_path}) — direction-only "
               "(run scripts/setup_wespeaker.py)")
        return None

    def _run():
        t0 = time.monotonic()
        try:
            emb = factory(model_path)
            noise = (np.random.default_rng(0).standard_normal(24000) * 1000).astype(np.int16)
            emb.embed(noise.tobytes())
        except Exception as e:
            log_fn(f"[Voice] voiceprint model failed to load ({type(e).__name__}: {e}) "
                   "— direction-only")
            return
        gate.embedder = emb
        log_fn(f"[Voice] voiceprint model ready ({time.monotonic() - t0:.1f}s warm-up)")

    t = threading.Thread(target=_run, name="voiceprint-warmup", daemon=True)
    t.start()
    return t
