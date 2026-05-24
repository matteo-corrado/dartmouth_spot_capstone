"""Kokoro TTS backend via kokoro-onnx (the same package spot_tts.py uses).

KokoroBackend exposes a Protocol-compatible interface (synthesize, stream,
list_voices) on top of kokoro-onnx so the TTS dispatch layer can treat
Kokoro and ElevenLabs the same way. Voice slugs map directly to
kokoro-onnx voice names (e.g. 'af_sarah') — no integer speaker-id
resolution needed.

Model directory matches the existing spot_tts.py path:
  models/tts/kokoro-v1.0/{kokoro-v1.0.fp16-gpu.onnx, voices-v1.0.bin}
"""
import os
from pathlib import Path
from typing import Iterator

import numpy as np
from kokoro_onnx import Kokoro

from . import register_backend

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "tts" / "kokoro-v1.0"
DEFAULT_SAMPLE_RATE = 24000


class KokoroBackend:
    def __init__(self, model_dir: Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else Path(
            os.environ.get("SPOT_KOKORO_MODEL_DIR", DEFAULT_MODEL_DIR)
        )
        self._tts = self._load()

    def _load(self) -> "Kokoro":
        model_file = self.model_dir / "kokoro-v1.0.fp16-gpu.onnx"
        voices_file = self.model_dir / "voices-v1.0.bin"
        if not model_file.exists() or not voices_file.exists():
            raise FileNotFoundError(
                f"Kokoro v1.0 files missing under {self.model_dir}. "
                f"Run scripts/setup_kokoro.py first."
            )
        return Kokoro(str(model_file), str(voices_file))

    def synthesize(self, text: str, voice_id: str) -> bytes:
        """Render text → 24 kHz int16 PCM bytes using the named voice slug."""
        samples, _rate = self._tts.create(text, voice=voice_id, speed=1.0, lang="en-us")
        samples = np.asarray(samples, dtype=np.float32)
        pcm_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        return pcm_int16.tobytes()

    def stream(self, text: str, voice_id: str) -> Iterator[bytes]:
        # kokoro-onnx 0.4.9 is not natively streaming; yield single chunk.
        yield self.synthesize(text, voice_id)

    def list_voices(self) -> list:
        """Return all kokoro v1.0 voice slugs packed in voices-v1.0.bin."""
        return sorted(getattr(self._tts, "voices", {}).keys())


register_backend("kokoro", lambda: KokoroBackend())
