# src/voice_control/wake/safety_kws.py
"""Always-on frame-level safety keyword spotter (Stage 2F C1).

Spots stop/freeze/estop in WAKE_WORD state with no wake required, regardless
of SPOT_WAKE_BACKEND. Separate, always-sherpa instance so it works even when
the wake backend is livekit. process_frame() returns the matched keyword id
('stop'|'freeze'|'estop') or '' — the caller maps it via check_safety_command.
"""
import pathlib

import numpy as np

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_MODEL_DIR = _PROJECT_ROOT / "models" / "kws"

_ENCODER = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_DECODER = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_JOINER = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_TOKENS = "tokens.txt"
_SAFETY_KEYWORDS = "safety_keywords.txt"

# Safety must over-recall: easier to trigger than the wake word. These are the
# per-detector overrides; tuned by eval_safety_recall.py (P2).
KEYWORDS_SCORE = 2.0
KEYWORDS_THRESHOLD = 0.20
NUM_TRAILING_BLANKS = 1
INPUT_GAIN = 8.0  # match the wake detector's post-AGC gain


class SafetyKeywordDetector:
    def __init__(self, model_dir=None):
        self._spotter = None
        self._stream = None
        self._available = False
        self._sample_rate = 16000
        self._load(pathlib.Path(model_dir) if model_dir else _MODEL_DIR)

    def _load(self, model_dir):
        try:
            import sherpa_onnx
            files = {k: model_dir / v for k, v in (
                ("encoder", _ENCODER), ("decoder", _DECODER), ("joiner", _JOINER),
                ("tokens", _TOKENS), ("keywords", _SAFETY_KEYWORDS))}
            for f in files.values():
                if not f.exists():
                    print(f"[SafetyKWS] missing {f} — run scripts/setup_safety_kws.py")
                    return
            self._spotter = sherpa_onnx.KeywordSpotter(
                encoder=str(files["encoder"]), decoder=str(files["decoder"]),
                joiner=str(files["joiner"]), tokens=str(files["tokens"]),
                keywords_file=str(files["keywords"]),
                num_threads=1, provider="cpu",
                keywords_score=KEYWORDS_SCORE,
                keywords_threshold=KEYWORDS_THRESHOLD,
                num_trailing_blanks=NUM_TRAILING_BLANKS,
            )
            self._stream = self._spotter.create_stream()
            self._available = True
            print("[SafetyKWS] ready — stop/freeze/estop always-on")
        except ImportError:
            print("[SafetyKWS] sherpa-onnx not installed")
        except Exception as e:
            print(f"[SafetyKWS] load failed: {e}")

    def is_available(self) -> bool:
        return self._available

    def process_frame(self, pcm16_bytes: bytes) -> str:
        """Return matched keyword id ('stop'|'freeze'|'estop') or ''."""
        if not self._available:
            return ""
        try:
            samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            samples = np.clip(samples * INPUT_GAIN, -1.0, 1.0)
            self._stream.accept_waveform(self._sample_rate, samples)
            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)
            result = self._spotter.get_result(self._stream).strip()
            if result:
                # Reset stream so the next safety word is independent.
                self._stream = self._spotter.create_stream()
                return result.lower()
            return ""
        except Exception as e:
            print(f"[SafetyKWS] error: {e}")
            return ""

    def reset(self):
        if self._available:
            self._stream = self._spotter.create_stream()


def make_safety_detector():
    return SafetyKeywordDetector()
