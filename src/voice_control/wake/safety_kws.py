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
KEYWORDS_SCORE = 1.5  # 2026-10-29 fine sweep (ch1): gain 3, thr 0.2, score 1.5 = 14/14 incl. all quiet, 2 false in 6.4 min
# 0.20 -> 0.25 with the staggered streams below: young streams over-fire at
# 0.20 (5 vs 2 fires on fan+fan_babble); 0.25 holds fires at the old level.
# (Single-stream, 0.10-0.20 gave identical recall: threshold wasn't binding.)
KEYWORDS_THRESHOLD = 0.25  # fallback; live value = Tuning.safety_kws_threshold (config/tuning.yaml)
NUM_TRAILING_BLANKS = 1
INPUT_GAIN = 3.0  # see KEYWORDS_SCORE note (x4 score 2.0 was 12/14, 0 false; live quiet misses were common)

# Staggered streams (kwsfix 2026-10-29). A sherpa stream that has already
# heard ~0.5 s+ of audio misses quiet/far stop/freeze that a fresh stream
# catches (fan prefix 0.78 s: 5/14 vs 14/14; reset_stream() does not help,
# only create_stream() does - the loss is in the stream's feature/encoder
# context, not the beam). A single long-lived stream was 6-8/14 on continuous
# audio. So NUM_STREAMS streams run in parallel, stream i reborn at frames
# i*STAGGER_FRAMES + m*NUM_STREAMS*STAGGER_FRAMES (lifetime 2.4 s), so one
# stream is always young. A hit returns the first label. Measured on 6
# random-pad continuous runs (14 clips each): mean 7.2-7.7 -> 9.0-10.7 / 14,
# fires 1-2 -> 2 in 6.4 min of fan+fan_babble. ~4 ms/frame mean (3 streams).
# Does NOT reach the fresh-stream 14/14: see kwsfix-report.md.
NUM_STREAMS = 3
STAGGER_FRAMES = 27  # 30 ms frames (0.81 s); 28 measured worse (9.0 vs 10.7 mean, 3 vs 2 fires)


THRESHOLD_MIN, THRESHOLD_MAX = 0.05, 0.40


def _threshold() -> float:
    """Operator knob Tuning.safety_kws_threshold, clamped to [THRESHOLD_MIN,
    THRESHOLD_MAX]; KEYWORDS_THRESHOLD if tuning is unavailable."""
    try:
        from src.voice_control.tuning import get_tuning
        t = float(get_tuning().safety_kws_threshold)
    except Exception as e:  # noqa: BLE001 - a bad tuning file must not disable safety
        print(f"[SafetyKWS] tuning unavailable ({e}); threshold {KEYWORDS_THRESHOLD}")
        return KEYWORDS_THRESHOLD
    if t != t:  # NaN
        t = KEYWORDS_THRESHOLD
    c = min(max(t, THRESHOLD_MIN), THRESHOLD_MAX)
    if c != t:
        print(f"[SafetyKWS] !!! safety_kws_threshold {t} out of range "
              f"[{THRESHOLD_MIN}, {THRESHOLD_MAX}]; using {c}")
    return c


class SafetyKeywordDetector:
    def __init__(self, model_dir=None):
        self._spotter = None
        self._streams = [None] * NUM_STREAMS
        self._frame = 0
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
            thr = _threshold()
            self._spotter = sherpa_onnx.KeywordSpotter(
                encoder=str(files["encoder"]), decoder=str(files["decoder"]),
                joiner=str(files["joiner"]), tokens=str(files["tokens"]),
                keywords_file=str(files["keywords"]),
                num_threads=1, provider="cpu",
                keywords_score=KEYWORDS_SCORE,
                keywords_threshold=thr,
                num_trailing_blanks=NUM_TRAILING_BLANKS,
            )
            self._available = True
            print(f"[SafetyKWS] ready — stop/freeze/estop always-on "
                  f"(threshold {thr}, {NUM_STREAMS} streams)")
        except ImportError:
            print("[SafetyKWS] sherpa-onnx not installed")
        except Exception as e:
            print(f"[SafetyKWS] load failed: {e}")

    def is_available(self) -> bool:
        return self._available

    def process_frame(self, pcm16_bytes: bytes) -> str:
        """Return matched keyword id ('stop'|'freeze'|'estop') or ''.
        Detector errors propagate: safety_kws_loop counts them, reports
        unhealthy after a run of failures, and recovers via reset()."""
        if not self._available:
            return ""
        samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        samples = np.clip(samples * INPUT_GAIN, -1.0, 1.0)
        self._rebirth()
        self._frame += 1
        for st in self._streams:
            if st is None:
                continue
            st.accept_waveform(self._sample_rate, samples)
            while self._spotter.is_ready(st):
                self._spotter.decode_stream(st)
            hit = self._spotter.get_result(st).strip()
            if hit:
                # Return before touching the other streams: an error there
                # must not lose this hit. Fresh streams for the next word.
                self.reset()
                return hit.lower()
        return ""

    def _rebirth(self):
        """Create stream i at frames i*STAGGER + m*NUM_STREAMS*STAGGER."""
        for i in range(NUM_STREAMS):
            age = self._frame - i * STAGGER_FRAMES
            if age >= 0 and age % (NUM_STREAMS * STAGGER_FRAMES) == 0:
                self._streams[i] = self._spotter.create_stream()

    def reset(self):
        """Drop all streams and restart the stagger schedule (stream 0 is
        born on the next frame)."""
        if self._available:
            self._streams = [None] * NUM_STREAMS
            self._frame = 0


def make_safety_detector():
    return SafetyKeywordDetector()
