"""Wake word detector using sherpa-onnx KeywordSpotter.

Detects "Hey Spot" in streaming 16kHz PCM audio with ~0ms latency.
Uses sherpa-onnx's zipformer-based keyword spotter (int8, CPU).

No new dependencies — sherpa-onnx is already installed for TTS.

Usage:
    # In code:
    from wake_word import WakeWordDetector
    det = WakeWordDetector()
    if det.process_frame(pcm16_bytes):
        print("Hey Spot detected!")

    # Standalone mic test:
    python src/voice_control/wake_word.py --test-mic --device 24
"""

import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MODEL_DIR = _PROJECT_ROOT / "models" / "kws"

# Model file names (int8 quantized for speed + low memory)
_ENCODER = "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_DECODER = "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_JOINER = "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"
_TOKENS = "tokens.txt"
_KEYWORDS = "keywords.txt"

# Detection tuning
KEYWORDS_SCORE = 1.5       # Boost keyword score (higher = easier to trigger)
KEYWORDS_THRESHOLD = 0.25  # Detection threshold (higher = harder to trigger)
NUM_TRAILING_BLANKS = 1    # Blanks after keyword before firing


class WakeWordDetector:
    """Streaming wake word detector for 'Hey Spot'.

    Wraps sherpa-onnx KeywordSpotter. Feed 16kHz PCM16 frames
    via process_frame() — returns True when keyword detected.
    """

    def __init__(self, model_dir=None):
        self._spotter = None
        self._stream = None
        self._available = False
        self._sample_rate = 16000

        model_dir = pathlib.Path(model_dir) if model_dir else _MODEL_DIR
        self._load(model_dir)

    def _load(self, model_dir):
        """Load sherpa-onnx keyword spotter."""
        try:
            import sherpa_onnx

            encoder = model_dir / _ENCODER
            decoder = model_dir / _DECODER
            joiner = model_dir / _JOINER
            tokens = model_dir / _TOKENS
            keywords = model_dir / _KEYWORDS

            # Check files exist
            for f in [encoder, decoder, joiner, tokens, keywords]:
                if not f.exists():
                    print(f"[WakeWord] Missing: {f}")
                    print(f"[WakeWord] Run: python scripts/setup_kws.py")
                    return

            self._spotter = sherpa_onnx.KeywordSpotter(
                encoder=str(encoder),
                decoder=str(decoder),
                joiner=str(joiner),
                tokens=str(tokens),
                keywords_file=str(keywords),
                num_threads=1,
                provider="cpu",
                keywords_score=KEYWORDS_SCORE,
                keywords_threshold=KEYWORDS_THRESHOLD,
                num_trailing_blanks=NUM_TRAILING_BLANKS,
            )
            self._stream = self._spotter.create_stream()
            self._available = True

            # Read keyword for display
            kw_text = keywords.read_text().strip()
            print(f"[WakeWord] Ready — keyword: {kw_text}")
            print(f"[WakeWord]   score={KEYWORDS_SCORE}, threshold={KEYWORDS_THRESHOLD}")

        except ImportError:
            print("[WakeWord] sherpa-onnx not installed")
        except Exception as e:
            print(f"[WakeWord] Load failed: {e}")

    def is_available(self) -> bool:
        return self._available

    def process_frame(self, pcm16_bytes: bytes) -> bool:
        """Feed a PCM16 audio frame and check for keyword.

        Args:
            pcm16_bytes: Raw 16-bit signed PCM at 16kHz (any frame size).

        Returns:
            True if "Hey Spot" was just detected.
        """
        if not self._available:
            return False

        try:
            import numpy as np
            samples = np.frombuffer(pcm16_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            self._stream.accept_waveform(self._sample_rate, samples)

            while self._spotter.is_ready(self._stream):
                self._spotter.decode_stream(self._stream)

            result = self._spotter.get_result(self._stream)
            if result.strip():
                print(f"[WakeWord] Detected: {result.strip()}")
                # Reset stream for next detection
                self._stream = self._spotter.create_stream()
                return True

            return False

        except Exception as e:
            print(f"[WakeWord] Error: {e}")
            return False

    def reset(self):
        """Reset the detector stream (call after handling a detection)."""
        if self._available:
            self._stream = self._spotter.create_stream()


# ---------------------------------------------------------------------------
# Standalone mic test
# ---------------------------------------------------------------------------
def _test_mic(device=None):
    """Live microphone test — say 'Hey Spot' and see if it triggers."""
    import numpy as np
    import sounddevice as sd
    import time

    SAMPLE_RATE = 16000
    FRAME_MS = 30
    FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480

    detector = WakeWordDetector()
    if not detector.is_available():
        print("Wake word detector not available.")
        return

    print(f"\n{'='*50}")
    print(f"WAKE WORD MIC TEST")
    print(f"  Device: {device or 'default'}")
    print(f"  Say 'Hey Spot' to test detection")
    print(f"  Ctrl+C to exit")
    print(f"{'='*50}\n")

    detections = 0

    def callback(indata, frames, time_info, status):
        nonlocal detections
        if status:
            pass  # ignore overflow
        if indata.shape[1] >= 2:
            mono = indata[:, 0].astype(np.float32)
        else:
            mono = indata[:, 0].astype(np.float32)
        pcm16 = (mono * 32767).astype(np.int16).tobytes()
        if detector.process_frame(pcm16):
            detections += 1
            print(f"  >>> DETECTED 'Hey Spot'! (#{detections}) <<<")

    try:
        try:
            stream = sd.InputStream(device=device, channels=2, samplerate=SAMPLE_RATE,
                                    callback=callback, blocksize=FRAME_SAMPLES)
        except Exception:
            stream = sd.InputStream(device=device, channels=1, samplerate=SAMPLE_RATE,
                                    callback=callback, blocksize=FRAME_SAMPLES)

        with stream:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        print(f"\n\nTotal detections: {detections}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Wake word detector test")
    parser.add_argument("--test-mic", action="store_true", help="Live mic test")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    args = parser.parse_args()

    if args.test_mic:
        _test_mic(device=args.device)
    else:
        # Quick load test
        det = WakeWordDetector()
        print(f"Available: {det.is_available()}")
        if det.is_available():
            import numpy as np
            silence = np.zeros(480, dtype=np.int16).tobytes()
            for _ in range(100):
                assert not det.process_frame(silence), "False positive on silence!"
            print("Silence test passed (100 frames, no false positives)")
