"""Text-to-Speech for Spot using sherpa-onnx Kokoro TTS.

Kokoro is a high-quality neural TTS model. sherpa-onnx bundles its own ONNX
runtime with aarch64 wheels that work on Jetson (no pip onnxruntime crash).
Runs on CPU — no GPU competition with LLM/ASR.

Install:
    pip install sherpa-onnx
    python scripts/setup_kokoro.py   # downloads model pack (~340MB)
"""

import threading
from pathlib import Path
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Speaker IDs for kokoro-en-v0_19 (English, 11 speakers):
#   0=af  1=af_bella  2=af_nicole  3=af_sarah  4=af_sky
#   5=am_adam  6=am_michael  7=bf_emma  8=bf_isabella  9=bm_george  10=bm_lewis
DEFAULT_SPEAKER_ID = 3       # af_sarah — warm/natural American female
DEFAULT_SPEED = 1.1
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "tts" / "kokoro-en-v0_19"


class SpotTTS:
    """Non-blocking text-to-speech via sherpa-onnx Kokoro TTS.

    Usage:
        tts = SpotTTS()
        tts.speak("Hello!")        # non-blocking (background thread)
        tts.speak_sync("Hello!")   # blocking (waits for playback)
    """

    def __init__(self, speaker_id: int = DEFAULT_SPEAKER_ID, speed: float = DEFAULT_SPEED,
                 output_device=None, on_mute=None, on_unmute=None):
        """Initialize sherpa-onnx Kokoro TTS.

        Args:
            speaker_id: Kokoro speaker ID (see table above).
            speed: Speech speed (1.0 = normal, >1 = faster).
            output_device: sounddevice output device index (None = system default).
            on_mute: Callback invoked before playback starts (use to mute mic).
            on_unmute: Callback invoked after playback ends (use to unmute mic).
        """
        self.speaker_id = speaker_id
        self.speed = speed
        self.output_device = output_device
        self._on_mute = on_mute
        self._on_unmute = on_unmute
        self._tts = None
        self._thread = None
        self._playing = False
        self._lock = threading.Lock()
        self._available = None

        self._load_model()

    def _load_model(self):
        """Load sherpa-onnx Kokoro TTS model."""
        if sd is None:
            print("[TTS] sounddevice not installed — no audio playback")
            print("[TTS] Install with: pip install sounddevice")
            self._available = False
            return

        try:
            import sherpa_onnx

            model_path = MODEL_DIR / "model.onnx"
            voices_path = MODEL_DIR / "voices.bin"
            tokens_path = MODEL_DIR / "tokens.txt"
            data_dir = MODEL_DIR / "espeak-ng-data"

            if not model_path.exists():
                print(f"[TTS] Model not found: {model_path}")
                print(f"[TTS] Run: python scripts/setup_kokoro.py")
                self._available = False
                return

            print(f"[TTS] Loading sherpa-onnx Kokoro from: {MODEL_DIR}")

            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                        model=str(model_path),
                        voices=str(voices_path),
                        tokens=str(tokens_path),
                        data_dir=str(data_dir),
                    ),
                    provider="cpu",
                    num_threads=2,
                ),
                max_num_sentences=1,
            )

            if not tts_config.validate():
                print("[TTS] Config validation failed — check model files")
                self._available = False
                return

            self._tts = sherpa_onnx.OfflineTts(tts_config)
            self._available = True
            print(f"[TTS] Ready — speaker_id={self.speaker_id}, speed={self.speed}, "
                  f"sample_rate={self._tts.sample_rate}Hz")

        except ImportError:
            print("[TTS] sherpa-onnx not installed. Install with: pip install sherpa-onnx")
            self._available = False
        except Exception as e:
            print(f"[TTS] Failed to load model: {e}")
            self._available = False

    def is_available(self) -> bool:
        """Check if TTS is ready."""
        return self._available is True

    def is_speaking(self) -> bool:
        """Check if audio is currently playing."""
        return self._playing

    def speak(self, text: str):
        """Speak text non-blocking (returns immediately, plays in background)."""
        if not self.is_available():
            print(f'[TTS] Not available — would say: "{text}"')
            return

        if not text or not text.strip():
            return

        # Cancel any in-progress speech
        with self._lock:
            if self._playing:
                try:
                    sd.stop()
                except Exception:
                    pass
                self._playing = False

        self._thread = threading.Thread(
            target=self._speak_impl, args=(text,), daemon=True
        )
        self._thread.start()

    def speak_sync(self, text: str):
        """Speak text and block until playback finishes."""
        self._speak_impl(text)

    def _speak_impl(self, text: str):
        """Generate and play TTS audio."""
        if not self.is_available() or not text:
            return

        try:
            self._playing = True
            if self._on_mute:
                self._on_mute()

            audio = self._tts.generate(text, sid=self.speaker_id, speed=self.speed)

            if audio.samples is not None and len(audio.samples) > 0:
                samples = np.array(audio.samples, dtype=np.float32)
                rate = audio.sample_rate  # 24000 for Kokoro

                # Resample if output device doesn't support native rate
                if self.output_device is not None:
                    try:
                        dev_info = sd.query_devices(self.output_device)
                        dev_rate = int(dev_info['default_samplerate'])
                        if dev_rate != rate:
                            # Simple linear interpolation resample
                            ratio = dev_rate / rate
                            n_out = int(len(samples) * ratio)
                            indices = np.arange(n_out) / ratio
                            samples = np.interp(indices, np.arange(len(samples)), samples)
                            rate = dev_rate
                    except Exception:
                        pass

                sd.play(samples, samplerate=rate, device=self.output_device)
                sd.wait()

        except Exception as e:
            print(f"[TTS] Playback error: {e}")
        finally:
            self._playing = False
            if self._on_unmute:
                self._on_unmute()

    def wait(self):
        """Wait for current speech to finish."""
        if self._thread and self._thread.is_alive():
            self._thread.join()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_tts_instance = None


def get_tts(speaker_id: int = DEFAULT_SPEAKER_ID, output_device=None) -> SpotTTS:
    """Get or create the singleton SpotTTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = SpotTTS(speaker_id=speaker_id, output_device=output_device)
    return _tts_instance


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import argparse as _ap

    p = _ap.ArgumentParser(description="Spot TTS test")
    p.add_argument("--sid", type=int, default=DEFAULT_SPEAKER_ID, help="Speaker ID (0-10)")
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Speech speed")
    p.add_argument("--wav", type=str, default=None, help="Save to WAV file instead of playing")
    p.add_argument("text", nargs="*", default=["Hello! I am Spot, a Boston Dynamics robot at Dartmouth College."])
    a = p.parse_args()
    text = " ".join(a.text)

    print(f"Speaker ID: {a.sid}, Speed: {a.speed}")
    print(f"Text: {text}")

    if a.wav:
        # Direct WAV export (no sounddevice needed)
        import sherpa_onnx, soundfile as _sf
        tts_config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                kokoro=sherpa_onnx.OfflineTtsKokoroModelConfig(
                    model=str(MODEL_DIR / "model.onnx"),
                    voices=str(MODEL_DIR / "voices.bin"),
                    tokens=str(MODEL_DIR / "tokens.txt"),
                    data_dir=str(MODEL_DIR / "espeak-ng-data"),
                ), num_threads=2,
            ), max_num_sentences=1,
        )
        tts_engine = sherpa_onnx.OfflineTts(tts_config)
        audio = tts_engine.generate(text, sid=a.sid, speed=a.speed)
        _sf.write(a.wav, audio.samples, samplerate=audio.sample_rate, subtype="PCM_16")
        dur = len(audio.samples) / audio.sample_rate
        print(f"Saved {dur:.1f}s to {a.wav}")
    else:
        tts = SpotTTS(speaker_id=a.sid, speed=a.speed)
        if tts.is_available():
            tts.speak_sync(text)
            print("Done!")
        else:
            print("TTS not available.")
