"""Text-to-Speech for Spot using Kokoro TTS (ONNX).

Kokoro is #1 on TTS Arena with near-human voice quality. Runs on CPU
via ONNX runtime (no GPU competition with LLM/ASR). 82M params, ~88MB int8.

Install:
    pip install kokoro-onnx sounddevice
    python scripts/setup_kokoro.py   # downloads model files (~88MB)

On Jetson with CUDA acceleration (optional, CPU is fast enough):
    Replace onnxruntime with NVIDIA's Jetson wheel for CUDAExecutionProvider.
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
DEFAULT_VOICE = "af_heart"  # American female, warm/natural tone
MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "tts"
MODEL_FILE = "kokoro-v1.0.onnx"
VOICES_FILE = "voices-v1.0.bin"


class SpotTTS:
    """Non-blocking text-to-speech via Kokoro TTS (ONNX).

    Usage:
        tts = SpotTTS()
        tts.speak("Hello!")        # non-blocking (background thread)
        tts.speak_sync("Hello!")   # blocking (waits for playback)
    """

    def __init__(self, voice: str = DEFAULT_VOICE, output_device=None,
                 on_mute=None, on_unmute=None):
        """Initialize Kokoro TTS.

        Args:
            voice: Kokoro voice name (e.g. "af_heart", "am_adam", "bf_emma").
            output_device: sounddevice output device index (None = system default).
            on_mute: Callback invoked before playback starts (use to mute mic).
            on_unmute: Callback invoked after playback ends (use to unmute mic).
        """
        self.voice_name = voice
        self.output_device = output_device
        self._on_mute = on_mute
        self._on_unmute = on_unmute
        self._kokoro = None
        self._thread = None
        self._playing = False
        self._lock = threading.Lock()
        self._available = None

        self._load_model()

    def _load_model(self):
        """Load Kokoro ONNX model and voice pack."""
        try:
            from kokoro_onnx import Kokoro

            # Search for model files
            search_dirs = [
                MODEL_DIR,                                      # project: models/tts/
                Path("~/.local/share/kokoro_models").expanduser(),  # user-local
                Path.cwd(),                                     # current directory
            ]

            model_path = None
            voices_path = None

            for d in search_dirs:
                m = d / MODEL_FILE
                v = d / VOICES_FILE
                if m.exists() and v.exists():
                    model_path = m
                    voices_path = v
                    break

            if not model_path or not voices_path:
                print(f"[TTS] Kokoro model files not found.")
                print(f"[TTS] Run: python scripts/setup_kokoro.py")
                print(f"[TTS] Searched: {[str(d) for d in search_dirs]}")
                self._available = False
                return

            print(f"[TTS] Loading Kokoro model from: {model_path.parent}")

            # Try GPU (Jetson CUDA EP) first, fall back to CPU
            try:
                import onnxruntime as ort
                providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in providers:
                    session = ort.InferenceSession(
                        str(model_path),
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    )
                    self._kokoro = Kokoro.from_session(session, str(voices_path))
                    print("[TTS] Kokoro loaded (CUDA)")
                else:
                    self._kokoro = Kokoro(str(model_path), str(voices_path))
                    print("[TTS] Kokoro loaded (CPU)")
            except Exception:
                self._kokoro = Kokoro(str(model_path), str(voices_path))
                print("[TTS] Kokoro loaded (CPU)")

            # Verify voice exists
            available_voices = self._kokoro.get_voices()
            if self.voice_name not in available_voices:
                print(f"[TTS] Voice '{self.voice_name}' not found. Available: {available_voices[:5]}...")
                print(f"[TTS] Falling back to '{DEFAULT_VOICE}'")
                self.voice_name = DEFAULT_VOICE

            self._available = True
            print(f"[TTS] Voice: {self.voice_name} ({len(available_voices)} voices available)")

        except ImportError:
            print("[TTS] kokoro-onnx not installed. Install with: pip install kokoro-onnx")
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

            # Synthesize with Kokoro (returns float32 samples + sample rate)
            samples, sample_rate = self._kokoro.create(
                text,
                voice=self.voice_name,
                speed=1.0,
                lang="en-us",
            )

            # Play audio
            sd.play(samples, samplerate=sample_rate, device=self.output_device)
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


def get_tts(voice: str = DEFAULT_VOICE, output_device=None) -> SpotTTS:
    """Get or create the singleton SpotTTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = SpotTTS(voice=voice, output_device=output_device)
    return _tts_instance


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    voice = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_VOICE
    text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "Hello! I am Spot, a Boston Dynamics robot at Dartmouth College."

    print(f"Voice: {voice}")
    print(f"Text: {text}")

    tts = SpotTTS(voice=voice)
    if tts.is_available():
        # List a few available voices
        print(f"Available voices: {tts._kokoro.get_voices()[:10]}...")
        tts.speak_sync(text)
        print("Done!")
    else:
        print("TTS not available.")
