"""Text-to-Speech for Spot using kokoro-onnx + onnxruntime-gpu (Jetson CUDA).

Kokoro is a high-quality neural TTS model. We run it directly via the
kokoro-onnx Python wrapper on top of onnxruntime-gpu, which targets the
Jetson AGX Orin's CUDA Execution Provider for ~7-10x realtime synthesis.

Why not the kokoro-onnx default provider selection?
    kokoro-onnx 0.4.9 has a bug at __init__.py:41 — it checks
    `importlib.util.find_spec("onnxruntime-gpu")` to detect the GPU build,
    but no Python module by that name exists (the importable module is
    just `onnxruntime` for both CPU and GPU packages). So the check always
    returns None and kokoro-onnx silently falls back to CPU. The escape
    hatch at __init__.py:46 is `ONNX_PROVIDER` env var override, which we
    set below before importing the package.

Install (Jetson AGX Orin, JetPack 6, CUDA 12.6, Python 3.10, aarch64):
    pip install --no-deps "numpy==1.26.4"
    pip install --no-deps "opencv-python==4.11.0.86"
    pip install --no-deps "kokoro-onnx==0.4.9"
    pip install joblib
    pip install --no-deps --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
                "onnxruntime-gpu==1.23.0"
    python scripts/setup_kokoro_v1.py    # downloads kokoro-v1.0.fp16-gpu.onnx + voices-v1.0.bin
"""

import os
import threading
from pathlib import Path
import numpy as np

# CRITICAL: must be set BEFORE `import kokoro_onnx`. Routes the InferenceSession
# through CUDA. See module docstring for the bug rationale.
os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")

try:
    import sounddevice as sd
except ImportError:
    sd = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Kokoro v1.0 ships 54 voices (af_*, am_*, bf_*, bm_*, etc.). af_sarah is a
# warm/natural American female voice; full list via Kokoro.get_voices().
DEFAULT_VOICE = "af_sarah"
DEFAULT_SPEED = 1.1
DEFAULT_LANG = "en-us"
DEFAULT_VOLUME = 1.0
MAX_VOLUME = 1.5  # >1.5 risks clipping on loud phonemes

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "tts" / "kokoro-v1.0"
MODEL_FILE = MODEL_DIR / "kokoro-v1.0.fp16-gpu.onnx"
VOICES_FILE = MODEL_DIR / "voices-v1.0.bin"


class SpotTTS:
    """Non-blocking GPU-accelerated text-to-speech via kokoro-onnx + CUDA.

    Usage:
        tts = SpotTTS()
        tts.speak("Hello!")        # non-blocking (background thread)
        tts.speak_sync("Hello!")   # blocking (waits for playback)
    """

    def __init__(self, voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED,
                 output_device=None, on_mute=None, on_unmute=None,
                 volume: float = DEFAULT_VOLUME):
        """Initialize kokoro-onnx Kokoro TTS on GPU.

        Args:
            voice: Kokoro voice name (e.g. "af_sarah"). See Kokoro.get_voices().
            speed: Speech speed (1.0 = normal, >1 = faster).
            output_device: sounddevice output device index (None = system default).
            on_mute: Callback invoked before playback starts (use to mute mic).
            on_unmute: Callback invoked after playback ends (use to unmute mic).
            volume: Output gain multiplier (0.0=silent, 1.0=full, 1.5=max).
        """
        self.voice = voice
        self.speed = speed
        self.output_device = output_device
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))
        self._on_mute = on_mute
        self._on_unmute = on_unmute
        self._tts = None
        self._thread = None
        self._playing = False
        self._lock = threading.Lock()
        self._available = None

        self._load_model()

    def _load_model(self):
        """Load kokoro-onnx Kokoro TTS model on the CUDA Execution Provider."""
        if sd is None:
            print("[TTS] sounddevice not installed — no audio playback")
            print("[TTS] Install with: pip install sounddevice")
            self._available = False
            return

        if not MODEL_FILE.exists():
            print(f"[TTS] Model not found: {MODEL_FILE}")
            print(f"[TTS] Run: python scripts/setup_kokoro_v1.py")
            self._available = False
            return
        if not VOICES_FILE.exists():
            print(f"[TTS] Voices file not found: {VOICES_FILE}")
            print(f"[TTS] Run: python scripts/setup_kokoro_v1.py")
            self._available = False
            return

        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            print("[TTS] kokoro-onnx not installed.")
            print("[TTS] Install with: pip install --no-deps kokoro-onnx==0.4.9")
            self._available = False
            return

        try:
            print(f"[TTS] Loading kokoro-onnx Kokoro from: {MODEL_DIR}")
            self._tts = Kokoro(model_path=str(MODEL_FILE), voices_path=str(VOICES_FILE))
        except Exception as e:
            print(f"[TTS] Failed to load model: {e}")
            self._available = False
            return

        # Verify which provider the loaded session is actually using. If
        # ONNX_PROVIDER didn't take effect (e.g. onnxruntime-gpu missing or
        # CUDA init failed), this will silently fall back to CPU.
        try:
            active_providers = self._tts.sess.get_providers()
            active = active_providers[0] if active_providers else "unknown"
        except Exception:
            active = "unknown"

        self._available = True
        sample_rate = getattr(self._tts.sess, "_sample_rate", 24000)
        print(f"[TTS] Ready — voice={self.voice}, speed={self.speed}, "
              f"provider={active}, sample_rate=24000Hz")
        if active != "CUDAExecutionProvider":
            print(f"[TTS] WARNING: not using CUDA — got '{active}'. "
                  f"Check that onnxruntime-gpu 1.23.0 is installed and that "
                  f"the ONNX_PROVIDER env var is set before module import.")

    def is_available(self) -> bool:
        """Check if TTS is ready."""
        return self._available is True

    def set_volume(self, volume: float) -> float:
        """Set output gain. Clamped to 0.0..MAX_VOLUME. Returns the clamped value."""
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))
        return self.volume

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

            samples, rate = self._tts.create(
                text, voice=self.voice, speed=self.speed, lang=DEFAULT_LANG
            )
            samples = np.asarray(samples, dtype=np.float32)

            # Apply volume gain (snapshot self.volume so a mid-utterance
            # set_volume() doesn't tear the buffer)
            gain = self.volume
            if gain != 1.0:
                samples = samples * gain

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


def get_tts(voice: str = DEFAULT_VOICE, output_device=None,
            volume: float = DEFAULT_VOLUME) -> SpotTTS:
    """Get or create the singleton SpotTTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = SpotTTS(voice=voice, output_device=output_device,
                                volume=volume)
    return _tts_instance


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse as _ap

    p = _ap.ArgumentParser(description="Spot TTS test (kokoro-onnx + GPU)")
    p.add_argument("--voice", type=str, default=DEFAULT_VOICE,
                   help="Kokoro voice name (default: af_sarah)")
    p.add_argument("--speed", type=float, default=DEFAULT_SPEED, help="Speech speed")
    p.add_argument("--volume", type=float, default=DEFAULT_VOLUME,
                   help=f"Output gain (0.0-{MAX_VOLUME}, default: {DEFAULT_VOLUME})")
    p.add_argument("--wav", type=str, default=None,
                   help="Save to WAV file instead of playing")
    p.add_argument("text", nargs="*",
                   default=["Hello! I am Spot, a Boston Dynamics robot at Dartmouth College."])
    a = p.parse_args()
    text = " ".join(a.text)

    print(f"Voice: {a.voice}, Speed: {a.speed}")
    print(f"Text: {text}")

    if a.wav:
        # Direct WAV export (no sounddevice needed)
        import soundfile as _sf
        from kokoro_onnx import Kokoro
        engine = Kokoro(model_path=str(MODEL_FILE), voices_path=str(VOICES_FILE))
        print(f"engine providers: {engine.sess.get_providers()}")
        samples, rate = engine.create(text, voice=a.voice, speed=a.speed, lang=DEFAULT_LANG)
        _sf.write(a.wav, samples, samplerate=rate, subtype="PCM_16")
        dur = len(samples) / rate
        print(f"Saved {dur:.1f}s to {a.wav}")
    else:
        tts = SpotTTS(voice=a.voice, speed=a.speed, volume=a.volume)
        if tts.is_available():
            tts.speak_sync(text)
            print("Done!")
        else:
            print("TTS not available.")
