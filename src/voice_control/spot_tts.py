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

Architecture
------------
``SpotTTS`` is a *renderer* — it knows how to turn text into audio samples
via Kokoro and applies the volume gain. It does NOT own an output stream
or play audio directly. Playback is delegated to ``AudioPlayer``, which
serializes all audio (TTS + beeps) through one persistent OutputStream
and one worker thread. See ``audio_player.py`` for the rationale.

Install (Jetson AGX Orin, JetPack 6, CUDA 12.6, Python 3.10, aarch64):
    pip install --no-deps "numpy==1.26.4"
    pip install --no-deps "opencv-python==4.11.0.86"
    pip install --no-deps "kokoro-onnx==0.4.9"
    pip install joblib
    pip install --no-deps --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 \
                "onnxruntime-gpu==1.23.0"
    python scripts/setup_kokoro.py    # downloads kokoro-v1.0.fp16-gpu.onnx + voices-v1.0.bin
"""

import os
from pathlib import Path
import numpy as np

# CRITICAL: must be set BEFORE `import kokoro_onnx`. Routes the InferenceSession
# through CUDA. See module docstring for the bug rationale.
os.environ.setdefault("ONNX_PROVIDER", "CUDAExecutionProvider")

from src.voice_control.audio_player import AudioPlayer
from src.voice_control.latency import get_recorder

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
    """GPU-accelerated text renderer that submits playback to an AudioPlayer.

    Usage:
        player = AudioPlayer(output_device=3)
        tts = SpotTTS(player=player)
        tts.speak("Hello!")        # non-blocking — queued in player
        tts.speak_sync("Hello!")   # blocks until playback finishes
        tts.wait()                 # wait for queued speech to drain

    The previous version of this class spawned a daemon thread per ``speak``
    call and used the global ``sd.play()`` / ``sd.wait()`` API. That raced
    with ``audio_feedback.beep`` over sounddevice's module-global stream
    state and could deadlock the mic-mute interlock. Now playback is owned
    by ``AudioPlayer`` and there is exactly one worker for the whole pipeline.
    """

    def __init__(self, player: AudioPlayer | None = None,
                 voice: str = DEFAULT_VOICE, speed: float = DEFAULT_SPEED,
                 output_device=None, volume: float = DEFAULT_VOLUME):
        """Initialize the Kokoro renderer.

        Args:
            player: Shared AudioPlayer to submit playback through. If None,
                a private AudioPlayer is created using ``output_device``
                (lazy fallback for the get_tts() singleton path).
            voice: Kokoro voice name (e.g. "af_sarah"). See Kokoro.get_voices().
            speed: Speech speed (1.0 = normal, >1 = faster).
            output_device: Only used if ``player`` is None. The production
                path always passes an explicit player from client_mic.py.
            volume: Output gain multiplier (0.0=silent, 1.0=full, 1.5=max).
        """
        self.voice = voice
        self.speed = speed
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))
        self._tts = None
        self._available = None
        self._player = player if player is not None else AudioPlayer(output_device=output_device)

        self._load_model()

    def _load_model(self):
        """Load kokoro-onnx Kokoro TTS model on the CUDA Execution Provider."""
        if not self._player.is_available():
            print("[TTS] AudioPlayer unavailable — no audio playback")
            self._available = False
            return

        if not MODEL_FILE.exists():
            print(f"[TTS] Model not found: {MODEL_FILE}")
            print(f"[TTS] Run: python scripts/setup_kokoro.py")
            self._available = False
            return
        if not VOICES_FILE.exists():
            print(f"[TTS] Voices file not found: {VOICES_FILE}")
            print(f"[TTS] Run: python scripts/setup_kokoro.py")
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
        print(f"[TTS] Ready — voice={self.voice}, speed={self.speed}, "
              f"provider={active}, sample_rate=24000Hz")
        if active != "CUDAExecutionProvider":
            print(f"[TTS] WARNING: not using CUDA — got '{active}'. "
                  f"Check that onnxruntime-gpu 1.23.0 is installed and that "
                  f"the ONNX_PROVIDER env var is set before module import.")

    def is_available(self) -> bool:
        """Check if TTS is ready (model loaded AND player ready)."""
        return self._available is True

    def set_volume(self, volume: float) -> float:
        """Set output gain. Clamped to 0.0..MAX_VOLUME. Returns the clamped value."""
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))
        return self.volume

    def speak(self, text: str):
        """Enqueue ``text`` for synthesis and playback. Returns immediately.

        Kokoro inference happens inside the AudioPlayer worker thread, so
        the caller does not pay the ~1s rendering latency on its own thread.
        Multiple ``speak`` calls play in submission order — there is no
        cancel-and-replace behavior anymore.
        """
        if not self.is_available():
            print(f'[TTS] Not available — would say: "{text}"')
            return
        if not text or not text.strip():
            return

        # Snapshot the volume at submission time so a later set_volume() does
        # not retroactively change the gain of an in-flight utterance.
        gain = self.volume
        voice = self.voice
        speed = self.speed

        def _render():
            samples, rate = self._tts.create(
                text, voice=voice, speed=speed, lang=DEFAULT_LANG
            )
            samples = np.asarray(samples, dtype=np.float32)
            if gain != 1.0:
                samples = samples * gain
            return samples, rate

        # Latency hooks: stamp render/play boundaries on the current trace if
        # the latency recorder is initialized. No-op when disabled.
        rec = get_recorder()
        trace = rec.current() if rec else None
        if trace is not None:
            cb_render_start = lambda: trace.mark("tts_render_start")
            cb_render_end = lambda: trace.mark("tts_render_end")
            cb_play_start = lambda: trace.mark("tts_play_start")
            cb_play_end = lambda: trace.mark("tts_play_end")
            trace.mark("tts_enqueue")
        else:
            cb_render_start = cb_render_end = cb_play_start = cb_play_end = None

        self._player.enqueue_render(
            _render,
            label=f"tts:{text[:40]}",
            on_render_start=cb_render_start,
            on_render_end=cb_render_end,
            on_play_start=cb_play_start,
            on_play_end=cb_play_end,
        )

    def speak_sync(self, text: str):
        """Speak text and block until playback finishes."""
        self.speak(text)
        self.wait()

    def wait(self):
        """Block until all queued speech (and any other player work) drains."""
        if self._player:
            self._player.wait()


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
_tts_instance = None


def get_tts() -> SpotTTS | None:
    """Return the singleton SpotTTS instance, or None if not registered yet.

    Production path: ``client_mic.main()`` constructs SpotTTS explicitly with
    the shared AudioPlayer and assigns the result to ``_tts_instance``. This
    accessor is read-only — callers (notably ``spot_dispatch._handle_set_volume``)
    must handle ``None`` and degrade gracefully if the pipeline hasn't booted.

    The previous version of this function lazy-created a fresh ``SpotTTS()``
    when ``_tts_instance`` was None. That construction path auto-creates a
    private ``AudioPlayer``, which would compete with the real one for the
    audio device. The lazy fallback was dead code in production but a
    footgun if any new caller ever ran before client_mic registered the
    singleton, so we removed it.
    """
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
        # Direct WAV export (no AudioPlayer needed)
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
