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

# Stage 2E.1: route TTS synth through swappable backend registry.
import src.voice_control.tts.kokoro      # noqa: F401  (registers backend)
import src.voice_control.tts.elevenlabs  # noqa: F401  (registers backend)
from src.voice_control.tts import get_backend


def _default_voice_for_backend(backend) -> str:
    """Pick a sensible default voice when caller didn't specify one."""
    name = type(backend).__name__
    if "Kokoro" in name:
        return "af_sarah"
    if "ElevenLabs" in name:
        return "EXAVITQu4vr4xnSDxMaL"  # Sarah (Premade; Rachel Default retires 2026-12-31)
    return ""  # backend may raise on empty — intentional surface for bad config

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
        self.mouth = None  # set by client_mic after the Spot session is up (Stage 2E.2)

        self._load_model()

    def _load_model(self):
        """Stage 2E.1: backend lives in the TTS registry (KokoroBackend or
        ElevenLabsBackend). We just smoke-check the active backend here so
        cold init still fails loud if kokoro-onnx/elevenlabs is misconfigured.
        """
        if not self._player.is_available():
            print("[TTS] AudioPlayer unavailable — no audio playback")
            self._available = False
            return

        try:
            backend = get_backend()
        except Exception as e:
            print(f"[TTS] Backend init failed: {e}")
            print(f"[TTS] Check SPOT_TTS_BACKEND env (kokoro|elevenlabs)")
            self._available = False
            return

        backend_name = type(backend).__name__
        self._available = True
        print(f"[TTS] Ready — backend={backend_name}, voice={self.voice}, "
              f"speed={self.speed}, sample_rate=24000Hz")

    def is_available(self) -> bool:
        """Check if TTS is ready (backend resolved AND player ready)."""
        return self._available is True

    def set_volume(self, volume: float) -> float:
        """Set output gain. Clamped to 0.0..MAX_VOLUME. Returns the clamped value."""
        self.volume = max(0.0, min(MAX_VOLUME, float(volume)))
        return self.volume

    def speak(self, text: str, voice: str | None = None):
        """Enqueue ``text`` for synthesis and playback. Returns immediately.

        Kokoro inference happens inside the AudioPlayer worker thread, so
        the caller does not pay the ~1s rendering latency on its own thread.
        Multiple ``speak`` calls play in submission order — there is no
        cancel-and-replace behavior anymore.

        Pass ``voice`` to override the instance default for a single call
        (used by Stage 2E.1 personas without mutating ``self.voice``).
        """
        if not self.is_available():
            print(f'[TTS] Not available — would say: "{text}"')
            return
        if not text or not text.strip():
            return

        # Snapshot the volume at submission time so a later set_volume() does
        # not retroactively change the gain of an in-flight utterance.
        gain = self.volume

        # Stage 2E.2: per-utterance holder for the gripper-mouth envelope.
        # _render (worker thread) fills it; cb_play_start (same thread, later)
        # reads it. Same worker → no cross-thread race.
        frame_holder = {}
        import time as _t

        def _render():
            backend = get_backend()
            # self.voice is a Kokoro slug by default (e.g. af_sarah). Passing it
            # to ElevenLabs yields a 404 voice_not_found. Backend default wins
            # over self.voice; self.voice is last-resort only.
            # Mid-turn set_backend can leave the closure carrying a voice ID
            # for the OLD backend (kokoro slug → ElevenLabs, or vice versa).
            # Detect by shape: kokoro slugs are short and contain '_'.
            backend_name = type(backend).__name__
            voice_is_kokoro_shape = bool(voice) and "_" in (voice or "") and len(voice or "") < 16
            mismatch = bool(voice) and (
                ("ElevenLabs" in backend_name and voice_is_kokoro_shape)
                or ("Kokoro" in backend_name and not voice_is_kokoro_shape)
            )
            use_voice = (None if mismatch else voice) or _default_voice_for_backend(backend) or self.voice
            try:
                pcm_bytes = backend.synthesize(text, use_voice)
            except Exception as _e:
                print(f"[DBG-render] FAIL backend={type(backend).__name__} voice={use_voice} text={text[:40]!r} err={type(_e).__name__}:{_e}")
                raise
            print(f"[DBG-render] OK backend={type(backend).__name__} voice={use_voice} text={text[:40]!r} bytes={len(pcm_bytes)}")
            # Backend contract: 24 kHz int16 PCM bytes. Convert to float32 in [-1, 1].
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            if gain != 1.0:
                samples = samples * gain
            if self.mouth is not None and self.mouth.enabled:
                from src.voice_control.animation.mouth import envelope
                frame_holder["frames"] = envelope(
                    samples, 24000, fps=self.mouth.fps,
                    intensity=self.mouth.intensity)
            return samples, 24000

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

        # Stage 2E.2: drive the gripper mouth from the play callbacks. Compose
        # with (don't overwrite) any latency-trace callbacks set above.
        _trace_play_start, _trace_play_end = cb_play_start, cb_play_end

        def cb_play_start():
            if _trace_play_start:
                _trace_play_start()
            frames = frame_holder.get("frames")
            if self.mouth is not None and self.mouth.enabled and frames is not None and len(frames):
                self.mouth.play(frames, _t.monotonic())

        def cb_play_end():
            if _trace_play_end:
                _trace_play_end()
            if self.mouth is not None:
                self.mouth.close()

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

    def enqueue_streaming(self, voice: str | None = None):
        """Return a callable accepting LLM token deltas. Sentences flush to
        speak() as they complete. The callable carries ``flush()`` to emit
        any trailing partial sentence at stream end.
        """
        return TTSChunker(self, voice=voice)


# ---------------------------------------------------------------------------
# Streaming sentence chunker
# ---------------------------------------------------------------------------
import re

# Matches `"response": "` (with optional whitespace between colon and opening
# quote of the value) — llama-server prettyprints JSON with a space after `:`.
_RESPONSE_KEY_RE = re.compile(r'"response"\s*:\s*"')
_DESCRIBE_ACTION_RE = re.compile(r'"action"\s*:\s*"describe"')
# Gate for mid-stream split: fires only when a word character is already
# buffered after a sentence-end punctuation + whitespace.  This prevents
# pysbd from seeing "Dr. " (trailing space, no following word) and
# erroneously marking it complete — pysbd needs the next word token to
# resolve abbreviation vs. sentence-end (e.g. "Dr. " fires False,
# "Dr. L" fires True).
_POST_PUNCT_WORD_RE = re.compile(r"[.!?]\s+\w")

class TTSChunker:
    """Buffer streaming LLM tokens; flush a sentence to TTS on each boundary.

    Parses surrounding GBNF JSON incrementally: once inside the
    ``"response":"…"`` string value, characters flow to a sentence buffer
    that drains on each `.`/`!`/`?` followed by whitespace.

    Instance is callable: ``chunker(delta)`` is shorthand for ``chunker.accept(delta)``.
    """

    def __init__(self, tts: "SpotTTS", voice: str | None = None):
        self._tts = tts
        self._voice = voice
        self.buf: list[str] = []
        self.in_response = False
        self.escape = False
        self.json_buf: list[str] = []
        self.suppressed = False

    def __call__(self, delta: str) -> None:
        self.accept(delta)

    def _emit(self, chunk: str) -> None:
        if self.suppressed:
            return
        self._tts.speak(chunk, voice=self._voice)

    def accept(self, delta: str) -> None:
        for ch in delta:
            self.json_buf.append(ch)
            if not self.in_response:
                tail = "".join(self.json_buf[-25:])
                if _RESPONSE_KEY_RE.search(tail):
                    # GBNF emits actions before response, so json_buf already
                    # has all actions. If a describe is queued, suppress
                    # response streaming — the LLM is hallucinating about a
                    # camera view it can't actually see; client_mic.py will
                    # speak a stock ack ("Let me take a look...") instead.
                    full = "".join(self.json_buf)
                    if _DESCRIBE_ACTION_RE.search(full):
                        self.suppressed = True
                    self.in_response = True
                continue
            if self.escape:
                self.buf.append(ch)
                self.escape = False
            elif ch == "\\":
                self.escape = True
            elif ch == '"':
                self.in_response = False
                rest = "".join(self.buf).strip()
                if rest:
                    self._emit(rest)
                self.buf = []
            else:
                self.buf.append(ch)
                if _POST_PUNCT_WORD_RE.search("".join(self.buf)):
                    from src.voice_control.text_segment import split_sentences
                    complete, remainder = split_sentences("".join(self.buf))
                    if complete:
                        for sent in complete:
                            self._emit(sent)
                        self.buf = list(remainder)

    def flush(self) -> None:
        """Emit any trailing partial sentence (called at end of stream)."""
        if self.in_response:
            from src.voice_control.text_segment import split_sentences
            text = "".join(self.buf).strip()
            if text:
                complete, remainder = split_sentences(text)
                for sent in complete:
                    self._emit(sent)
                if remainder:
                    self._emit(remainder)
        self.buf = []


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
