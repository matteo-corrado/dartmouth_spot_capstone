"""
Voice control client for Spot with LLM brain.

Architecture (Boston Dynamics "Robots That Can Chat" style):
    Mic → VAD → Whisper ASR → LLM Brain (state + history + personality) → Action + Response

Safety commands (stop/estop/freeze) bypass the LLM for zero-latency execution.
Everything else goes through the LLM brain which decides what to do AND what to say.
"""
import os
import sys
import pathlib

# Add parent and project root to path for local imports — must run BEFORE
# any `from src.*` import below, otherwise launching client_mic.py as a
# script (e.g. via run_voice_control.py subprocess) fails with
# ModuleNotFoundError because sys.path[0] is src/voice_control/, not the
# project root.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Auto-load .env from project root so callers don't need `source .env`
# first. `override=True` so .env wins over stale shell exports (e.g. an old
# SPOT_TTS_BACKEND=kokoro from a prior session). Broad except: dotenv may
# not be installed, .env may be unreadable (PermissionError) or non-UTF-8
# (UnicodeDecodeError); none of those should crash the launcher.
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env", override=True)
except Exception:
    pass

from src.voice_control.chatbot.persona import get_persona, voice_id_for

import re
import glob
import json
import math
import time
import queue
import random
import signal
import argparse
import subprocess
import threading
import numpy as np
import sounddevice as sd
from sherpa_onnx import VoiceActivityDetector, VadModelConfig, SileroVadModelConfig
import grpc


# Shutdown signal handling.
#
# First SIGINT or SIGTERM → raise KeyboardInterrupt so the main loop's
# finally block runs the graceful shutdown (cancel nav, sit robot down,
# power off, drain audio player).
#
# Second signal → os._exit() immediately. cleanup_spot() can take tens of
# seconds for legitimate reasons (blocking_sit + power_off are bosdyn
# gRPC blocking calls that aren't interruptible from Python while they're
# inside C++). Without an escape hatch, a user pressing Ctrl+C in the
# terminal can find themselves unable to exit at all — we observed this
# in the field, where Ctrl+C fired but the pipeline kept running and the
# user had to manually `kill` the processes. The hard-exit on the second
# signal guarantees they can always escape.
#
# Both SIGINT and SIGTERM share this handler so the web panel's
# SIGINT-then-SIGTERM escalation also gets the hard-exit on the SIGTERM.
_shutdown_requested = False

def _shutdown_signal_handler(signum, frame):
    global _shutdown_requested
    if _shutdown_requested:
        # Second signal — escape hatch. cleanup is wedged; bail.
        print("\n[Shutdown] second signal received — force exiting", flush=True)
        os._exit(130 if signum == signal.SIGINT else 143)
    _shutdown_requested = True
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, _shutdown_signal_handler)
signal.signal(signal.SIGTERM, _shutdown_signal_handler)

from contextlib import nullcontext

from asr_pb2 import StreamingRequest, StreamingConfig, AudioChunk
from asr_pb2_grpc import ASRStub
from intent import parse_intent
from llm_brain import LLMBrain, DEFAULT_MODEL
from audio_player import AudioPlayer
from spot_tts import SpotTTS
from audio_feedback import beep
from latency import init_recorder, get_recorder
from src.voice_control import startup_status  # noqa: F401 — available for later use
from src.voice_control.response_gate import should_arm_reopen, should_reopen_mic
from src.voice_control.tuning import get_tuning
from src.voice_control.barge_in import should_reset_after_player
from src.voice_control.audio_frontend import (
    frame_dbfs, FloorTracker, Endpointer, UtteranceStats, passes_gate)
from src.voice_control.speaker_lock import (
    SpeakerLock, circular_median_deg, mic_to_body_deg)

_tuning = get_tuning()
from src.voice_control.voice_state import VoiceState
from src.voice_control.state_feedback import chime_for
try:
    from src.voice_control.state_feedback import ring_for as _ring_for
except ImportError:  # pre-Gate-2 base (Lane C adds ring_for): no ring
    _ring_for = None


# ============================================================================
# Device Lookup (by name, so index changes don't break things)
# ============================================================================
MIC_DEVICE_NAME = "XVF3800"          # reSpeaker XVF3800 4-Mic Array
SPEAKER_DEVICE_NAME = "UACDemoV1.0"  # USB speaker on Spot CORE I/O

def _find_device_by_name(substring: str, kind: str = "input") -> int | None:
    """Find audio device index by substring match on its name.

    Args:
        substring: Partial name to match (case-insensitive).
        kind: "input" to require input channels, "output" for output channels.

    Returns:
        Device index or None if not found.
    """
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if substring.lower() in dev["name"].lower():
            if kind == "input" and dev["max_input_channels"] > 0:
                return i
            elif kind == "output" and dev["max_output_channels"] > 0:
                return i
    return None

# VoiceState moved to src/voice_control/voice_state.py (leaf module) and
# chime_for is imported up top — see the import block near the file header.
# Both broke the old client_mic <-> state_feedback circular import.


# The conversation timeout is Tuning.lock_timeout_s (20 s default; was a fixed
# 15 s LISTENING_TIMEOUT before the speaker lock).

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_S = FRAME_MS / 1000.0      # matches FloorTracker/UtteranceStats frame_s=0.03
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples per frame
BYTES_PER_FRAME = FRAME_SAMPLES * 2  # int16 = 2 bytes

SILERO_VAD_PATH = "/mnt/ssd/vad-models/silero_vad.onnx"
VAD_WINDOW_SAMPLES = 512         # Silero default @ 16 kHz = 32 ms; VAD buffers internally
# End silence: Tuning.end_silence_s_short until the utterance is this long,
# then Tuning.end_silence_s (chained commands pause mid-sentence).
SILENCE_CROSSOVER = 2.0
PREROLL_FRAMES = 10              # frames kept before onset (~300ms; covers Silero's onset lag)
TRAILING_KEEP_FRAMES = 10        # silence kept after the last speech frame for ASR (~300ms)
NOISE_CALIBRATION_SECONDS = 2    # Seconds to measure ambient noise
NAV_SPEECH_DB_BOOST = 6.0        # added to Tuning.min_speech_dbfs while the robot moves
LOCK_ACCEPT_MAX_SAMPLES = 100    # cap DoA samples per lock.accept (medoid is O(n^2))
UTTERANCE_LOG_DIR = project_root / "logs" / "utterances"


def _frames(seconds: float) -> int:
    """Seconds -> 30 ms frame count (Endpointer is frame-based)."""
    return max(1, int(round(seconds / FRAME_S)))


def dbfs_from_rms(rms: float) -> float:
    """Linear RMS (full scale 1.0) -> dBFS, floored at -120."""
    if rms <= 1e-6:
        return -120.0
    return max(-120.0, 20.0 * math.log10(rms))


def speech_degs(samples) -> list:
    """Mic-frame degrees of the speech-flagged DoA samples (t, deg, speech).
    Non-finite readings are dropped before lock/accept."""
    out = []
    for _t, deg, speech in samples:
        if speech and deg is not None and math.isfinite(float(deg)):
            out.append(float(deg))
    return out


def last_speech_t(utt) -> float:
    """Timestamp of the last speech-flagged frame (span end for DoA, so the
    end-silence tail isn't counted); last frame if none flagged."""
    for t, _f, _d, s in reversed(utt):
        if s:
            return t
    return utt[-1][0]


def trim_trailing(utt, keep_frames: int):
    """Drop end-silence frames beyond `keep_frames` after the last speech
    frame (the endpointer waits ~1.5 s of silence; ASR doesn't need it)."""
    last = None
    for i in range(len(utt) - 1, -1, -1):
        if utt[i][3]:
            last = i
            break
    if last is None:
        return list(utt)
    return list(utt[:last + 1 + keep_frames])


def conversation_expired(lock, listening_since: float, now: float, timeout_s: float) -> bool:
    """Conversation timeout. Locked: SpeakerLock.expired. Unlocked (no mic /
    no direction readings): wall clock since listening started, so an
    unlocked LISTENING state can't stay open forever in a crowd."""
    if lock.locked:
        return lock.expired(now)
    return (now - listening_since) > timeout_s


def _body(deg):
    if deg is None:
        return None
    return round(mic_to_body_deg(deg, _tuning.mic_yaw_offset_deg, _tuning.mic_yaw_sign), 1)


def _rnd(x):
    return None if x is None else round(float(x), 1)


def build_utterance_record(endpoint, end_silence_s, duration_s, floor_db, stats,
                           lock_mic_deg, utt_mic_deg, accepted, transcript=None,
                           reached_llm=False, latency_s=None, drop_reason=None) -> dict:
    """One metrics line per utterance (logs/utterances/<date>.jsonl)."""
    return {
        "kind": "utterance",
        "ts": time.time(),
        "floor_dbfs": _rnd(floor_db),
        "speech_dbfs_median": _rnd(stats.speech_db_median),
        "speech_dbfs_p90": _rnd(stats.speech_db_p90),
        "snr_db": _rnd(stats.snr_db(floor_db)),
        "endpoint": endpoint,
        "end_silence_s": end_silence_s,
        "duration_s": round(duration_s, 2),
        "lock_mic_deg": _rnd(lock_mic_deg),
        "utt_mic_deg": _rnd(utt_mic_deg),
        "lock_body_deg": _body(lock_mic_deg),
        "utt_body_deg": _body(utt_mic_deg),
        "accepted": accepted,
        "drop_reason": drop_reason,
        "transcript": transcript,
        "reached_llm": reached_llm,
        "latency_s": None if latency_s is None else round(latency_s, 2),
    }


def build_idle_record(state_name, floor_db, frame_db, lock_mic_deg) -> dict:
    return {"kind": "idle", "ts": time.time(), "state": state_name,
            "floor_dbfs": _rnd(floor_db), "frame_dbfs": _rnd(frame_db),
            "lock_mic_deg": _rnd(lock_mic_deg), "lock_body_deg": _body(lock_mic_deg)}


class UtteranceLog:
    """Append-only JSONL metrics. Opens lazily; an I/O error is logged once
    and never reaches the voice loop. store_transcript=False -> null."""

    def __init__(self, log_dir, store_transcript: bool = True):
        self._dir = pathlib.Path(log_dir)
        self.store_transcript = store_transcript
        self._failed = False

    def write(self, record: dict) -> None:
        if self._failed:
            return
        if not self.store_transcript and "transcript" in record:
            record = dict(record, transcript=None)
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"{time.strftime('%Y-%m-%d')}.jsonl"
            with open(path, "a") as fh:
                fh.write(json.dumps(record) + "\n")
        except (OSError, TypeError, ValueError) as e:
            self._failed = True
            print(f"[Metrics] utterance log disabled: {type(e).__name__}: {e}")

# Inter-action waits when chaining LLM actions (e.g. "walk forward then sit").
# Empirical values — give Spot enough time for the previous command to finish
# before issuing the next, otherwise the second command races against the first.
CHAIN_MOVEMENT_WAIT_S = 3.0  # walk / strafe / turn (timed locomotion)
CHAIN_POSTURE_WAIT_S = 4.0   # sit / stand / selfright / body_height (slower body changes)

# Stock acknowledgments spoken before a VLM query. The LLM's own "response"
# field for a describe action is often a hallucinated answer (it can't see
# the camera), so we override it with one of these to fill the ~2-5s gap
# while the VLM actually runs. See process_utterance() for the override.
_DESCRIBE_ACKS = (
    "Let me take a look...",
    "One sec, checking now...",
    "Hold on, looking around...",
    "Taking a look...",
    "Let me see what's in front of me...",
    "Checking my camera...",
)

# ============================================================================
# Audio Queue (filled by callback)
# ============================================================================
# Bounded so that a wedged main loop or a long mic-mute window can't grow
# the queue without limit. 33 frames at 30ms each = ~1s of headroom, well
# above the 0.15s Tuning.speaker_tail_s grace period. On overflow we
# drop frames in the callback (see audio_callback) rather than blocking
# PortAudio's high-priority thread.
AUDIO_QUEUE_MAX = 33
audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
MIC_CHANNEL = _tuning.mic_channel  # 0=left (beamformed), 1=right (ASR); --mic-channel

# Mic-mute interlock — derived from AudioPlayer state instead of a flag.
# The audio callback discards input frames whenever the player is busy
# (TTS speaking or beep playing) so the mic can never feed Spot's own
# voice back into ASR. Wired up in main() before the input stream opens.
_audio_player: AudioPlayer | None = None

# Stage 2F P6: hard mic gate during THINKING/RESPONDING + the 0.5s reopen
# settle. The audio callback drops frames while gated, so no speech uttered
# from the moment Spot commits to a response until 0.5s after it stops
# speaking is ever captured (no barge-in, no late phantom commands).
_mic_gated = False
_mic_reopen_at = None  # monotonic time the post-response settle ends
_state_leds = None     # StateLeds instance (set in main); None = LEDs off

# Cooldown after the player goes idle: the OS audio buffer keeps draining
# for ~tens of ms after stream.write() returns, so the mic would otherwise
# pick up the tail of Spot's own utterance. Length = Tuning.speaker_tail_s,
# measured from the last frame where Spot's SPEECH was busy.
_player_last_busy_at = 0.0  # monotonic timestamp of the most recent speech-busy frame

# Operator control (spotctl, Task 4). Plain module globals: the HTTP threads
# read/flip them without locks so a slow request can never delay a stop.
_operator_muted = False
_crowd_mode = False        # after each reply drop back to WAKE_WORD
_voiceprint_mode = False   # flag only (voiceprint gating is Wave 3)
# Set by an operator stop/freeze/sit/stand/stow/hush: the chained-action loop
# in process_utterance checks it before each step and waits on it instead of
# sleeping, so an operator stop also cancels the rest of a chain.
_abort_chain = threading.Event()
# Snapshot the main loop writes and `status` reads (no locks, never blocks).
_status = {
    "state": "WAKE_WORD", "locked_deg": None, "floor_dbfs": None,
    "last_transcript": None, "last_latency_s": None,
}
_MOTION_COMMANDS = ("stop", "freeze", "sit", "stand", "stow")

# XVF3800 control (DoA, LED ring, fixed beams). Optional: None when unplugged
# and treated as absent once `dead` (any USB error) — the loop never stops.
_mic = None
_mic_dead_logged = False
# Speaker lock, MIC frame degrees (body frame only for status/metrics/gaze).
_lock = SpeakerLock(_tuning.lock_window_deg, _tuning.lock_timeout_s, _tuning.lock_nudge)


def _mic_usable() -> bool:
    return _mic is not None and not getattr(_mic, "dead", True)


def _check_mic_alive() -> None:
    """Log once when the XVF3800 dies mid-session; lock/ring then degrade."""
    global _mic_dead_logged
    if _mic is not None and getattr(_mic, "dead", False) and not _mic_dead_logged:
        _mic_dead_logged = True
        print("[MicArray] XVF3800 control lost — speaker lock + ring disabled, voice continues")


def _mic_speech_degs(t0: float, t1: float) -> list:
    """Speech-flagged DoA degrees (mic frame) in [t0, t1] (monotonic)."""
    if not _mic_usable():
        return []
    try:
        return speech_degs(_mic.samples_between(t0, t1))
    except Exception as e:
        print(f"[MicArray] DoA read failed: {e}")
        return []


def _body_deg(deg):
    return None if deg is None else mic_to_body_deg(
        deg, _tuning.mic_yaw_offset_deg, _tuning.mic_yaw_sign)


def _open_mic_array():
    """Open the XVF3800 control interface + start the DoA poller. Never raises."""
    try:
        from src.voice_control.mic_array import MicArray
        mic = MicArray.open()
    except Exception as e:
        print(f"[MicArray] unavailable: {type(e).__name__}: {e}")
        return None
    if mic is None:
        print("[MicArray] XVF3800 control not found — no speaker lock / ring")
        return None
    try:
        mic.start_poller(20.0)
    except Exception as e:
        print(f"[MicArray] DoA poller failed: {e} — no speaker lock")
    print("[MicArray] XVF3800 control ready (DoA poller 20 Hz)")
    return mic


def _new_endpointer() -> Endpointer:
    return Endpointer(_tuning.onset_frames, _frames(_tuning.end_silence_s_short),
                      _frames(_tuning.max_utterance_s))

# The mic-mute watchdog lives inside AudioPlayer (audio_player.py) as its own
# daemon thread. Putting it there instead of the main loop here means it can
# still recover from a wedge that happens *inside* process_utterance, when
# this loop is blocked on tts.wait() — which was the exact failure shape that
# motivated the refactor.


def _speech_busy() -> bool:
    """True while Spot's own SPEECH is queued/playing. Chimes (beep: labels)
    don't count, so a turn cue never mutes the mic (Task 8)."""
    p = _audio_player
    if p is None:
        return False
    fn = getattr(p, "is_speech_busy", None) or p.is_busy  # pre-Gate-2 player lacks it
    return fn()


def audio_callback(indata, frames, time_info, status):
    """Sounddevice callback - converts stereo to mono PCM16.

    Mute logic: while the AudioPlayer is busy (TTS speaking or beep playing)
    we discard frames, and we keep discarding for Tuning.speaker_tail_s
    after it goes idle so the OS audio buffer has time to fully drain.
    The watchdog that recovers a wedged player runs in the main loop, not
    here — this callback runs on PortAudio's high-priority thread and must
    not do heavy work or block.
    """
    global _player_last_busy_at
    now = time.monotonic()  # local clock — PortAudio's per-stream clock isn't comparable
    if _operator_muted:
        return  # operator mute (spotctl mute) — discard everything
    if _mic_gated or _speech_busy():
        _player_last_busy_at = now
        return  # discard frame — gated for a response, or Spot is talking
    if now - _player_last_busy_at < _tuning.speaker_tail_s:
        return  # cooldown — speaker buffer still draining
    if status:
        print(f"[Audio: {status}]")
    # Select channel: 0=left (beamformed), 1=right (ASR) for stereo mics like XVF3800
    if indata.shape[1] >= 2:
        mono = indata[:, MIC_CHANNEL].astype(np.float32)
    else:
        mono = indata[:, 0].astype(np.float32)
    pcm16 = (mono * 32767).astype(np.int16).tobytes()
    try:
        audio_queue.put_nowait((now, pcm16))  # monotonic stamp (DoA/lock clock)
    except queue.Full:
        # Drop the frame instead of blocking PortAudio's callback thread.
        # ~1s of frames are already buffered; if the main loop is that far
        # behind, the right answer is to keep the audio device responsive.
        pass


# ============================================================================
# Mic stream open (with retry on transient device-busy)
# ============================================================================
# PortAudio reports both "wrong channel count" and "device busy" through the
# same generic exception. The original code caught any failure, assumed it
# was a channel mismatch, and printed "Stereo not supported, falling back to
# mono" — which masked the real cause when the actual problem was that a
# previous client_mic.py instance was still releasing the snd_usb_audio
# substream (kernel-release lag after the prior process exited). The helper
# below distinguishes the two cases and retries on busy.

def _is_format_error(err: Exception) -> bool:
    """True if a PortAudio error is a real channel/sample-rate mismatch.

    Anything else (especially -9985 / paDeviceUnavailable) is treated as a
    transient busy condition worth retrying.
    """
    msg = str(err).lower()
    return ("invalid number of channels" in msg
            or "invalid sample rate" in msg
            or "incompatible" in msg)


def _log_pcm_holders() -> None:
    """Print processes holding any ALSA capture PCM. Best-effort, never raises.

    Called when a stream open fails with a non-format error so we can see
    *who* is squatting on the mic. Output is empty when the kernel hasn't
    finished releasing the substream from a just-exited holder — that empty
    case is itself the diagnosis (kernel-release lag).
    """
    try:
        pcm_files = sorted(glob.glob("/dev/snd/pcmC*c"))
        if not pcm_files:
            print("[mic-busy probe] no /dev/snd/pcmC*c devices found")
            return
        result = subprocess.run(
            ["fuser", "-v"] + pcm_files,
            capture_output=True, text=True, timeout=2,
        )
        # fuser writes its table to stderr; combine for safety.
        out = ((result.stderr or "") + (result.stdout or "")).strip()
        if out:
            print(f"[mic-busy probe] capture PCM holders:\n{out}")
        else:
            print("[mic-busy probe] no PCM holders — likely kernel-release lag")
    except FileNotFoundError:
        print("[mic-busy probe] fuser not installed; skipping")
    except Exception as exc:
        print(f"[mic-busy probe failed: {exc}]")


def _open_input_stream_with_retry(device, callback, *,
                                  attempts: int = 6,
                                  backoff: float = 0.5) -> sd.InputStream:
    """Open a stereo InputStream, retrying on transient device-busy errors.

    - Real channel/format mismatches → fall back to mono on the first attempt.
    - Anything else (e.g. -9985 paDeviceUnavailable) → retry with backoff.
      On the first failure we dump the current PCM holders so the next
      occurrence is diagnosable from the log.
    """
    last_err: Exception | None = None
    for attempt in range(attempts):
        try:
            return sd.InputStream(
                device=device, channels=2, samplerate=SAMPLE_RATE,
                callback=callback, blocksize=FRAME_SAMPLES,
            )
        except Exception as e:
            if _is_format_error(e):
                print(f"[Stereo not supported on device {device}, falling back to mono]")
                return sd.InputStream(
                    device=device, channels=1, samplerate=SAMPLE_RATE,
                    callback=callback, blocksize=FRAME_SAMPLES,
                )
            last_err = e
            if attempt == 0:
                print(f"[mic open failed: {e}]")
                _log_pcm_holders()
            if attempt < attempts - 1:
                print(f"[mic busy; retry {attempt + 1}/{attempts - 1} in {backoff}s]")
                time.sleep(backoff)
    assert last_err is not None
    raise last_err


# ============================================================================
# Noise Calibration
# ============================================================================
def calibrate_noise_floor(duration_sec: float, device=None) -> float:
    """
    Measure ambient noise. Returns the MEASURED noise RMS (always — it seeds
    the FloorTracker once, A2); a high reading only warns.
    """
    print(f"\n[Calibrating noise floor for {duration_sec}s - please stay quiet...]")

    samples = []
    samples_needed = int(SAMPLE_RATE * duration_sec)

    def callback(indata, frames, time_info, status):
        if indata.shape[1] >= 2:
            mono = indata[:, MIC_CHANNEL].astype(np.float32)
        else:
            mono = indata[:, 0].astype(np.float32)
        samples.append(mono.copy())

    try:
        stream_ctx = _open_input_stream_with_retry(device, callback)
        with stream_ctx:
            start = time.time()
            while sum(len(s) for s in samples) < samples_needed:
                time.sleep(0.05)
                if time.time() - start > duration_sec + 1:
                    break

        noise_audio = np.concatenate(samples)[:samples_needed]
        noise_rms = np.sqrt(np.mean(noise_audio ** 2))
        print(f"[Noise floor RMS: {noise_rms:.5f}]")

        # Suspiciously high noise: robot motors / crowd during calibration.
        # Keep the measured value (the floor tracker falls instantly if the
        # room gets quieter) but tell the operator.
        if noise_rms > 0.003:
            print(f"[WARNING: High noise ({noise_rms:.5f}, {dbfs_from_rms(noise_rms):.1f} dBFS) "
                  f"— robot motors or crowd running during calibration?]")

        return float(noise_rms)

    except Exception as e:
        print(f"[Calibration error: {e}]")
        return 0.01  # Default fallback


# ============================================================================
# Audio Processing
# ============================================================================
def pcm16_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """Convert PCM16 bytes to float32 array."""
    return np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert float32 array to PCM16 bytes."""
    return (audio * 32767).astype(np.int16).tobytes()


# ============================================================================
# ASR Communication
# ============================================================================
def send_to_asr(stub, pcm_bytes: bytes) -> str:
    """Stream audio to ASR server and return transcript."""
    def request_generator():
        # Send config first
        yield StreamingRequest(config=StreamingConfig(
            language_code="en",
            sample_rate_hz=SAMPLE_RATE,
            enable_punctuation=True
        ))
        # Send audio chunks
        chunk_size = BYTES_PER_FRAME
        for i in range(0, len(pcm_bytes), chunk_size):
            yield StreamingRequest(audio=AudioChunk(pcm16=pcm_bytes[i:i+chunk_size]))

    try:
        responses = stub.StreamingRecognize(request_generator())
        for response in responses:
            if response.is_final:
                return response.transcript.strip()
        return ""
    except Exception as e:
        print(f"[ASR error: {e}]")
        return ""


# ============================================================================
# Intent Execution
# ============================================================================
# State-only intents mutate brain/TTS state and do NOT touch the robot;
# dispatch_intent handles them BEFORE ensure_spot_session(), so they must run
# even in dry-run mode (Spot off). Otherwise "switch to pirate" no-ops and
# every subsequent voice_id resolves to the prior persona's voice.
_STATE_ONLY_INTENTS = {"set_persona", "add_persona", "set_volume", "set_backend"}


def execute_on_spot(intent: dict, brain=None, speak=None) -> bool:
    """Execute parsed intent on Spot robot.

    `brain` is forwarded to dispatch_intent so handlers needing across-turn
    state (set_persona today, follow-ups later) can mutate it. Safety + regex
    fallback paths leave it None — those intents never touch brain state.
    """
    intent_name = intent.get("intent", "")
    if os.environ.get("SPOT_DRY_RUN") == "1" and intent_name not in _STATE_ONLY_INTENTS:
        print(f"[dry-run] would dispatch: {intent}")
        return True
    try:
        from src.voice_control.spot_dispatch import dispatch_intent
        return dispatch_intent(intent, brain=brain, speak=speak)
    except Exception as e:
        print(f"[Spot error: {e}]")
        return False


def get_spot_state() -> dict:
    """Get current robot state for LLM context. Short-circuits in dry-run
    mode so the auto-degrade path stays robot-free (no spot_dispatch
    import, no disk I/O for locations.json / maps dir) per turn."""
    if os.environ.get("SPOT_DRY_RUN") == "1":
        return {}
    try:
        from src.voice_control.spot_dispatch import get_robot_state_dict
        return get_robot_state_dict()
    except Exception as e:
        print(f"[State error: {e}]")
        return {}


def cleanup_spot():
    """Clean up Spot session and audio player on exit.

    Order matters:

    1. Cancel any in-progress navigation thread FIRST. follow_me /
       go_to_object / tour / patrol all run in a daemon thread that
       keeps issuing velocity / trajectory commands until its
       ``stop_event`` is set. If we skip this and go straight to
       ``blocking_sit``, the daemon's velocity stream fights the sit
       command — both bosdyn calls hit their full 15s + 20s timeouts
       and the whole shutdown drags on so long that the user thinks
       Ctrl+C didn't work and reaches for ``kill``.

    2. Then close the Spot session (sit + power_off + lease release).

    3. Then shut the AudioPlayer down. Doing this BEFORE the process
       exits keeps the daemon worker thread from being torn down mid
       ``stream.write()``, which would corrupt the heap (the same
       ``malloc(): unaligned tcache chunk`` bug we already fixed for
       force_reset). The shutdown call drains the queue, joins the
       worker, and closes the stream gracefully.
    """
    # In dry-run mode there is no Spot session to close; skip the
    # spot_dispatch imports entirely so cleanup doesn't trigger bosdyn
    # module-load side effects in a path advertised as robot-free.
    if os.environ.get("SPOT_DRY_RUN") != "1":
        try:
            from src.voice_control.spot_dispatch import _cancel_nav
            _cancel_nav()
        except Exception as e:
            print(f"[Spot] cancel_nav error during cleanup: {e}")
        try:
            from src.voice_control.spot_dispatch import close_spot_session
            close_spot_session()
        except Exception:
            pass
    if _audio_player is not None:
        try:
            _audio_player.shutdown()
        except Exception as e:
            print(f"[Player] shutdown error during cleanup: {e}")


def _is_robot_moving() -> bool:
    """Check if robot is actively navigating (motor noise expected)."""
    if os.environ.get("SPOT_DRY_RUN") == "1":
        return False  # no robot; don't import spot_dispatch (bosdyn) per utterance
    try:
        from src.voice_control.spot_dispatch import is_navigating
        return is_navigating()
    except Exception:
        return False


def _is_follow_mode() -> bool:
    """Check if robot is in follow-person mode (user is nearby speaking)."""
    try:
        from src.voice_control.spot_dispatch import is_follow_mode
        return is_follow_mode()
    except Exception:
        return False


def _wait_for_nav_complete(timeout: float = 120.0):
    """Block until current navigation finishes (for command chaining)."""
    start = time.time()
    while _is_robot_moving() and (time.time() - start) < timeout:
        if _abort_chain.wait(0.5):
            return  # operator stop — don't keep the chain waiting


# Safety commands (stop/freeze/estop) bypass the LLM. The rule + patterns live
# in intent.py (single source); re-exported here because tests and the
# barge-in eval import check_safety_command from client_mic.
from src.voice_control.intent import check_safety_command, is_filler_only  # noqa: E402


def _run_safety(text: str) -> bool:
    """If `text` is a safety command, execute it now and return True.
    Also aborts any action chain in progress."""
    safety = check_safety_command(text)
    if not safety:
        return False
    print(f"[SAFETY] {safety['intent']} — executing immediately")
    _abort_chain.set()
    if execute_on_spot(safety):
        beep.command_ok()
        print(">>> SAFETY COMMAND EXECUTED")
    else:
        beep.error()
        print(">>> SAFETY COMMAND FAILED")
    return True


# Wake phrase pattern — fallback for ASR-based detection if dedicated detector unavailable
WAKE_PHRASE_PATTERN = re.compile(
    r"(?:hey|a|stay|say|heh)\s*[,\-]?\s*spot\b",
    re.IGNORECASE
)


def _enter_state(new_state):
    """Play the entry chime + set LEDs for a state transition. Best-effort —
    a chime or LED failure must NEVER crash the voice loop, so each indicator
    is isolated in its own try/except (the loop only catches KeyboardInterrupt)."""
    try:
        name = chime_for(new_state)
        if name is not None:
            getattr(beep, name)()
    except Exception as e:
        print(f"[Indicator] chime failed: {type(e).__name__}: {e}")
    if _state_leds is not None:
        try:
            _state_leds.set_state(new_state)
        except Exception as e:
            print(f"[Indicator] LED failed: {type(e).__name__}: {e}")
    global _ring_state
    _ring_state = new_state
    _update_ring()


# ----------------------------------------------------------------------------
# XVF3800 LED ring (Task 11): the primary turn signal; points at the locked
# speaker while LISTENING. Writes are throttled to Tuning.ring_max_hz; a
# throttled update is marked pending and flushed from the main loop. Every
# device call is isolated — a missing/dead mic only means no ring.
# ----------------------------------------------------------------------------
REJECT_FLASH_S = 0.2
REJECT_COLOR = 0xC00000
LED_SOLID = 3
_ring_state = None          # VoiceState the ring currently shows
_ring_last_write = 0.0
_ring_pending = False
_reject_until = 0.0


class RingThrottle:
    """Pure rate limiter for ring writes (min interval = 1 / max_hz)."""

    def __init__(self, max_hz: float):
        self.min_interval = 1.0 / max_hz if max_hz > 0 else 0.0
        self._last = None

    def allow(self, now: float) -> bool:
        if self._last is not None and now - self._last < self.min_interval:
            return False
        self._last = now
        return True


_ring_throttle = RingThrottle(_tuning.ring_max_hz)


def _current_ring_spec():
    if _ring_for is None or _ring_state is None:
        return None
    # Lock angle is MIC frame, which is what ring_for/ring_toward expect.
    return _ring_for(_ring_state, _lock.angle, _operator_muted,
                     ring_offset_leds=_tuning.ring_offset_leds,
                     reverse=_tuning.ring_reverse)


def _write_ring(effect, color, ring):
    for fn, val in ((_mic.set_led_effect, effect), (_mic.set_led_color, color),
                    (_mic.set_ring, ring)):
        if val is None:
            continue
        try:
            fn(val)
        except Exception as e:
            print(f"[Indicator] ring write failed: {type(e).__name__}: {e}")


def _update_ring() -> None:
    """Render the ring for (state, lock, muted) now, or mark it pending if
    throttled / mid reject-flash. Never raises, never sleeps."""
    global _ring_pending
    if not _mic_usable():
        return
    try:
        now = time.monotonic()
        if now < _reject_until or not _ring_throttle.allow(now):
            _ring_pending = True
            return
        _ring_pending = False
        spec = _current_ring_spec()
        if spec is not None:
            _write_ring(spec.effect, spec.color, spec.ring)
    except Exception as e:
        print(f"[Indicator] ring failed: {type(e).__name__}: {e}")


def _flush_ring() -> None:
    """Main-loop hook: apply a throttled/pending ring update."""
    if _ring_pending:
        _update_ring()


def _end_reject_flash() -> None:
    global _reject_until, _ring_pending
    _reject_until = 0.0
    _ring_pending = True
    _update_ring()


def _signal_reject():
    """Visual-only cue that an utterance from outside the lock was ignored:
    the ring flashes red for 200 ms, then a Timer restores the current spec
    (never sleeps on the main loop)."""
    global _reject_until
    print("[Indicator] reject (off-lock speaker)")
    if not _mic_usable():
        return
    _reject_until = time.monotonic() + REJECT_FLASH_S
    _write_ring(LED_SOLID, REJECT_COLOR, None)
    t = threading.Timer(REJECT_FLASH_S, _end_reject_flash)
    t.daemon = True
    t.start()


# ============================================================================
# Operator control (spotctl) — immediate commands
# ============================================================================
def _hush():
    """Cut speech now: flush the player and latch the gripper mouth shut."""
    if _audio_player is not None:
        try:
            _audio_player.force_reset()
        except Exception as e:
            print(f"[Control] hush: player reset failed: {e}")
    if os.environ.get("SPOT_DRY_RUN") != "1":
        try:
            from src.voice_control.spot_dispatch import _close_mouth
            _close_mouth()
        except Exception as e:
            print(f"[Control] hush: close mouth failed: {e}")


def _status_snapshot() -> dict:
    """Plain-attribute read; never blocks."""
    s = dict(_status)
    s.update(muted=_operator_muted, crowd=_crowd_mode, voiceprint=_voiceprint_mode)
    return s


def _on_immediate(name: str, arg=None) -> dict:
    """spotctl immediate commands, run on HTTP worker threads.

    Re-entrant and lock-free: touches only the player, the mouth and the
    dispatcher (via execute_on_spot, which honors dry-run), plus plain flags.
    Motion commands also set _abort_chain so a chain in progress stops.
    """
    global _operator_muted, _ring_pending
    if name == "status":
        return _status_snapshot()
    if name == "hush":
        _abort_chain.set()
        _hush()
        return {"ok": True}
    if name in _MOTION_COMMANDS:
        _abort_chain.set()
        ok = execute_on_spot({"intent": name, "params": {}, "raw": f"spotctl {name}"})
        return {"ok": bool(ok)}
    if name in ("mute", "unmute"):
        _operator_muted = (name == "mute")
        _ring_pending = True  # main loop repaints (red while muted)
        return {"ok": True, "muted": _operator_muted}
    return {"error": f"unhandled immediate command {name!r}"}


def _toggle(current: bool, arg) -> bool:
    """spotctl toggle: explicit on/off arg wins, otherwise flip."""
    a = (arg or "").strip().lower()
    if a in ("1", "on", "true", "yes"):
        return True
    if a in ("0", "off", "false", "no"):
        return False
    return not current


def _play_cue(name: str) -> None:
    """Play a named beep cue if this AudioFeedback has it; never raises."""
    try:
        fn = getattr(beep, name, None)
        if fn is not None:
            fn()
    except Exception as e:
        print(f"[Indicator] {name} failed: {type(e).__name__}: {e}")


def _start_control_server(ctl_q):
    """Start the localhost control server; None if the port is taken etc."""
    try:
        from src.voice_control.control_server import ControlServer
        srv = ControlServer(port=8770, on_immediate=_on_immediate, queue=ctl_q)
        srv.start()
        return srv
    except OSError as e:
        print(f"[Control] server not started (port 8770: {e}) — continuing without spotctl")
    except Exception as e:
        print(f"[Control] server failed: {type(e).__name__}: {e} — continuing without spotctl")
    return None


# ============================================================================
# Main Voice Control Loop
# ============================================================================
def main():
    global _mic_gated, _mic_reopen_at

    parser = argparse.ArgumentParser(description="Spot Voice Control Client")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Mic input device index (auto-detected from XVF3800)")
    parser.add_argument("--output-device", type=int, default=None, help="Speaker output device index (auto-detected from UACDemoV1.0)")
    parser.add_argument("--no-brain", action="store_true", help="Disable LLM brain (regex-only)")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--tts-backend", choices=["kokoro", "elevenlabs"], default=None,
                        help="TTS backend (overrides SPOT_TTS_BACKEND env). "
                             "kokoro=local ONNX, elevenlabs=cloud premium.")
    parser.add_argument("--no-wake-word", action="store_true", help="Always listening (skip wake word)")
    parser.add_argument("--dry-run", action="store_true", help="Skip robot dispatch (transcript+brain only; for mic-verify)")
    parser.add_argument("--debug-audio", action="store_true", help="Print audio levels for mic diagnostics")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="TTS + beep output gain (0.0-1.5, default 1.0)")
    parser.add_argument("--map", type=str, default=None,
                        help="GraphNav map path to upload during session bring-up.")
    parser.add_argument("--enable-arm", action="store_true",
                        help="Deploy the arm + enable gripper-mouth animation at startup (Stage 2E.2).")
    parser.add_argument(
        "--latency",
        choices=["off", "summary", "file", "all"],
        default="off",
        help="Latency telemetry mode: off (default), summary (in-memory only), file (write JSONL), all (both)",
    )
    parser.add_argument(
        "--latency-out",
        type=str,
        default=None,
        help="Latency JSONL file path (default: logs/latency-{timestamp}.jsonl). Only used when --latency=file or all.",
    )
    parser.add_argument("--mic-channel", type=int, default=_tuning.mic_channel,
                        help="XVF3800 input channel (0=beamformed, 1=ASR); default Tuning.mic_channel")
    parser.add_argument("--no-transcript-log", action="store_true",
                        help="Privacy: conversation + utterance logs store transcript: null")
    args = parser.parse_args()

    global MIC_CHANNEL
    MIC_CHANNEL = args.mic_channel
    if args.no_transcript_log:
        os.environ["SPOT_NO_TRANSCRIPT_LOG"] = "1"  # read by ConversationLog
    store_transcript = os.environ.get("SPOT_NO_TRANSCRIPT_LOG", "").strip().lower() not in (
        "1", "true", "yes", "on")
    utt_log = UtteranceLog(UTTERANCE_LOG_DIR, store_transcript=store_transcript)

    if args.dry_run:
        os.environ["SPOT_DRY_RUN"] = "1"
        print("[dry-run] robot dispatch disabled (transcript + brain only)")

    if args.tts_backend is not None:
        os.environ["SPOT_TTS_BACKEND"] = args.tts_backend
        print(f"[TTS] backend override: {args.tts_backend}")

    # Latency telemetry — no-op when --latency=off
    init_recorder(mode=args.latency, file_path=args.latency_out)

    # Auto-detect devices by name if not explicitly specified
    if args.device is None:
        args.device = _find_device_by_name(MIC_DEVICE_NAME, "input")
        if args.device is None:
            print(f"ERROR: Could not find mic device matching '{MIC_DEVICE_NAME}'")
            print("Run with --list-devices to see available devices, then pass --device <index>")
            return
        print(f"[Auto-detected mic: device {args.device} ({MIC_DEVICE_NAME})]")
    if args.output_device is None:
        args.output_device = _find_device_by_name(SPEAKER_DEVICE_NAME, "output")
        if args.output_device is None:
            print(f"WARNING: Could not find speaker matching '{SPEAKER_DEVICE_NAME}', using system default")
        else:
            print(f"[Auto-detected speaker: device {args.output_device} ({SPEAKER_DEVICE_NAME})]")

    if args.list_devices:
        print(sd.query_devices())
        return

    # ========================================================================
    # Startup
    # ========================================================================
    print("=" * 60)
    print("SPOT VOICE CONTROL — LLM Brain Mode")
    print("=" * 60)

    # Show device info
    if args.device is not None:
        print(f"Audio device: {args.device}")
    else:
        try:
            info = sd.query_devices(sd.default.device[0])
            print(f"Audio device: {info['name']} (default)")
        except Exception as e:
            print(f"Audio device: system default (query failed: {e})")

    # Calibrate noise floor (seeds the adaptive noise tracker)
    noise_rms = calibrate_noise_floor(NOISE_CALIBRATION_SECONDS, args.device)
    floor = FloorTracker(_tuning.floor_window_s, _tuning.floor_percentile,
                         _tuning.floor_rise_db_per_s, frame_s=FRAME_S)
    floor.seed(dbfs_from_rms(noise_rms))  # seed ONCE (seed clears the window)
    print(f"[Noise floor: {floor.floor_db:.1f} dBFS (gate: speech >= {_tuning.min_speech_dbfs:.0f} dBFS, "
          f"SNR >= {_tuning.min_snr_db:.0f} dB)]")

    # Connect to ASR
    asr_server = "localhost:50055"
    print(f"\nConnecting to ASR server at {asr_server}...")
    channel = grpc.insecure_channel(asr_server)
    stub = ASRStub(channel)

    # Initialize LLM Brain
    brain = None
    if not args.no_brain:
        print(f"\nInitializing LLM brain (model: {DEFAULT_MODEL})...")
        brain = LLMBrain(model=DEFAULT_MODEL)
        # After a reboot llama-server answers 503 while loading: wait for it
        # instead of disabling the brain for the session.
        wait = getattr(brain, "wait_available", None)
        if wait is not None:
            print("[Brain] waiting up to 90s for llama-server...")
            ready = wait(timeout_s=90.0)
        else:
            ready = brain.is_available()
        global _brain_ok
        _brain_ok = bool(ready)
        if ready:
            print(f"[Brain] Ready — model: {DEFAULT_MODEL}")
            # Warm the single brain model (gemma4:e4b handles both LLM and VLM after
            # Phase 2 collapse). Cold-start ~15-20s blocks AudioPlayer + TTS init, but
            # surfaces model-load errors before the mic opens.
            rec = get_recorder()
            ctx = rec.startup_phase("brain_warmup") if rec else nullcontext()
            with ctx:
                brain.warm_up()
        else:
            # Keep the brain object: process_utterance re-checks before each
            # utterance and switches to the LLM as soon as it answers.
            print("[Brain] llama-server not available yet — regex-only until it answers "
                  "(re-checked every utterance)")
            print("[Brain] Check: systemctl status llama-server (port 11435)")
    else:
        print("\n[Brain] Disabled (--no-brain flag). Using regex-only mode.")

    # Initialize the shared audio player BEFORE TTS / beeps. Owns the single
    # persistent OutputStream that all playback (TTS and beeps) flows through.
    # The mic-mute interlock in audio_callback reads its is_busy() state.
    global _audio_player
    print("\nInitializing audio player...")
    _audio_player = AudioPlayer(output_device=args.output_device)

    # Initialize TTS — submits playback through the shared player.
    tts = None
    if not args.no_tts:
        print("\nInitializing TTS...")
        tts = SpotTTS(player=_audio_player, volume=args.volume)
        if tts.is_available():
            print(f"[TTS] Ready (output device: {args.output_device}, "
                  f"volume={args.volume}, mic will mute during playback)")
        else:
            print("[TTS] Not available (kokoro-onnx model missing). Continuing without speech output.")
            tts = None
        # Register the live TTS instance as the module singleton so the LLM
        # set_volume action (in spot_dispatch._handle_set_volume) can find it.
        if tts is not None:
            import src.voice_control.spot_tts as _spot_tts_mod
            _spot_tts_mod._tts_instance = tts
    else:
        print("\n[TTS] Disabled (--no-tts flag)")

    # Wire the audio feedback beeps to the shared player and apply volume.
    # ``beep.device`` is kept as a no-op attribute for backwards compat —
    # device selection is now owned by the player.
    beep.set_player(_audio_player)
    beep.set_volume(args.volume)

    # Stage 2F P7: onboard RGB LED state indicator. Lazily binds to the Spot
    # AV service on first state change; no-op in dry-run or if the robot has
    # no AV system. Chimes are the fallback indicator either way.
    global _state_leds
    if os.environ.get("SPOT_DRY_RUN") == "1":
        _state_leds = None
    else:
        from src.voice_control.spot_leds import StateLeds
        import src.voice_control.spot_dispatch as _spot_dispatch

        def _led_robot():
            # Return the ALREADY-established Spot session's robot, or None.
            # Never call ensure_spot_session() here: it would connect (and
            # possibly stand) the robot synchronously on the mic-loop thread,
            # freezing voice control for ~30s — and could fire on a noise burst.
            # LEDs stay off until a real command brings the session up, then
            # activate on the next state change.
            try:
                sess = getattr(_spot_dispatch, "_spot_session", None)
                return sess["robot"] if sess else None
            except Exception:
                return None

        _state_leds = StateLeds(get_robot=_led_robot)

    # Initialize VAD (Silero via sherpa-onnx — replaces webrtcvad 2012-era model)
    vad_cfg = VadModelConfig(
        silero_vad=SileroVadModelConfig(
            model=SILERO_VAD_PATH,
            threshold=0.5,
            min_silence_duration=0.5,
            min_speech_duration=0.2,
            window_size=VAD_WINDOW_SAMPLES,
        ),
        sample_rate=SAMPLE_RATE,
        debug=False,
    )
    vad = VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=10)

    # Wake word detector (sherpa-onnx keyword spotter)
    wake_detector = None
    use_wake_word = not args.no_wake_word
    if use_wake_word:
        try:
            from src.voice_control.wake import make_wake_detector
            wake_detector = make_wake_detector()
            if wake_detector.is_available():
                print("[WakeWord] keyword spotter ready")
            else:
                print("[WakeWord] Detector not available — falling back to ASR-based")
                wake_detector = None
        except Exception as e:
            print(f"[WakeWord] Import failed: {e} — falling back to ASR-based")
            wake_detector = None
    else:
        print("[WakeWord] Disabled (--no-wake-word) — always listening")

    # Stage 2F C1: always-on safety KWS — stop/freeze/estop fire at frame level
    # in WAKE_WORD state with no wake required, even when the wake backend is
    # livekit. Degraded (not fail-closed): if it cannot load, the frame-level
    # cold-stop check is skipped and safety falls back to the VAD->ASR path
    # (which needs a wake first in WAKE_WORD state); a future task adds a
    # startup health gate to enforce this explicitly.
    safety_detector = None
    try:
        from src.voice_control.wake.safety_kws import make_safety_detector
        safety_detector = make_safety_detector()
        if not safety_detector.is_available():
            print("[SafetyKWS] UNAVAILABLE — cold stop/freeze/estop will not work "
                  "in WAKE_WORD state; run scripts/setup_safety_kws.py")
            safety_detector = None
    except Exception as e:
        print(f"[SafetyKWS] import failed: {e}")
        safety_detector = None

    # YOLO models (YOLOv8n, YOLO-World) lazy-load on first use.
    # Pre-loading them at startup starves the audio thread (CPU-bound
    # PyTorch init causes PortAudio init overflow and delays wake word).

    # Probe Spot connectivity BEFORE starting the audio loop.
    #
    # Two paths:
    #   --map present: eager full session init (stand_on_enter=True) so
    #     the map uploads as part of the SAME bring-up. This DOES power
    #     on the robot and requires e-stop to be running already. If it
    #     fails (missing SDK, unreachable, lease conflict), we auto-
    #     degrade to dry-run.
    #   no --map: light TCP probe to SPOT_IP:443 only — does NOT power
    #     on, does NOT acquire lease, does NOT require e-stop. The real
    #     session bring-up happens lazily on first dispatch_intent.
    #     If the probe fails, auto-degrade.
    #
    # Caller can opt out by passing --dry-run explicitly (skips everything).
    if not args.dry_run:
        if args.map:
            print(f"\n[Spot] Eager session init with map: {args.map}")
            try:
                from src.voice_control.spot_dispatch import ensure_spot_session
                ensure_spot_session(map_path=args.map)
                print("[Spot] Map loaded; session standing.")
            except ModuleNotFoundError as e:
                print(f"[Spot] SDK not installed ({e}); auto-enabling dry-run.")
                os.environ["SPOT_DRY_RUN"] = "1"
                args.dry_run = True
            except Exception as e:
                print(f"[Spot] Eager session init failed "
                      f"({type(e).__name__}: {e}); auto-enabling dry-run.")
                os.environ["SPOT_DRY_RUN"] = "1"
                args.dry_run = True
        else:
            import socket
            spot_host = os.environ.get("SPOT_IP", "192.168.80.3")
            try:
                print(f"\n[Spot] TCP-probing {spot_host}:443 ...")
                with socket.create_connection((spot_host, 443), timeout=2.0):
                    pass
                print("[Spot] Reachable; full session deferred to first command.")
            except OSError as e:
                print(f"[Spot] Not reachable ({type(e).__name__}: {e})")
                print("[Spot] Auto-enabling dry-run mode (voice pipeline "
                      "only; no robot dispatch, no e-stop required).")
                os.environ["SPOT_DRY_RUN"] = "1"
                args.dry_run = True

    # XVF3800 control (DoA for the speaker lock, LED ring). Optional.
    global _mic
    _mic = _open_mic_array()

    # Open audio stream (stereo for XVF3800, mono fallback, retry on busy)
    try:
        stream = _open_input_stream_with_retry(args.device, audio_callback)
        stream.start()
    except Exception as e:
        print(f"Audio error: {e}")
        print("Run with --list-devices to see available devices")
        return

    mode = f"LLM Brain ({DEFAULT_MODEL})" if brain else "Regex-only (legacy)"
    if not use_wake_word:
        ww_status = "OFF"
    elif wake_detector:
        ww_status = "ON (sherpa-onnx keyword spotter)"
    else:
        ww_status = "ON (ASR-based fallback)"
    print("\n" + "=" * 60)
    print(f"LISTENING — Mode: {mode}")
    print(f"  Wake word: {ww_status}")
    print("  Safety commands (stop/freeze/estop) always instant")
    print("  Everything else goes through the LLM brain")
    print("=" * 60 + "\n")

    beep.ready()

    # Stage 2E.2: --enable-arm deploys the arm + enables the gripper mouth at
    # startup (reuses the safety-gated enable_mouth action, which forces session
    # bring-up + power/estop checks). Skipped in dry-run.
    if getattr(args, "enable_arm", False) and not args.dry_run:
        try:
            from src.voice_control.spot_dispatch import dispatch_intent
            print("[client_mic] --enable-arm: deploying arm + enabling mouth...")
            dispatch_intent({"intent": "enable_mouth"})
        except Exception as e:
            print(f"[client_mic] --enable-arm failed: {e}")

    # ========================================================================
    # Main Loop State
    # ========================================================================
    window = b""
    window_t = time.monotonic()
    frame_count = 0
    last_vad_error_log_time = 0.0  # Token-bucket guard for VAD failure logs (one/min max)
    from collections import deque
    # Pre-roll: appended on EVERY frame in every state (A16). Each entry is
    # (t_monotonic, pcm16, dbfs, is_speech). Sized to cover the onset frames
    # plus Silero's min_speech_duration lag so first words aren't clipped.
    preroll_buffer = deque(maxlen=PREROLL_FRAMES + _tuning.onset_frames)

    # Front-end (Task 10): VAD on every frame -> Endpointer; FloorTracker
    # seeded ONCE from calibration; SNR gate at utterance end.
    ep = _new_endpointer()
    in_utt = False
    utt = []                       # entries as in preroll_buffer
    stats = None
    end_sil_s = _tuning.end_silence_s_short
    last_idle_log = 0.0

    # Stage 2F P0a: barge-in reset bookkeeping. `player_was_busy` tracks the
    # AudioPlayer busy edge across iterations; `response_pending` is armed only
    # after a processed utterance (TTS response), never after the wake beep,
    # so the in-breath command audio kept on a wake fire is not drained.
    player_was_busy = False
    response_pending = False

    # Stage 2F P6: True once the current turn has announced THINKING — i.e. a
    # real transcript reached the brain (past STT + the safety/wake early-returns).
    # Gates the RESPONDING indicator so a noise/empty-STT turn stays silent.
    gate_announced = False

    def _announce_thinking():
        # Passed into process_utterance; fired the moment a real transcript is
        # dispatched to the brain, so THINKING never chimes on noise/empty STT.
        nonlocal gate_announced
        gate_announced = True
        _enter_state(VoiceState.THINKING)

    # Wake word state
    state = VoiceState.WAKE_WORD if use_wake_word else VoiceState.LISTENING
    # Conversation clock (monotonic): fallback timeout when no direction lock.
    listening_since = time.monotonic()
    _enter_state(state)

    def _on_wake(t_wake):
        """Wake fired (KWS frame at t_wake, or ASR fallback): lock onto the
        speaker's direction, then enter LISTENING (A10). A wake while already
        locked re-locks to the new speaker."""
        nonlocal state, listening_since
        now_m = time.monotonic()
        degs = _mic_speech_degs(t_wake - 1.5, t_wake)
        ang = _lock.lock(degs, now_m)
        if ang is not None:
            print(f"[Lock] speaker at {ang:.0f} deg (mic), "
                  f"{_body_deg(ang):.0f} deg (body)")
            if _tuning.beam_steering and _mic_usable():
                _mic.set_fixed_beams(math.radians(ang))
        else:
            print("[Lock] no direction readings — accepting all directions")
        _status["locked_deg"] = _body_deg(ang)
        state = VoiceState.LISTENING
        listening_since = now_m
        _enter_state(VoiceState.LISTENING)

    def _release(cue: bool):
        """Drop the speaker lock + fixed beams and go back to WAKE_WORD."""
        nonlocal state, ep, in_utt, utt
        ep = _new_endpointer()  # drop any half-recorded utterance
        in_utt = False
        utt = []
        _lock.release()
        _status["locked_deg"] = None
        if _mic_usable():
            _mic.set_fixed_beams(None)
        state = VoiceState.WAKE_WORD
        if wake_detector:
            wake_detector.reset()
        _enter_state(VoiceState.WAKE_WORD)
        if cue:
            _play_cue("released_cue")

    def _finish_utterance(endpoint):
        """Endpoint reached: SNR gate, lock check, then ASR/brain."""
        nonlocal state, listening_since, response_pending, gate_announced
        global _mic_gated
        floor_db = floor.floor_db
        t0 = utt[0][0]
        t1 = last_speech_t(utt)
        degs = _mic_speech_degs(t0, t1)
        record = dict(
            endpoint=endpoint, end_silence_s=end_sil_s,
            duration_s=len(utt) * FRAME_S, floor_db=floor_db, stats=stats,
            lock_mic_deg=_lock.angle, utt_mic_deg=circular_median_deg(degs))

        min_db = _tuning.min_speech_dbfs + (NAV_SPEECH_DB_BOOST if _is_robot_moving() else 0.0)
        if not passes_gate(stats, floor_db, min_db, _tuning.min_snr_db):
            print(f"[Gate] dropped: speech {stats.speech_db_median:.1f} dBFS, "
                  f"SNR {stats.snr_db(floor_db):.1f} dB (need >= {min_db:.0f} dBFS, "
                  f">= {_tuning.min_snr_db:.0f} dB)")
            utt_log.write(build_utterance_record(**record, accepted=False,
                                                 drop_reason="gate"))
            return
        if not _lock.accept(degs[-LOCK_ACCEPT_MAX_SAMPLES:], time.monotonic()):
            print(f"[Lock] rejected utterance from {record['utt_mic_deg']:.0f} deg "
                  f"(locked {_lock.angle:.0f} deg)")
            _signal_reject()
            utt_log.write(build_utterance_record(**record, accepted=False,
                                                 drop_reason="lock"))
            return
        listening_since = time.monotonic()
        _update_ring()  # accept() may have nudged the lock angle

        print(f"\n>>> Processing ({endpoint})...")
        safety_only = (state == VoiceState.WAKE_WORD)
        if not safety_only:
            # Commit: mute the mic now (gate at STT start), but defer the
            # THINKING indicator until process_utterance confirms real words
            # bound for the LLM (via on_thinking).
            _mic_gated = True
            gate_announced = False
        info = {}
        t_proc = time.monotonic()
        audio = [pcm16_to_float32(f) for _, f, _, _ in trim_trailing(utt, TRAILING_KEEP_FRAMES)]
        try:
            result = process_utterance(stub, None, audio, brain,
                                       safety_only=safety_only, tts=tts,
                                       has_wake_detector=bool(wake_detector),
                                       on_thinking=(None if safety_only else _announce_thinking),
                                       info=info)
        except Exception as e:
            # A per-utterance failure (ASR/LLM/TTS) must drop the turn, not
            # crash voice control. result=None routes to the idle-based gate
            # release so the mic reopens.
            print(f"[Error] utterance processing failed: {type(e).__name__}: {e}")
            result = None
        latency = time.monotonic() - t_proc
        _status["last_latency_s"] = round(latency, 2)
        if info.get("transcript"):
            _status["last_transcript"] = info["transcript"] if store_transcript else "<not logged>"
        utt_log.write(build_utterance_record(
            **record, accepted=True, transcript=info.get("transcript"),
            reached_llm=bool(info.get("reached_llm")), latency_s=latency))

        if result != "wake_detected" and not safety_only and gate_announced:
            response_pending = True
            _enter_state(VoiceState.RESPONDING)
            state = VoiceState.RESPONDING
        elif result == "wake_detected" and state == VoiceState.WAKE_WORD:
            print(">>> Now listening for commands...")
            _on_wake(t1)
            _drain_audio_queue()
        elif use_wake_word:
            # safety-only / ignored utterance: keep prior behavior
            _drain_audio_queue()

    # Operator control server (spotctl). Immediate commands run on its HTTP
    # threads via _on_immediate; queued ones are drained here between frames.
    from src.voice_control.control_server import drain as _drain_control
    ctl_q = queue.Queue()
    ctl_srv = _start_control_server(ctl_q)

    def _on_queued(name, arg):
        nonlocal state, listening_since
        global _crowd_mode, _voiceprint_mode
        if name == "listen":
            # Open the conversation with no direction lock (accept all).
            _lock.release()
            _status["locked_deg"] = None
            state = VoiceState.LISTENING
            listening_since = time.monotonic()
            _enter_state(VoiceState.LISTENING)
        elif name in ("release", "reset"):
            _release(cue=True)
            if name == "reset" and brain is not None:
                brain.history.clear()
        elif name == "say":
            if tts is not None and arg:
                tts.speak(arg)
        elif name == "persona":
            if brain is None:
                print("[Control] persona: no brain")
            else:
                from src.voice_control.spot_dispatch import do_set_persona
                print(f"[Control] persona -> {do_set_persona({'name': arg}, brain)}")
        elif name == "crowd":
            _crowd_mode = _toggle(_crowd_mode, arg)
            print(f"[Control] crowd mode {'ON' if _crowd_mode else 'OFF'}")
        elif name == "voiceprint":
            _voiceprint_mode = _toggle(_voiceprint_mode, arg)
            print(f"[Control] voiceprint flag {'ON' if _voiceprint_mode else 'OFF'}")

    try:
        while True:
            _drain_control(ctl_q, _on_queued)
            _status["state"] = state.name
            _check_mic_alive()
            _flush_ring()

            # Stage 2F P0a: on the player busy->idle edge after a TTS response,
            # reset streaming-detector state so the speaker tail + stale VAD/KWS
            # decisions can't corrupt the next detection. Never touches `state`.
            busy = _speech_busy()  # chimes don't hold the gate or count as a response
            if should_reset_after_player(response_pending, player_was_busy, busy):
                vad.reset()
                if wake_detector:
                    wake_detector.reset()
                preroll_buffer.clear()
                ep = _new_endpointer()
                in_utt = False
                utt = []
                _drain_audio_queue(keep_tail=False)  # full drain: no late phantom commands
                response_pending = False
                print("[Mic] detector reset after TTS response")
            player_was_busy = busy

            # Stage 2F P6 gate release (IDLE-based, not TTS-edge-based, so a turn
            # that plays no audio — e.g. a noise burst with an empty ASR result —
            # can't wedge the gate shut). Once gated and the player is idle, start
            # the settle; reopen when it elapses.
            if _mic_gated:
                now_m = time.monotonic()
                if busy:
                    _mic_reopen_at = None  # still speaking — hold the gate
                elif should_arm_reopen(_mic_gated, _mic_reopen_at, busy):
                    _mic_reopen_at = now_m + _tuning.reopen_delay_s
                elif should_reopen_mic(_mic_gated, _mic_reopen_at, now_m):
                    _mic_gated = False
                    _mic_reopen_at = None
                    response_pending = False
                    vad.reset()
                    if wake_detector:
                        wake_detector.reset()
                    # Keep no settle-window audio; pre-roll restarts on the
                    # very next frame so a quick reply's first word is kept.
                    _drain_audio_queue(keep_tail=False)
                    preroll_buffer.clear()
                    ep = _new_endpointer()
                    in_utt = False
                    utt = []
                    gate_announced = False
                    if _crowd_mode and use_wake_word:
                        # Crowd mode: every turn needs a fresh wake (A11/10).
                        _release(cue=False)
                        print(">>> Crowd mode — say the wake word for the next turn")
                    else:
                        state = VoiceState.LISTENING
                        listening_since = now_m
                        _lock.touch(now_m)  # conversation clock restarts at reopen
                        # Always cue: the listen chime fires as the mic opens
                        # (beeps no longer mute the mic, so nothing is lost).
                        _enter_state(VoiceState.LISTENING)
                        print(">>> Listening for follow-up (no wake needed)...")
            # Get audio from queue. Short timeout so we still cycle (and stay
            # responsive to SIGTERM / KeyboardInterrupt / spotctl) even when the
            # audio callback has stopped putting frames in the queue.
            try:
                window_t, pcm = audio_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            window += pcm

            # Process complete frames
            while len(window) >= BYTES_PER_FRAME:
                frame = window[:BYTES_PER_FRAME]
                window = window[BYTES_PER_FRAME:]
                t_frame = window_t  # callback blocksize == 1 frame, so exact
                frame_count += 1

                # ---- per-frame front end: dBFS + VAD on EVERY frame ----
                db = frame_dbfs(frame)
                try:
                    vad.accept_waveform(pcm16_to_float32(frame))
                    is_speech = vad.is_speech_detected()
                    while not vad.empty():  # A15: don't let segments pile up
                        vad.pop()
                except Exception as e:
                    if time.time() - last_vad_error_log_time > 60.0:
                        print(f"[VAD] accept_waveform failed: {type(e).__name__}: {e}")
                        last_vad_error_log_time = time.time()
                    is_speech = False
                floor.update(db, is_speech)
                _status["floor_dbfs"] = round(floor.floor_db, 1)
                preroll_buffer.append((t_frame, frame, db, is_speech))

                # Stage 2F C1: always-on safety check (cold stop/freeze/estop).
                # Runs in WAKE_WORD state before the wake early-continue so a
                # safety command halts the robot with no prior wake.
                if state == VoiceState.WAKE_WORD and safety_detector:
                    kw = safety_detector.process_frame(frame)
                    if kw:
                        intent = check_safety_command(kw)
                        if intent:
                            print(f"[SAFETY-KWS] '{kw}' — executing immediately")
                            _abort_chain.set()
                            if execute_on_spot(intent):
                                beep.command_ok()
                            else:
                                beep.error()
                            safety_detector.reset()
                            continue

                # Wake word (dedicated detector). Runs in LISTENING too, so a
                # new speaker's wake re-locks onto them (Task 10 change 9).
                if wake_detector and state in (VoiceState.WAKE_WORD, VoiceState.LISTENING):
                    if wake_detector.process_frame(frame):
                        print(">>> Wake word detected!")
                        was_idle = state == VoiceState.WAKE_WORD
                        _on_wake(t_frame)
                        if was_idle:
                            # Fresh endpointer; the in-breath command ("Hey Spot
                            # stand up") is still arriving and pre-roll has the tail.
                            ep = _new_endpointer()
                            in_utt = False
                            utt = []

                # In WAKE_WORD mode with a dedicated detector, don't record —
                # the keyword spotter handles wake at frame level (no ASR on
                # crowd noise).
                if state == VoiceState.WAKE_WORD and wake_detector:
                    continue
                if state not in (VoiceState.WAKE_WORD, VoiceState.LISTENING):
                    continue

                # Conversation timeout (A11): lock.expired replaces the old
                # fixed LISTENING_TIMEOUT; wall-clock fallback when unlocked.
                if (state == VoiceState.LISTENING and not in_utt and use_wake_word
                        and conversation_expired(_lock, listening_since, time.monotonic(),
                                                 _tuning.lock_timeout_s)):
                    print("[Conversation timeout — say 'Hey Spot' to activate]")
                    _release(cue=True)
                    continue

                ev = ep.feed(is_speech, db, floor.floor_db)
                if ev == "start":
                    in_utt = True
                    utt = list(preroll_buffer)  # includes this frame
                    stats = UtteranceStats(frame_s=FRAME_S)
                    for _, _, d, s in utt:
                        stats.add(d, s)
                    end_sil_s = _tuning.end_silence_s_short
                    ep.set_end_silence_frames(_frames(end_sil_s))
                    if state == VoiceState.WAKE_WORD:
                        print("\n>>> Speech detected (wake word mode — safety only)...")
                    else:
                        print("\n>>> Speech detected...")
                elif in_utt:
                    utt.append((t_frame, frame, db, is_speech))
                    stats.add(db, is_speech)
                    dur = len(utt) * FRAME_S
                    if dur >= SILENCE_CROSSOVER and end_sil_s != _tuning.end_silence_s:
                        # Longer utterances (chains) get the longer end silence.
                        end_sil_s = _tuning.end_silence_s
                        ep.set_end_silence_frames(_frames(end_sil_s))
                    if len(utt) % 17 == 0:  # ~every 0.5s of audio
                        print(f"    Recording: {dur:.1f}s")

                if ev in ("end", "cap") and in_utt:
                    in_utt = False
                    if ev == "cap":
                        print(f"\n>>> Max duration reached ({len(utt) * FRAME_S:.1f}s)")
                    _finish_utterance(ev)
                    utt = []
                    stats = None

                # Idle metrics line (1 Hz) + periodic console status
                now_m = time.monotonic()
                if not in_utt and now_m - last_idle_log >= 1.0:
                    last_idle_log = now_m
                    utt_log.write(build_idle_record(state.name, floor.floor_db, db, _lock.angle))
                if not in_utt and frame_count % 166 == 0:  # ~5 seconds
                    if args.debug_audio:
                        print(f"[{state.name} dbfs={db:.1f} floor={floor.floor_db:.1f}]")
                    elif state == VoiceState.WAKE_WORD:
                        print("[Say 'Hey Spot' to activate...]")
                    else:
                        print("[Listening...]")

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        if ctl_srv is not None:
            try:
                ctl_srv.stop()
            except Exception:
                pass
        try:
            stream.stop()
            time.sleep(0.1)  # let PortAudio drain buffers before close
            stream.close()
        except Exception:
            pass
        cleanup_spot()


def partition_drain(frames, keep_count):
    """Split drained frames into (kept, dropped_count).

    keep_count > 0 keeps the most recent `keep_count` frames (used after a
    wake fire so an in-breath command isn't lost); keep_count == 0 keeps
    nothing (used after a response so post-decision speech can't replay).
    """
    if not frames:
        return [], 0
    if keep_count <= 0:
        return [], len(frames)
    dropped = max(0, len(frames) - keep_count)
    return frames[dropped:], dropped


def _drain_audio_queue(keep_tail: bool = True):
    """Discard stale audio frames accumulated during processing.

    keep_tail=True keeps the most recent ~1s (a wake fire's in-breath command
    could be in there). keep_tail=False drains everything — used after a
    response so speech uttered while Spot was thinking/speaking is never
    replayed as a phantom command.
    """
    frames = []
    while not audio_queue.empty():
        try:
            frames.append(audio_queue.get_nowait())
        except queue.Empty:
            break

    keep_count = max(1000 // FRAME_MS, 1) if keep_tail else 0  # ~33 frames at 30ms
    kept, dropped = partition_drain(frames, keep_count)

    for frame in kept:
        audio_queue.put(frame)

    if dropped:
        print(f"[Drained {dropped} stale audio chunks, kept {len(kept)}]")


THINKING_FILLER_LINE = "Hmm, let me think."
_PERSONA_INTENTS = ("set_persona", "add_persona")
_PERSONA_ACTION_RE = re.compile(r'"action"\s*:\s*"(?:set_persona|add_persona)"')
_RESPONSE_KEY_RE = re.compile(r'"response"\s*:\s*"')


class ThinkingFiller:
    """B12: speak one filler line if the brain is still silent after
    `delay_s`. `cancel()` is idempotent; the line never plays over speech."""

    def __init__(self, speak, delay_s, is_busy, timer_factory=None):
        self._speak, self._is_busy = speak, is_busy
        self._cancelled = threading.Event()
        make = timer_factory or threading.Timer
        self._timer = make(delay_s, self._fire)
        self._timer.daemon = True

    def start(self):
        self._timer.start()

    def cancel(self):
        self._cancelled.set()
        try:
            self._timer.cancel()
        except Exception:
            pass

    def _fire(self):
        if self._cancelled.is_set() or self._is_busy():
            return
        self._cancelled.set()
        try:
            self._speak()
        except Exception as e:
            print(f"[Filler] failed: {e}")


class _TokenWatch:
    """Brain token callback wrapper around the streaming TTSChunker.

    GBNF emits actions before "response", so when the response key arrives
    we know the action list: a persona switch suppresses streamed speech
    (B6 — spoken after the switch in the new voice), and the thinking filler
    is cancelled as soon as the reply text starts (B12)."""

    def __init__(self, chunker, filler=None):
        self._chunker, self._filler = chunker, filler
        self._head = []
        self._seen_response = False

    def __call__(self, delta: str) -> None:
        if not self._seen_response:
            self._head.append(delta)
            head = "".join(self._head)
            if _RESPONSE_KEY_RE.search(head):
                self._seen_response = True
                if _PERSONA_ACTION_RE.search(head):
                    self._chunker.suppressed = True
                if self._filler is not None:
                    self._filler.cancel()
        self._chunker.accept(delta)


_brain_ok = False


def _brain_ready(brain) -> bool:
    """Positive availability is cached; a not-yet-ready llama-server (503
    while loading after a reboot) is re-checked on each utterance instead
    of disabling the brain for the whole session."""
    global _brain_ok
    if _brain_ok:
        return True
    try:
        _brain_ok = bool(brain.is_available())
    except Exception:
        _brain_ok = False
    if _brain_ok:
        print("[Brain] llama-server is up — LLM brain active")
    else:
        print("[Brain] llama-server still not ready — regex fallback for this utterance")
    return _brain_ok


def _resolve_voice(brain):
    """Voice ID for the brain's current persona on the active TTS backend."""
    from src.voice_control.tts import get_active_backend_name
    persona = get_persona(brain.session_state.current_persona, brain._persona_registry)
    return voice_id_for(persona, get_active_backend_name()) or None


MIN_SPEECH_DURATION = 0.3  # Reject utterances shorter than this (catches noise bursts, real words are 0.3s+)


def process_utterance(stub, speech_buffer: bytearray, speech_float_buffer: list,
                      brain=None, safety_only=False, tts=None,
                      has_wake_detector=False, on_thinking=None, info=None):
    """Process recorded speech through ASR then LLM brain (or regex fallback).

    Args:
        safety_only: If True, only execute safety commands (stop/freeze/estop).
                     Non-safety speech is discarded. Used when wake word not detected.
        tts: SpotTTS instance for spoken responses (None = no speech output).
        has_wake_detector: If True, dedicated wake word detector is active —
                          skip ASR-based wake phrase detection (detector handles it).
        info: optional dict filled with "transcript" and "reached_llm" for
              the utterance metrics line.

    Returns:
        "wake_detected" if a wake phrase was found (ASR fallback only), None otherwise.

    Flow:
        Audio → Whisper ASR → transcript
        transcript → safety check (instant regex for stop/estop/freeze)
        transcript → wake phrase check (ASR fallback, only if no dedicated detector)
        transcript → LLM brain (state + history → action + response)
        OR (legacy) → regex parser → intent → dispatch
    """
    if not speech_float_buffer:
        return None

    # New turn: forget earlier operator stops. A stop arriving from here on
    # (during ASR, LLM or the chain itself) aborts this turn's action chain.
    _abort_chain.clear()
    t_utterance_start = time.time()

    # Combine audio
    audio_float = np.concatenate(speech_float_buffer)
    duration = len(audio_float) / SAMPLE_RATE

    # Reject very short clips (noise bursts, not speech)
    if duration < MIN_SPEECH_DURATION:
        print(f"[Too short ({duration:.2f}s) — skipping]")
        return None

    # Convert to PCM16 and send to ASR
    pcm_bytes = float32_to_pcm16(audio_float)
    print(f"[Sending {duration:.1f}s to ASR...]")

    t_asr_start = time.time()
    transcript = send_to_asr(stub, pcm_bytes)
    t_asr_end = time.time()
    print(f"[Timing] ASR: {t_asr_end - t_asr_start:.2f}s")

    if info is None:
        info = {}
    if not transcript:
        print("[No speech recognized]")
        return None
    info["transcript"] = transcript

    print("\n" + "=" * 60)
    print(f"HEARD: \"{transcript}\"")
    print("=" * 60)

    # Clean transcript (remove Whisper punctuation artifacts)
    clean = re.sub(r'\.\s*', ' ', transcript).strip()
    clean = re.sub(r'\s+', ' ', clean)
    if clean != transcript:
        print(f"CLEAN: \"{clean}\"")

    # Reject empty or punctuation-only transcripts (Whisper hallucination on noise)
    if not clean or len(clean) < 2:
        print("[Empty transcript — likely noise, skipping]")
        return None

    # ------------------------------------------------------------------
    # 1. Safety fast-path: stop/freeze/estop bypass LLM (zero latency)
    #    Safety commands ALWAYS execute, regardless of wake word state.
    # ------------------------------------------------------------------
    if _run_safety(clean):
        return None

    # ------------------------------------------------------------------
    # 1b. Wake phrase detection (ASR fallback — only when no dedicated detector)
    # ------------------------------------------------------------------
    wake_activated = False
    if safety_only:
        if has_wake_detector:
            # Dedicated detector handles wake word at frame level.
            # If we're here with safety_only=True, it means VAD triggered
            # but it wasn't a safety command — just discard.
            print(f"[Ignored \"{clean}\" — waiting for 'Hey Spot' (detector)]")
            return None
        # ASR-based fallback: check transcript for wake phrase
        wake_match = WAKE_PHRASE_PATTERN.search(clean)
        if wake_match:
            # Strip wake phrase from transcript
            remainder = clean[wake_match.end():].strip()
            remainder = re.sub(r'^[,\s]+', '', remainder)  # strip leading punctuation
            print(f"[Wake] Detected in: \"{clean}\"")
            if remainder and len(remainder) >= 2:
                # Command follows wake phrase — process it now ("Hey Spot stand")
                print(f"[Wake] Command: \"{remainder}\"")
                clean = remainder
                if _run_safety(clean):  # "Hey Spot, stop ..." — re-check after strip
                    return None
                safety_only = False  # Allow full processing below
                wake_activated = True
            else:
                # Just the wake phrase, no command — wait for next utterance
                print("[Wake] Activated — waiting for command...")
                return "wake_detected"
        else:
            # No wake phrase, no safety command — discard
            print(f"[Ignored \"{clean}\" — say 'Hey Spot' first]")
            return None

    # ------------------------------------------------------------------
    # 1c. Strip wake phrase from transcript (dedicated detector mode)
    #     Handles "Hey Spot stand up" → "stand up" when said in one breath
    # ------------------------------------------------------------------
    if has_wake_detector and not safety_only:
        wake_match = WAKE_PHRASE_PATTERN.search(clean)
        if wake_match:
            remainder = clean[wake_match.end():].strip()
            remainder = re.sub(r'^[,\s]+', '', remainder)
            if remainder and len(remainder) >= 2:
                print(f"[Wake strip] \"{clean}\" → \"{remainder}\"")
                clean = remainder
                if _run_safety(clean):  # re-check after strip, before LLM/regex
                    return None
            else:
                # Just the wake phrase with no command — ignore
                print(f"[Wake phrase only — no command after stripping]")
                return None

    # Stage 2F P6: the utterance survived STT (real words, not a safety command,
    # not a bare wake phrase) and is about to be dispatched to the brain — signal
    # THINKING now (not before STT) so noise/empty transcripts never chime.
    # B11: "uh um hmm" — nothing worth a turn.
    if is_filler_only(clean):
        print(f"[Filler only \"{clean}\" — skipping]")
        return None

    info["reached_llm"] = True
    if on_thinking is not None:
        on_thinking()

    rec = get_recorder()
    trace = rec.begin_utterance() if rec else None
    try:
        # ------------------------------------------------------------------
        # 2. LLM Brain mode (primary)
        # ------------------------------------------------------------------
        if brain is not None and _brain_ready(brain):
            # Collect current robot state for context
            state = get_spot_state()

            # Stage 2E.1: resolve the persona-mapped voice ID for this turn.
            # B6: if the reply switches persona, the streamed speech is held
            # back (see _TokenWatch), the switch runs first, and the reply is
            # spoken in the NEW voice.
            voice_id = _resolve_voice(brain)

            # Streaming TTS: chunker emits response sentences as they arrive.
            # Suppresses itself if a describe action appears in the action list
            # (GBNF emits actions before response), so the LLM's hallucinated
            # camera description never reaches the speaker — client_mic.py
            # speaks a stock ack instead (see describe_action branch below).
            chunker = tts.enqueue_streaming(voice=voice_id) if tts and tts.is_available() else None
            filler = None
            if chunker is not None:
                # B12: one neutral line if nothing has been said after
                # Tuning.thinking_filler_s; cancelled when the reply starts.
                filler = ThinkingFiller(lambda: tts.speak(THINKING_FILLER_LINE, voice=voice_id),
                                        _tuning.thinking_filler_s, _speech_busy)
                brain.on_token_callback = _TokenWatch(chunker, filler)
                filler.start()

            t_llm_start = time.time()
            try:
                result = brain.process(clean, state)
            except Exception as e:
                # A failed LLM call (server crash/restart → ConnectionError,
                # slot contention → ReadTimeout, 5xx → HTTPError) must NOT kill
                # the voice process. Bail this utterance and keep listening.
                import traceback
                print(f"[Brain] process() failed: {type(e).__name__}: {e}")
                traceback.print_exc()
                try:
                    beep.error()
                except Exception:
                    pass
                if tts is not None:
                    try:
                        tts.speak("Sorry, I lost my train of thought. Could you say that again?")
                    except Exception:
                        pass
                return None
            finally:
                if filler is not None:
                    filler.cancel()
                if chunker is not None:
                    chunker.flush()
                    brain.on_token_callback = None
            t_llm_end = time.time()
            print(f"[Timing] LLM: {t_llm_end - t_llm_start:.2f}s")

            response = result.get("response", "")
            actions = result.get("actions", [])

            # Backward compat: old single "action" field
            if not actions and result.get("action"):
                actions = [result["action"]]

            # If a describe action is queued, the LLM's "response" field is
            # often a hallucinated answer (the model can't actually see the
            # camera). Replace it with a stock acknowledgment so we don't
            # speak a fake description before the VLM provides the real one.
            # Note: this only suppresses what gets spoken — the raw LLM JSON
            # is still stored in brain.history, so the model can still see
            # its own hallucination on the next turn (acceptable footnote).
            describe_action = next(
                (a for a in actions if a.get("intent") == "describe"),
                None,
            )
            if describe_action:
                query = str(describe_action.get("params", {}).get("query", "")).replace("_", " ").strip()
                response = (
                    f"Let me take a look for the {query}..."
                    if query
                    else random.choice(_DESCRIBE_ACKS)
                )

            # B6: run persona switches BEFORE speaking so the announcing reply
            # uses the new voice (the chunker held it back — _TokenWatch).
            persona_actions = [a for a in actions if a.get("intent") in _PERSONA_INTENTS]
            if persona_actions:
                for a in persona_actions:
                    execute_on_spot(a, brain=brain)
                actions = [a for a in actions if a.get("intent") not in _PERSONA_INTENTS]
                voice_id = _resolve_voice(brain)
                if tts and response:
                    tts.speak(response, voice=voice_id)
                    response = ""  # spoken; don't repeat below

            # Show what the robot "says"
            if response:
                print(f"\nSPOT (LLM): \"{response}\"")

            # Track which models actually handled this utterance for the
            # end-of-turn [Models] summary line. vlm_used flips inside the
            # describe-action branch below.
            vlm_used = False
            vlm_error = False

            if actions:
                beep.command_ok()

                # Streaming chunker already spoke response sentence-by-sentence.
                # Only fall back to tts.speak when chunker suppressed (describe
                # path: `response` is now a stock ack) or chunker absent.
                chunker_played = chunker is not None and not chunker.suppressed
                if tts and response and not chunker_played:
                    tts.speak(response, voice=voice_id)

                if trace:
                    trace.mark("intent_dispatch")
                for i, intent in enumerate(actions):
                    if _abort_chain.is_set():
                        print("[Chain] aborted by operator stop/hush")
                        break
                    cmd = intent["intent"]
                    params = intent.get("params", {})

                    if len(actions) > 1:
                        print(f"[Chain] Executing {i+1}/{len(actions)}: {cmd}")
                        if i > 0:
                            beep.chain_next()

                    # Special VLM flow for "describe" action
                    if cmd == "describe":
                        vlm_used = True
                        camera = params.get("camera", "front")
                        query = params.get("query", "")
                        try:
                            from src.voice_control.spot_dispatch import capture_frame
                            image_bytes = capture_frame(camera)
                            if image_bytes:
                                # If asking about a specific object, run YOLO first
                                yolo_hint = ""
                                if query:
                                    try:
                                        from src.voice_control.visual_nav import detect_in_image
                                        det = detect_in_image(image_bytes, query)
                                        if det["found"]:
                                            yolo_hint = (
                                                f"IMPORTANT: Object detection confirms a '{query}' "
                                                f"IS visible in this image ({det['position']}, "
                                                f"confidence {det['confidence']:.0%}). "
                                                f"Describe it and its surroundings."
                                            )
                                            print(f"[YOLO] Found '{query}' — {det['position']}, "
                                                  f"conf={det['confidence']:.2f}")
                                        else:
                                            yolo_hint = (
                                                f"NOTE: Object detection did NOT find '{query}' "
                                                f"in this image. If you also don't see it, say so."
                                            )
                                            print(f"[YOLO] '{query}' not detected")
                                    except ImportError:
                                        pass
                                with (trace.span("vlm") if trace else nullcontext()):
                                    vlm_response = brain.query_vlm(image_bytes, clean, yolo_hint=yolo_hint)
                                print(f"\nSPOT (VLM): \"{vlm_response}\"")
                                if tts:
                                    tts.wait()
                                    tts.speak(vlm_response, voice=voice_id)
                            else:
                                beep.error()
                                vlm_error = True
                                fallback = "Sorry, I couldn't capture an image right now."
                                print(f"\nSPOT (VLM, error): \"{fallback}\"")
                                if tts:
                                    tts.wait()
                                    tts.speak(fallback, voice=voice_id)
                        except Exception as e:
                            beep.error()
                            vlm_error = True
                            print(f"[VLM] Error: {e}")
                            fallback = "Sorry, my vision system isn't working right now."
                            print(f"\nSPOT (VLM, error): \"{fallback}\"")
                            if tts:
                                tts.wait()
                                tts.speak(fallback, voice=voice_id)
                    else:
                        if execute_on_spot(intent, brain=brain,
                                           speak=(tts.speak if tts else None)):
                            print(f">>> SUCCESS" if len(actions) == 1 else f">>> {cmd} SUCCESS")
                        else:
                            beep.error()
                            print(f">>> FAILED" if len(actions) == 1 else f">>> {cmd} FAILED — stopping chain")
                            break

                        # Wait for movement/posture to complete before next action in chain
                        if len(actions) > 1 and i < len(actions) - 1:
                            if cmd in ("go_to", "go_to_object", "follow_me", "tour", "patrol", "come_back"):
                                _wait_for_nav_complete()
                            elif cmd in ("walk", "strafe", "turn"):
                                _abort_chain.wait(CHAIN_MOVEMENT_WAIT_S)
                            elif cmd in ("sit", "stand", "selfright", "body_height"):
                                _abort_chain.wait(CHAIN_POSTURE_WAIT_S)
                if trace:
                    trace.mark("dispatch_complete")
            else:
                # Conversation only — chunker already streamed sentences; speak
                # only if no chunker (legacy) or chunker suppressed.
                chunker_played = chunker is not None and not chunker.suppressed
                if tts and response and not chunker_played:
                    tts.speak(response, voice=voice_id)
                print("(No physical action — conversation only)")

            # One-line summary of which model(s) actually handled this utterance.
            # Makes it obvious when a "describe surroundings" prompt got
            # hallucinated by the LLM instead of routed to the VLM.
            if vlm_used and not vlm_error:
                models_tag = "LLM + VLM"
            elif vlm_used and vlm_error:
                models_tag = "LLM + VLM (failed → fell back)"
            elif actions:
                models_tag = f"LLM only (action: {actions[0]['intent']})"
            else:
                models_tag = "LLM only (conversation)"
            print(f"[Models] {models_tag}")

            print(f"[Timing] Total: {time.time() - t_utterance_start:.2f}s")
            return "wake_detected" if wake_activated else None

        # ------------------------------------------------------------------
        # 3. Legacy regex-only fallback (--no-brain mode)
        # ------------------------------------------------------------------
        intent = parse_intent(clean)

        if intent:
            cmd = intent['intent']
            params = intent.get('params', {})
            print(f"Command: {cmd}" + (f" {params}" if params else ""))

            if execute_on_spot(intent):
                print(">>> SUCCESS")
            else:
                print(">>> FAILED")
        else:
            print("(Not recognized — try: stand, sit, stop, turn left/right, go to [location])")

        print(f"[Timing] Total: {time.time() - t_utterance_start:.2f}s")
        return "wake_detected" if wake_activated else None
    finally:
        if rec is not None and trace is not None:
            if tts is not None:
                try:
                    tts.wait()  # drain TTS worker so render/play marks land before serialize
                except Exception:
                    pass
            rec.complete(trace)


if __name__ == "__main__":
    main()
