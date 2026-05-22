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

# Add parent and project root to path for local imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import re
import glob
import time
import queue
import random
import signal
import argparse
import subprocess
import threading
from enum import Enum, auto
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

class VoiceState(Enum):
    WAKE_WORD = auto()   # Waiting for "hey spot" (detected via ASR, not a separate model)
    LISTENING = auto()   # Wake word heard, waiting for speech
    RECORDING = auto()   # Speech detected, accumulating audio


LISTENING_TIMEOUT = 15.0  # seconds before requiring wake word again

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples per frame
BYTES_PER_FRAME = FRAME_SAMPLES * 2  # int16 = 2 bytes

SILERO_VAD_PATH = "/mnt/ssd/vad-models/silero_vad.onnx"
VAD_WINDOW_SAMPLES = 512         # Silero default @ 16 kHz = 32 ms; VAD buffers internally
SILENCE_TIMEOUT_SHORT = 0.8      # Silence timeout for short utterances (< 2s)
SILENCE_TIMEOUT_LONG = 1.5       # Silence timeout for longer utterances (> 2s, e.g. chained commands)
SILENCE_CROSSOVER = 2.0          # Switch from short to long timeout after this much speech (seconds)
MAX_UTTERANCE_SECONDS = 20       # Force-send after this duration (up from 8 — allows long chains)
MAX_UTTERANCE_FRAMES = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)
SPEECH_ONSET_FRAMES = 3          # Consecutive VAD+energy frames to confirm speech onset (~90ms)
NAV_ONSET_FRAMES = 5             # Higher onset threshold during navigation (150ms, reduces motor noise false triggers)
PREROLL_FRAMES = 5               # Keep last N frames before onset (~150ms) to capture word beginnings
NOISE_CALIBRATION_SECONDS = 2    # Seconds to measure ambient noise
ENERGY_THRESHOLD_MULTIPLIER = 2.0  # Speech must be this many times louder than noise
NOISE_EMA_ALPHA = 0.03           # EMA smoothing for adaptive noise floor (faster adaptation)
NOISE_FLOOR_MIN = 0.001          # Minimum noise floor (prevent threshold from dropping to zero)
NOISE_FLOOR_MAX = 0.05           # Maximum noise floor (prevent threshold from going absurdly high)
NAV_ENERGY_MULT = 5.0            # Extra energy multiplier during navigation (suppresses motor noise)

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
# above the 0.15s SPEAKER_TAIL_COOLDOWN_S grace period. On overflow we
# drop frames in the callback (see audio_callback) rather than blocking
# PortAudio's high-priority thread.
AUDIO_QUEUE_MAX = 33
audio_queue = queue.Queue(maxsize=AUDIO_QUEUE_MAX)
MIC_CHANNEL = 0  # 0=left (beamformed), 1=right (ASR); set from --channel arg

# Mic-mute interlock — derived from AudioPlayer state instead of a flag.
# The audio callback discards input frames whenever the player is busy
# (TTS speaking or beep playing) so the mic can never feed Spot's own
# voice back into ASR. Wired up in main() before the input stream opens.
_audio_player: AudioPlayer | None = None

# Cooldown after the player goes idle: the OS audio buffer keeps draining
# for ~tens of ms after stream.write() returns, so the mic would otherwise
# pick up the tail of Spot's own utterance. 150ms is enough headroom on
# the USB speaker; raise it if you ever see TTS-tail bleed into ASR.
SPEAKER_TAIL_COOLDOWN_S = 0.15
_player_last_busy_at = 0.0  # monotonic timestamp of the most recent busy frame

# The mic-mute watchdog lives inside AudioPlayer (audio_player.py) as its own
# daemon thread. Putting it there instead of the main loop here means it can
# still recover from a wedge that happens *inside* process_utterance, when
# this loop is blocked on tts.wait() — which was the exact failure shape that
# motivated the refactor.


def audio_callback(indata, frames, time_info, status):
    """Sounddevice callback - converts stereo to mono PCM16.

    Mute logic: while the AudioPlayer is busy (TTS speaking or beep playing)
    we discard frames, and we keep discarding for SPEAKER_TAIL_COOLDOWN_S
    after it goes idle so the OS audio buffer has time to fully drain.
    The watchdog that recovers a wedged player runs in the main loop, not
    here — this callback runs on PortAudio's high-priority thread and must
    not do heavy work or block.
    """
    global _player_last_busy_at
    now = time.monotonic()  # local clock — PortAudio's per-stream clock isn't comparable
    if _audio_player is not None and _audio_player.is_busy():
        _player_last_busy_at = now
        return  # discard frame — speaker is active
    if now - _player_last_busy_at < SPEAKER_TAIL_COOLDOWN_S:
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
        audio_queue.put_nowait(pcm16)
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
    Measure ambient noise to establish initial energy threshold.
    Returns noise_rms (seeds the adaptive noise floor).
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

        # If noise is suspiciously high, robot motors may be running during calibration.
        # Cap to a reasonable value so speech detection still works.
        if noise_rms > 0.003:
            print(f"[WARNING: High noise ({noise_rms:.5f}) — robot motors running? Capping to 0.001]")
            noise_rms = 0.001

        return noise_rms

    except Exception as e:
        print(f"[Calibration error: {e}]")
        return 0.01  # Default fallback


# ============================================================================
# Audio Processing
# ============================================================================
def compute_rms(pcm_bytes: bytes) -> float:
    """Compute RMS energy of PCM16 audio."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return np.sqrt(np.mean(audio ** 2))


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
def execute_on_spot(intent: dict) -> bool:
    """Execute parsed intent on Spot robot."""
    try:
        from src.voice_control.spot_dispatch import dispatch_intent
        return dispatch_intent(intent)
    except Exception as e:
        print(f"[Spot error: {e}]")
        return False


def get_spot_state() -> dict:
    """Get current robot state for LLM context."""
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
        time.sleep(0.5)


# Safety commands that bypass the LLM for zero-latency execution
SAFETY_PATTERNS = [
    (re.compile(r"\b(?:stop|halt)\b", re.IGNORECASE), "stop"),
    (re.compile(r"\bfreeze\b", re.IGNORECASE), "freeze"),
    (re.compile(r"\b(?:emergency\s+stop|e[\s-]?stop)\b", re.IGNORECASE), "estop"),
]


def check_safety_command(text: str):
    """Check if text is a safety command (zero-latency, no LLM needed).

    Returns intent dict if safety command, None otherwise.
    """
    for pattern, intent_name in SAFETY_PATTERNS:
        if pattern.search(text):
            return {"intent": intent_name, "params": {}, "raw": text}
    return None


# Wake phrase pattern — fallback for ASR-based detection if dedicated detector unavailable
WAKE_PHRASE_PATTERN = re.compile(
    r"(?:hey|a|stay|say|heh)\s*[,\-]?\s*spot\b",
    re.IGNORECASE
)


# ============================================================================
# Main Voice Control Loop
# ============================================================================
def main():

    parser = argparse.ArgumentParser(description="Spot Voice Control Client")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices and exit")
    parser.add_argument("--device", type=int, default=None, help="Mic input device index (auto-detected from XVF3800)")
    parser.add_argument("--output-device", type=int, default=None, help="Speaker output device index (auto-detected from UACDemoV1.0)")
    parser.add_argument("--no-brain", action="store_true", help="Disable LLM brain (regex-only)")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--no-wake-word", action="store_true", help="Always listening (skip wake word)")
    parser.add_argument("--debug-audio", action="store_true", help="Print audio levels for mic diagnostics")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="TTS + beep output gain (0.0-1.5, default 1.0)")
    parser.add_argument("--map", type=str, default=None,
                        help="GraphNav map path to upload during session bring-up.")
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
    args = parser.parse_args()

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
    rolling_noise_rms = noise_rms  # Will be updated adaptively during operation
    energy_threshold = rolling_noise_rms * ENERGY_THRESHOLD_MULTIPLIER
    print(f"[Energy threshold: {energy_threshold:.5f} ({ENERGY_THRESHOLD_MULTIPLIER}x noise, adaptive)]")

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
        if brain.is_available():
            print(f"[Brain] Ready — model: {DEFAULT_MODEL}")
            # Warm the single brain model (gemma4:e4b handles both LLM and VLM after
            # Phase 2 collapse). Cold-start ~15-20s blocks AudioPlayer + TTS init, but
            # surfaces model-load errors before the mic opens.
            rec = get_recorder()
            ctx = rec.startup_phase("brain_warmup") if rec else nullcontext()
            with ctx:
                brain.warm_up()
        else:
            print(f"[Brain] Ollama not available — falling back to regex-only mode")
            print(f"[Brain] To enable: sudo systemctl start ollama && ollama pull {DEFAULT_MODEL}")
            brain = None
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
            from wake_word import WakeWordDetector
            wake_detector = WakeWordDetector()
            if wake_detector.is_available():
                print("[WakeWord] sherpa-onnx keyword spotter ready")
            else:
                print("[WakeWord] Detector not available — falling back to ASR-based")
                wake_detector = None
        except Exception as e:
            print(f"[WakeWord] Import failed: {e} — falling back to ASR-based")
            wake_detector = None
    else:
        print("[WakeWord] Disabled (--no-wake-word) — always listening")

    # YOLO models (YOLOv8n, YOLO-World) lazy-load on first use.
    # Pre-loading them at startup starves the audio thread (CPU-bound
    # PyTorch init causes PortAudio init overflow and delays wake word).

    # Eagerly initialize the Spot session (and upload a map if --map was
    # passed) BEFORE starting the audio loop. Without --map this is still
    # useful: it surfaces auth/lease errors before the user starts speaking,
    # rather than at the moment of the first command. With --map, this is
    # the only way to load the map as part of the same session bring-up
    # (lazy init from dispatch would happen too late).
    if args.map:
        print(f"\n[Spot] Eager session init with map: {args.map}")
        from src.voice_control.spot_dispatch import ensure_spot_session
        try:
            ensure_spot_session(map_path=args.map)
        except Exception as e:
            print(f"[Spot] Eager session init failed: {e}")
            print("[Spot] Will retry lazily on first command.")

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

    # ========================================================================
    # Main Loop State
    # ========================================================================
    window = b""
    speech_buffer = bytearray()
    speech_float_buffer = []
    speech_frame_count = 0  # track audio frames (not wall clock) for max duration

    is_speaking = False
    last_speech_time = None
    frame_count = 0
    consecutive_speech = 0       # Tracks consecutive VAD-positive frames for onset debounce
    pending_speech_frames = []   # Buffers frames during onset confirmation
    last_vad_error_log_time = 0.0  # Token-bucket guard for VAD failure logs (one/min max)
    from collections import deque
    preroll_buffer = deque(maxlen=PREROLL_FRAMES)  # Ring buffer for pre-onset audio

    # Wake word state
    state = VoiceState.WAKE_WORD if use_wake_word else VoiceState.LISTENING
    listening_start_time = time.time()

    try:
        while True:
            # Get audio from queue. Short timeout so we still cycle (and stay
            # responsive to SIGTERM / KeyboardInterrupt) even when the audio
            # callback has stopped putting frames in the queue — which is
            # what happens whenever the AudioPlayer is busy (mic muted) or
            # the watchdog has just force-reset the player.
            try:
                pcm = audio_queue.get(timeout=1.0)
            except queue.Empty:
                continue
            window += pcm

            # Process complete frames
            while len(window) >= BYTES_PER_FRAME:
                frame = window[:BYTES_PER_FRAME]
                window = window[BYTES_PER_FRAME:]
                frame_count += 1

                # ============================================================
                # Wake word detection (dedicated detector, runs on every frame)
                # ============================================================
                if state == VoiceState.WAKE_WORD and wake_detector:
                    if wake_detector.process_frame(frame):
                        print(">>> Wake word detected!")
                        beep.wake_detected()
                        state = VoiceState.LISTENING
                        listening_start_time = time.time()
                        # Don't drain audio — remaining speech ("stand up" in
                        # "Hey Spot stand up") stays in buffer for VAD to pick up
                    # Still process VAD for safety commands below
                    # (stop/freeze/estop work even in WAKE_WORD state)

                # Calculate frame energy
                frame_rms = compute_rms(frame)

                # Energy gating: only check VAD if energy is above threshold
                # During navigation, raise threshold to suppress motor noise
                # Follow mode uses lower multiplier — user is nearby speaking
                effective_threshold = energy_threshold
                if _is_robot_moving():
                    if _is_follow_mode():
                        effective_threshold *= 2.0  # Light — user is close
                    else:
                        effective_threshold *= NAV_ENERGY_MULT

                if frame_rms > effective_threshold:
                    try:
                        # Silero VAD: feed float32, then check global is_speech_detected.
                        # 480-sample frames vs 512 window is fine — VAD buffers internally.
                        samples = np.frombuffer(frame, dtype=np.int16).astype(np.float32) / 32768.0
                        vad.accept_waveform(samples)
                        is_speech = vad.is_speech_detected()
                    except Exception as e:
                        if time.time() - last_vad_error_log_time > 60.0:
                            print(f"[VAD] accept_waveform failed: {type(e).__name__}: {e}")
                            last_vad_error_log_time = time.time()
                        is_speech = False
                else:
                    is_speech = False

                # ============================================================
                # LISTENING timeout: return to WAKE_WORD after idle period
                # ============================================================
                if state == VoiceState.LISTENING and not is_speaking:
                    if time.time() - listening_start_time > LISTENING_TIMEOUT:
                        print("[Listening timeout — say 'Hey Spot' to activate]")
                        state = VoiceState.WAKE_WORD
                        if wake_detector:
                            wake_detector.reset()
                        continue

                # ============================================================
                # Speech Detection State Machine (with onset debounce)
                # ============================================================
                if is_speech:
                    consecutive_speech += 1

                    if not is_speaking:
                        # In WAKE_WORD mode with dedicated detector, don't record —
                        # keyword spotter handles wake detection at frame level.
                        # Skips ASR entirely, eliminating crowd noise latency.
                        if state == VoiceState.WAKE_WORD and wake_detector:
                            consecutive_speech = 0
                            pending_speech_frames.clear()
                            continue

                        # Buffer frames while confirming onset
                        pending_speech_frames.append(frame)
                        # Use higher onset threshold during navigation to reduce false triggers
                        onset_threshold = NAV_ONSET_FRAMES if _is_robot_moving() else SPEECH_ONSET_FRAMES
                        if consecutive_speech < onset_threshold:
                            continue  # Wait for more consecutive VAD frames

                        # Onset confirmed — start recording with pre-roll + pending frames
                        if state == VoiceState.WAKE_WORD:
                            print("\n>>> Speech detected (wake word mode — safety only)...")
                        else:
                            print("\n>>> Speech detected...")
                        is_speaking = True
                        speech_frame_count = 0
                        speech_buffer.clear()
                        speech_float_buffer.clear()
                        # Include pre-roll frames (captures word beginnings before VAD trigger)
                        for pf in preroll_buffer:
                            speech_buffer.extend(pf)
                            speech_float_buffer.append(pcm16_to_float32(pf))
                            speech_frame_count += 1
                        preroll_buffer.clear()
                        for pf in pending_speech_frames:
                            speech_buffer.extend(pf)
                            speech_float_buffer.append(pcm16_to_float32(pf))
                            speech_frame_count += 1
                        pending_speech_frames.clear()
                        last_speech_time = time.time()
                    else:
                        # Already recording — add frame normally
                        last_speech_time = time.time()
                        speech_buffer.extend(frame)
                        speech_float_buffer.append(pcm16_to_float32(frame))
                        speech_frame_count += 1

                        # Progress indicator (based on audio frames, not wall clock)
                        audio_duration = speech_frame_count * FRAME_MS / 1000.0
                        if speech_frame_count % 17 == 0:  # ~every 0.5s of audio
                            print(f"    Recording: {audio_duration:.1f}s")

                        # Max duration check (based on audio frames, not wall clock)
                        if speech_frame_count >= MAX_UTTERANCE_FRAMES:
                            print(f"\n>>> Max duration reached ({audio_duration:.1f}s), processing...")
                            safety_only = (state == VoiceState.WAKE_WORD)
                            result = process_utterance(stub, speech_buffer, speech_float_buffer,
                                                       brain, safety_only=safety_only, tts=tts,
                                                       has_wake_detector=bool(wake_detector))
                            is_speaking = False
                            speech_buffer.clear()
                            speech_float_buffer.clear()
                            _drain_audio_queue()
                            if result == "wake_detected" and state == VoiceState.WAKE_WORD:
                                print(">>> Now listening for commands...")
                                beep.wake_detected()
                                state = VoiceState.LISTENING
                                listening_start_time = time.time()
                            elif use_wake_word and state == VoiceState.LISTENING:
                                listening_start_time = time.time()

                else:
                    consecutive_speech = 0
                    pending_speech_frames.clear()
                    # Fill pre-roll ring buffer when idle (captures audio before speech onset)
                    if not is_speaking:
                        preroll_buffer.append(frame)

                    if is_speaking:
                        # Add trailing frames for context
                        if len(speech_float_buffer) > 0:
                            elapsed_silence = time.time() - last_speech_time

                            # Add some trailing silence
                            if elapsed_silence < 0.3:
                                speech_buffer.extend(frame)
                                speech_float_buffer.append(pcm16_to_float32(frame))

                            # Check silence timeout
                            speech_duration = len(speech_float_buffer) * FRAME_MS / 1000.0
                            effective_silence = (SILENCE_TIMEOUT_SHORT if speech_duration < SILENCE_CROSSOVER
                                                 else SILENCE_TIMEOUT_LONG)
                            if elapsed_silence > effective_silence:
                                print(f"\n>>> Processing...")
                                safety_only = (state == VoiceState.WAKE_WORD)
                                result = process_utterance(stub, speech_buffer, speech_float_buffer,
                                                           brain, safety_only=safety_only, tts=tts,
                                                           has_wake_detector=bool(wake_detector))
                                is_speaking = False
                                speech_buffer.clear()
                                speech_float_buffer.clear()
                                _drain_audio_queue()
                                if result == "wake_detected" and state == VoiceState.WAKE_WORD:
                                    print(">>> Now listening for commands...")
                                    beep.wake_detected()
                                    state = VoiceState.LISTENING
                                    listening_start_time = time.time()
                                elif use_wake_word and state == VoiceState.LISTENING:
                                    listening_start_time = time.time()
                    else:
                        # Fully idle: update adaptive noise floor
                        rolling_noise_rms = (1 - NOISE_EMA_ALPHA) * rolling_noise_rms + NOISE_EMA_ALPHA * frame_rms
                        rolling_noise_rms = max(NOISE_FLOOR_MIN, min(NOISE_FLOOR_MAX, rolling_noise_rms))
                        energy_threshold = rolling_noise_rms * ENERGY_THRESHOLD_MULTIPLIER

                # Periodic status when idle
                if not is_speaking and frame_count % 166 == 0:  # ~5 seconds
                    if args.debug_audio:
                        print(f"[{state.name} rms={frame_rms:.5f} noise={rolling_noise_rms:.5f} threshold={energy_threshold:.5f}]")
                    elif state == VoiceState.WAKE_WORD:
                        print("[Say 'Hey Spot' to activate...]")
                    else:
                        print("[Listening...]")

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        try:
            stream.stop()
            time.sleep(0.1)  # let PortAudio drain buffers before close
            stream.close()
        except Exception:
            pass
        cleanup_spot()


def _drain_audio_queue():
    """Discard stale audio frames that accumulated during processing,
    but keep the most recent ~1 second so new speech isn't lost."""
    # Drain everything into a list first
    frames = []
    while not audio_queue.empty():
        try:
            frames.append(audio_queue.get_nowait())
        except queue.Empty:
            break

    if not frames:
        return

    # Keep the last ~1 second of audio (could contain new speech)
    keep_count = max(1000 // FRAME_MS, 1)  # ~33 frames at 30ms
    drained = max(0, len(frames) - keep_count)

    # Put the kept frames back into the queue
    for frame in frames[drained:]:
        audio_queue.put(frame)

    if drained:
        print(f"[Drained {drained} stale audio chunks, kept {len(frames) - drained}]")


MIN_SPEECH_DURATION = 0.3  # Reject utterances shorter than this (catches noise bursts, real words are 0.3s+)


def process_utterance(stub, speech_buffer: bytearray, speech_float_buffer: list,
                      brain=None, safety_only=False, tts=None,
                      has_wake_detector=False):
    """Process recorded speech through ASR then LLM brain (or regex fallback).

    Args:
        safety_only: If True, only execute safety commands (stop/freeze/estop).
                     Non-safety speech is discarded. Used when wake word not detected.
        tts: SpotTTS instance for spoken responses (None = no speech output).
        has_wake_detector: If True, dedicated wake word detector is active —
                          skip ASR-based wake phrase detection (detector handles it).

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

    if not transcript:
        print("[No speech recognized]")
        return None

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
    safety = check_safety_command(clean)
    if safety:
        print(f"[SAFETY] {safety['intent']} — executing immediately")
        if execute_on_spot(safety):
            beep.command_ok()
            print(">>> SAFETY COMMAND EXECUTED")
        else:
            beep.error()
            print(">>> SAFETY COMMAND FAILED")
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
            else:
                # Just the wake phrase with no command — ignore
                print(f"[Wake phrase only — no command after stripping]")
                return None

    rec = get_recorder()
    trace = rec.begin_utterance() if rec else None
    try:
        # ------------------------------------------------------------------
        # 2. LLM Brain mode (primary)
        # ------------------------------------------------------------------
        if brain is not None:
            # Collect current robot state for context
            state = get_spot_state()

            t_llm_start = time.time()
            result = brain.process(clean, state)
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
                query = describe_action.get("params", {}).get("query", "")
                response = (
                    f"Let me take a look for the {query}..."
                    if query
                    else random.choice(_DESCRIBE_ACKS)
                )

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

                # Speak response first, then execute actions
                if tts and response:
                    tts.speak(response)

                if trace:
                    trace.mark("intent_dispatch")
                for i, intent in enumerate(actions):
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
                                    tts.speak(vlm_response)
                            else:
                                beep.error()
                                vlm_error = True
                                fallback = "Sorry, I couldn't capture an image right now."
                                print(f"\nSPOT (VLM, error): \"{fallback}\"")
                                if tts:
                                    tts.wait()
                                    tts.speak(fallback)
                        except Exception as e:
                            beep.error()
                            vlm_error = True
                            print(f"[VLM] Error: {e}")
                            fallback = "Sorry, my vision system isn't working right now."
                            print(f"\nSPOT (VLM, error): \"{fallback}\"")
                            if tts:
                                tts.wait()
                                tts.speak(fallback)
                    else:
                        if execute_on_spot(intent):
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
                                time.sleep(CHAIN_MOVEMENT_WAIT_S)
                            elif cmd in ("sit", "stand", "selfright", "body_height"):
                                time.sleep(CHAIN_POSTURE_WAIT_S)
                if trace:
                    trace.mark("dispatch_complete")
            else:
                # Conversation only — speak response
                if tts and response:
                    tts.speak(response)
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
