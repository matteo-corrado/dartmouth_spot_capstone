"""
Voice control client for Spot with LLM brain.

Architecture (Boston Dynamics "Robots That Can Chat" style):
    Mic → VAD → Whisper ASR → LLM Brain (state + history + personality) → Action + Response

Safety commands (stop/estop/freeze) bypass the LLM for zero-latency execution.
Everything else goes through the LLM brain which decides what to do AND what to say.
"""
import sys
import pathlib

# Add parent and project root to path for local imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import re
import time
import queue
import signal
import argparse
from enum import Enum, auto
import numpy as np
import sounddevice as sd
import webrtcvad
import grpc


# Convert SIGTERM to KeyboardInterrupt so the main loop's finally block
# runs the graceful shutdown (sit robot down, power off).  This matters
# when the web panel stops the pipeline — it sends SIGINT first, but
# falls back to SIGTERM if the process hasn't exited.
def _sigterm_handler(signum, frame):
    raise KeyboardInterrupt

signal.signal(signal.SIGTERM, _sigterm_handler)

from asr_pb2 import StreamingRequest, StreamingConfig, AudioChunk
from asr_pb2_grpc import ASRStub
from intent import parse_intent
from llm_brain import SpotBrain, DEFAULT_MODEL
from spot_tts import SpotTTS
from audio_feedback import beep

class VoiceState(Enum):
    WAKE_WORD = auto()   # Waiting for "hey spot" (detected via ASR, not a separate model)
    LISTENING = auto()   # Wake word heard, waiting for speech
    RECORDING = auto()   # Speech detected, accumulating audio


LISTENING_TIMEOUT = 300.0  # 5 minutes before requiring wake word again

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples per frame
BYTES_PER_FRAME = FRAME_SAMPLES * 2  # int16 = 2 bytes

VAD_LEVEL = 1                    # webrtcvad aggressiveness (0-3); 1 = catches softer speech at close range
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

# ============================================================================
# Audio Queue (filled by callback)
# ============================================================================
audio_queue = queue.Queue()
MIC_CHANNEL = 0  # 0=left (beamformed), 1=right (ASR); set from --channel arg
_mic_muted = False  # Set True during TTS playback to prevent feedback


def audio_callback(indata, frames, time_info, status):
    """Sounddevice callback - converts stereo to mono PCM16."""
    if _mic_muted:
        return  # Mic muted during TTS — discard all audio
    if status:
        print(f"[Audio: {status}]")
    # Select channel: 0=left (beamformed), 1=right (ASR) for stereo mics like XVF3800
    if indata.shape[1] >= 2:
        mono = indata[:, MIC_CHANNEL].astype(np.float32)
    else:
        mono = indata[:, 0].astype(np.float32)
    pcm16 = (mono * 32767).astype(np.int16).tobytes()
    audio_queue.put(pcm16)


def _mute_mic():
    """Mute mic (called by TTS before playback)."""
    global _mic_muted
    _mic_muted = True


def _unmute_mic():
    """Unmute mic and drain stale audio (called by TTS after playback)."""
    global _mic_muted
    _mic_muted = False
    # Drain any residual audio that leaked through during mute transition
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break


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
        # Try stereo first (XVF3800), fall back to mono if device doesn't support it
        try:
            stream_ctx = sd.InputStream(device=device, channels=2, samplerate=SAMPLE_RATE,
                                        callback=callback, blocksize=FRAME_SAMPLES)
        except Exception:
            print(f"[Stereo not supported on device {device}, falling back to mono]")
            stream_ctx = sd.InputStream(device=device, channels=1, samplerate=SAMPLE_RATE,
                                        callback=callback, blocksize=FRAME_SAMPLES)
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
    """Clean up Spot session on exit."""
    try:
        from src.voice_control.spot_dispatch import close_spot_session
        close_spot_session()
    except Exception:
        pass


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
    parser.add_argument("--device", type=int, default=24, help="Audio device index (default: 24 = XVF3800)")
    parser.add_argument("--no-brain", action="store_true", help="Disable LLM brain (regex-only)")
    parser.add_argument("--no-tts", action="store_true", help="Disable text-to-speech")
    parser.add_argument("--no-wake-word", action="store_true", help="Always listening (skip wake word)")
    parser.add_argument("--debug-audio", action="store_true", help="Print audio levels for mic diagnostics")
    args = parser.parse_args()

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
        except:
            print("Audio device: system default")

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
        brain = SpotBrain(model=DEFAULT_MODEL)
        if brain.is_available():
            print(f"[Brain] Ready — model: {DEFAULT_MODEL}")
            brain.warm_up()
        else:
            print(f"[Brain] Ollama not available — falling back to regex-only mode")
            print(f"[Brain] To enable: sudo systemctl start ollama && ollama pull {DEFAULT_MODEL}")
            brain = None
    else:
        print("\n[Brain] Disabled (--no-brain flag). Using regex-only mode.")

    # Initialize TTS (with mic mute/unmute callbacks to prevent feedback)
    tts = None
    if not args.no_tts:
        print("\nInitializing TTS...")
        tts = SpotTTS(on_mute=_mute_mic, on_unmute=_unmute_mic)
        if tts.is_available():
            print("[TTS] Ready (mic will mute during playback)")
        else:
            print("[TTS] Not available (install piper-tts). Continuing without speech output.")
            tts = None
    else:
        print("\n[TTS] Disabled (--no-tts flag)")

    # Initialize VAD
    vad = webrtcvad.Vad(VAD_LEVEL)

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

    # Pre-load YOLO models in background (non-blocking, CPU only)
    try:
        from visual_nav import preload_models
        preload_models()
        print("[YOLO] Pre-loading models in background...")
    except ImportError:
        pass

    # Open audio stream (try stereo for XVF3800, fall back to mono)
    try:
        try:
            stream = sd.InputStream(
                device=args.device,
                channels=2,
                samplerate=SAMPLE_RATE,
                callback=audio_callback,
                blocksize=FRAME_SAMPLES
            )
        except Exception:
            print(f"[Stereo not supported on device {args.device}, falling back to mono]")
            stream = sd.InputStream(
                device=args.device,
                channels=1,
                samplerate=SAMPLE_RATE,
                callback=audio_callback,
                blocksize=FRAME_SAMPLES
            )
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
    from collections import deque
    preroll_buffer = deque(maxlen=PREROLL_FRAMES)  # Ring buffer for pre-onset audio

    # Wake word state
    state = VoiceState.WAKE_WORD if use_wake_word else VoiceState.LISTENING
    listening_start_time = time.time()

    try:
        while True:
            # Get audio from queue
            pcm = audio_queue.get()
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
                        is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except:
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

        # Show what the robot "says"
        if response:
            print(f"\nSPOT: \"{response}\"")

        if actions:
            beep.command_ok()

            # Speak response first, then execute actions
            if tts and response:
                tts.speak(response)

            for i, intent in enumerate(actions):
                cmd = intent["intent"]
                params = intent.get("params", {})

                if len(actions) > 1:
                    print(f"[Chain] Executing {i+1}/{len(actions)}: {cmd}")
                    if i > 0:
                        beep.chain_next()

                # Special VLM flow for "describe" action
                if cmd == "describe":
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
                            vlm_response = brain.query_vlm(image_bytes, clean, yolo_hint=yolo_hint)
                            print(f"\nSPOT (VLM): \"{vlm_response}\"")
                            if tts:
                                tts.wait()
                                tts.speak(vlm_response)
                        else:
                            beep.error()
                            fallback = "Sorry, I couldn't capture an image right now."
                            print(f"\nSPOT: \"{fallback}\"")
                            if tts:
                                tts.wait()
                                tts.speak(fallback)
                    except Exception as e:
                        beep.error()
                        print(f"[VLM] Error: {e}")
                        fallback = "Sorry, my vision system isn't working right now."
                        print(f"\nSPOT: \"{fallback}\"")
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
                            time.sleep(3.0)  # timed movement commands
                        elif cmd in ("sit", "stand", "selfright", "body_height"):
                            time.sleep(4.0)  # postural changes need time to complete
        else:
            # Conversation only — speak response
            if tts and response:
                tts.speak(response)
            print("(No physical action — conversation only)")
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


if __name__ == "__main__":
    main()
