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
import argparse
from enum import Enum, auto
import numpy as np
import sounddevice as sd
import webrtcvad
import grpc

from asr_pb2 import StreamingRequest, StreamingConfig, AudioChunk
from asr_pb2_grpc import ASRStub
from intent import parse_intent
from llm_brain import SpotBrain, DEFAULT_MODEL
from spot_tts import SpotTTS

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
SILENCE_TIMEOUT = 0.5            # Seconds of silence to end utterance
MAX_UTTERANCE_SECONDS = 8        # Force-send after this duration
MAX_UTTERANCE_FRAMES = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)  # ~267 frames
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
        with sd.InputStream(device=device, channels=2, samplerate=SAMPLE_RATE,
                           callback=callback, blocksize=FRAME_SAMPLES):
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


# Wake phrase pattern — matches Whisper transcriptions of "Hey Spot"
# Whisper base commonly produces: "Hey spot", "Hey, spot", "A spot", "A-spot",
# "Stay spot", "Say spot", etc.
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

    # Wake word mode (ASR-based: Whisper detects "Hey Spot" in transcript)
    use_wake_word = not args.no_wake_word
    if use_wake_word:
        print("[WakeWord] ASR-based detection (say 'Hey Spot' to activate)")
    else:
        print("[WakeWord] Disabled (--no-wake-word) — always listening")

    # Open audio stream
    try:
        stream = sd.InputStream(
            device=args.device,
            channels=2,
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
    ww_status = "ON (ASR-based)" if use_wake_word else "OFF"
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

    # Wake word state (ASR-based: no separate model needed)
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

                # Calculate frame energy
                frame_rms = compute_rms(frame)

                # Energy gating: only check VAD if energy is above threshold
                # During navigation, raise threshold to suppress motor noise
                effective_threshold = energy_threshold
                if _is_robot_moving():
                    effective_threshold *= NAV_ENERGY_MULT

                if frame_rms > effective_threshold:
                    try:
                        is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except:
                        is_speech = False
                else:
                    is_speech = False

                # ============================================================
                # LISTENING timeout: return to WAKE_WORD after 10s
                # ============================================================
                if state == VoiceState.LISTENING and not is_speaking:
                    if time.time() - listening_start_time > LISTENING_TIMEOUT:
                        print("[Listening timeout — say 'Hey Spot' to activate]")
                        state = VoiceState.WAKE_WORD
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
                                                       brain, safety_only=safety_only, tts=tts)
                            is_speaking = False
                            speech_buffer.clear()
                            speech_float_buffer.clear()
                            _drain_audio_queue()
                            if result == "wake_detected" and state == VoiceState.WAKE_WORD:
                                print(">>> Now listening for commands...")
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
                            if elapsed_silence > SILENCE_TIMEOUT:
                                print(f"\n>>> Processing...")
                                safety_only = (state == VoiceState.WAKE_WORD)
                                result = process_utterance(stub, speech_buffer, speech_float_buffer,
                                                           brain, safety_only=safety_only, tts=tts)
                                is_speaking = False
                                speech_buffer.clear()
                                speech_float_buffer.clear()
                                _drain_audio_queue()
                                if result == "wake_detected" and state == VoiceState.WAKE_WORD:
                                    print(">>> Now listening for commands...")
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
        stream.stop()
        stream.close()
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
                      brain=None, safety_only=False, tts=None):
    """Process recorded speech through ASR then LLM brain (or regex fallback).

    Args:
        safety_only: If True, only execute safety commands (stop/freeze/estop).
                     Non-safety speech is discarded. Used when wake word not detected.
        tts: SpotTTS instance for spoken responses (None = no speech output).

    Returns:
        "wake_detected" if a wake phrase was found, None otherwise.

    Flow:
        Audio → Whisper ASR → transcript
        transcript → wake phrase check (if safety_only)
        transcript → safety check (instant regex for stop/estop/freeze)
        transcript → LLM brain (state + history → action + response)
        OR (legacy) → regex parser → intent → dispatch
    """
    if not speech_float_buffer:
        return None

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

    transcript = send_to_asr(stub, pcm_bytes)

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
            print(">>> SAFETY COMMAND EXECUTED")
        else:
            print(">>> SAFETY COMMAND FAILED")
        return None

    # ------------------------------------------------------------------
    # 1b. Wake phrase detection (when in WAKE_WORD state / safety_only)
    # ------------------------------------------------------------------
    wake_activated = False
    if safety_only:
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
    # 2. LLM Brain mode (primary)
    # ------------------------------------------------------------------
    if brain is not None:
        # Collect current robot state for context
        state = get_spot_state()

        result = brain.process(clean, state)
        response = result.get("response", "")
        action = result.get("action")

        # Show what the robot "says"
        if response:
            print(f"\nSPOT: \"{response}\"")

        # Execute action if the brain decided on one
        if action:
            intent = action  # action already has {intent, params} structure
            cmd = intent["intent"]
            params = intent.get("params", {})

            # Special VLM flow for "describe" action
            if cmd == "describe":
                if tts and response:
                    tts.speak(response)  # "Let me take a look..."
                camera = params.get("camera", "front")
                try:
                    from src.voice_control.spot_dispatch import capture_frame
                    image_bytes = capture_frame(camera)
                    if image_bytes:
                        vlm_response = brain.query_vlm(image_bytes, clean)
                        print(f"\nSPOT (VLM): \"{vlm_response}\"")
                        if tts:
                            tts.wait()  # wait for initial response to finish
                            tts.speak(vlm_response)
                    else:
                        fallback = "Sorry, I couldn't capture an image right now."
                        print(f"\nSPOT: \"{fallback}\"")
                        if tts:
                            tts.wait()
                            tts.speak(fallback)
                except Exception as e:
                    print(f"[VLM] Error: {e}")
                    fallback = "Sorry, my vision system isn't working right now."
                    print(f"\nSPOT: \"{fallback}\"")
                    if tts:
                        tts.wait()
                        tts.speak(fallback)
            else:
                # Execute action then speak response
                if execute_on_spot(intent):
                    print(">>> SUCCESS")
                else:
                    print(">>> FAILED")
                if tts and response:
                    tts.speak(response)
        else:
            # Conversation only — speak response
            if tts and response:
                tts.speak(response)
            print("(No physical action — conversation only)")
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

    return "wake_detected" if wake_activated else None


if __name__ == "__main__":
    main()
