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
import numpy as np
import sounddevice as sd
import webrtcvad
import grpc

from asr_pb2 import StreamingRequest, StreamingConfig, AudioChunk
from asr_pb2_grpc import ASRStub
from intent import parse_intent
from llm_brain import SpotBrain, DEFAULT_MODEL

# Optional noise reduction (for final audio, not real-time)
try:
    import noisereduce as nr
    NOISE_REDUCE_AVAILABLE = True
except ImportError:
    NOISE_REDUCE_AVAILABLE = False

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480 samples per frame
BYTES_PER_FRAME = FRAME_SAMPLES * 2  # int16 = 2 bytes

VAD_LEVEL = 2                    # webrtcvad aggressiveness (0-3)
SILENCE_TIMEOUT = 1.0            # Seconds of silence to end utterance (robot fans are loud)
MAX_UTTERANCE_SECONDS = 8        # Force-send after this duration
MAX_UTTERANCE_FRAMES = int(MAX_UTTERANCE_SECONDS * SAMPLE_RATE / FRAME_SAMPLES)  # ~267 frames
NOISE_CALIBRATION_SECONDS = 2    # Seconds to measure ambient noise
ENERGY_THRESHOLD_MULTIPLIER = 3.0  # Speech must be this many times louder than noise (robot is noisy)

# ============================================================================
# Audio Queue (filled by callback)
# ============================================================================
audio_queue = queue.Queue()
MIC_CHANNEL = 0  # 0=left (beamformed), 1=right (ASR); set from --channel arg


def audio_callback(indata, frames, time_info, status):
    """Sounddevice callback - converts stereo to mono PCM16."""
    if status:
        print(f"[Audio: {status}]")
    # Select channel: 0=left (beamformed), 1=right (ASR) for stereo mics like XVF3800
    if indata.shape[1] >= 2:
        mono = indata[:, MIC_CHANNEL].astype(np.float32)
    else:
        mono = indata[:, 0].astype(np.float32)
    pcm16 = (mono * 32767).astype(np.int16).tobytes()
    audio_queue.put(pcm16)


# ============================================================================
# Noise Calibration
# ============================================================================
def calibrate_noise_floor(duration_sec: float, device=None) -> tuple:
    """
    Measure ambient noise to establish energy threshold.
    Returns (noise_rms, noise_samples_float32).
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
        return noise_rms, noise_audio

    except Exception as e:
        print(f"[Calibration error: {e}]")
        return 0.01, None  # Default fallback


# ============================================================================
# Audio Processing
# ============================================================================
def compute_rms(pcm_bytes: bytes) -> float:
    """Compute RMS energy of PCM16 audio."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    return np.sqrt(np.mean(audio ** 2))


def apply_noise_reduction(audio_float32: np.ndarray, noise_profile: np.ndarray) -> np.ndarray:
    """Apply noise reduction to audio (batch processing, not real-time)."""
    if not NOISE_REDUCE_AVAILABLE or noise_profile is None:
        return audio_float32

    try:
        return nr.reduce_noise(
            y=audio_float32,
            sr=SAMPLE_RATE,
            y_noise=noise_profile,
            prop_decrease=0.85,         # More aggressive (was 0.75) - better for loud Jetson/Spot fans
            stationary=True,            # Spot's fan noise is constant
            freq_mask_smooth_hz=500,    # Smooth out robot fan noise (low freq hum)
            time_mask_smooth_ms=50      # Reduce motor noise (short bursts)
        ).astype(np.float32)
    except Exception as e:
        print(f"[Noise reduction error: {e}]")
        return audio_float32


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


# ============================================================================
# Main Voice Control Loop
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Spot Voice Control Client")
    parser.add_argument("--server", default="localhost:50055", help="ASR server address")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--no-noise-reduction", action="store_true", help="Disable noise reduction")
    parser.add_argument("--energy-mult", type=float, default=ENERGY_THRESHOLD_MULTIPLIER,
                        help=f"Energy threshold multiplier (default: {ENERGY_THRESHOLD_MULTIPLIER})")
    parser.add_argument("--channel", type=int, default=0, choices=[0, 1],
                        help="Mic channel: 0=left (beamformed), 1=right (ASR). Default: 0")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model for LLM brain (default: {DEFAULT_MODEL})")
    parser.add_argument("--no-brain", action="store_true",
                        help="Disable LLM brain, use regex+LLM-fallback (legacy mode)")
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    global MIC_CHANNEL
    MIC_CHANNEL = args.channel

    # ========================================================================
    # Startup
    # ========================================================================
    print("=" * 60)
    print("SPOT VOICE CONTROL — LLM Brain Mode")
    print("=" * 60)

    # Show device info
    ch_label = "left/beamformed" if MIC_CHANNEL == 0 else "right/ASR"
    if args.device is not None:
        print(f"Audio device: {args.device} (channel {MIC_CHANNEL}: {ch_label})")
    else:
        try:
            info = sd.query_devices(sd.default.device[0])
            print(f"Audio device: {info['name']} (default)")
        except:
            print("Audio device: system default")

    # Calibrate noise floor
    noise_rms, noise_profile = calibrate_noise_floor(NOISE_CALIBRATION_SECONDS, args.device)
    energy_threshold = noise_rms * args.energy_mult
    print(f"[Energy threshold: {energy_threshold:.5f} ({args.energy_mult}x noise)]")

    if NOISE_REDUCE_AVAILABLE and not args.no_noise_reduction:
        print("[Noise reduction: ENABLED]")
    else:
        print("[Noise reduction: DISABLED]")
        noise_profile = None

    # Connect to ASR
    print(f"\nConnecting to ASR server at {args.server}...")
    channel = grpc.insecure_channel(args.server)
    stub = ASRStub(channel)

    # Initialize LLM Brain
    brain = None
    if not args.no_brain:
        print(f"\nInitializing LLM brain (model: {args.model})...")
        brain = SpotBrain(model=args.model)
        if brain.is_available():
            print(f"[Brain] Ready — model: {args.model}")
        else:
            print(f"[Brain] Ollama not available — falling back to regex-only mode")
            print(f"[Brain] To enable: sudo systemctl start ollama && ollama pull {args.model}")
            brain = None
    else:
        print("\n[Brain] Disabled (--no-brain flag). Using regex-only mode.")

    # Initialize VAD
    vad = webrtcvad.Vad(VAD_LEVEL)

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

    mode = f"LLM Brain ({args.model})" if brain else "Regex-only (legacy)"
    print("\n" + "=" * 60)
    print(f"LISTENING — Mode: {mode}")
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
                if frame_rms > energy_threshold:
                    try:
                        is_speech = vad.is_speech(frame, SAMPLE_RATE)
                    except:
                        is_speech = False
                else:
                    is_speech = False

                # ============================================================
                # Speech Detection State Machine
                # ============================================================
                if is_speech:
                    if not is_speaking:
                        # Speech started
                        print("\n>>> Speech detected...")
                        is_speaking = True
                        speech_frame_count = 0
                        speech_buffer.clear()
                        speech_float_buffer.clear()

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
                        process_utterance(stub, speech_buffer, speech_float_buffer, noise_profile, brain)
                        is_speaking = False
                        speech_buffer.clear()
                        speech_float_buffer.clear()
                        # Drain queued audio that accumulated during processing
                        _drain_audio_queue()

                else:
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
                                process_utterance(stub, speech_buffer, speech_float_buffer, noise_profile, brain)
                                is_speaking = False
                                speech_buffer.clear()
                                speech_float_buffer.clear()
                                # Drain queued audio that accumulated during processing
                                _drain_audio_queue()

                # Periodic status when idle
                if not is_speaking and frame_count % 166 == 0:  # ~5 seconds
                    print("[Listening...]")

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        stream.stop()
        stream.close()
        cleanup_spot()


def _drain_audio_queue():
    """Discard all queued audio frames that accumulated during processing."""
    drained = 0
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
            drained += 1
        except queue.Empty:
            break
    if drained:
        print(f"[Drained {drained} stale audio chunks]")


MIN_SPEECH_DURATION = 0.5  # Reject utterances shorter than this (likely noise)


def process_utterance(stub, speech_buffer: bytearray, speech_float_buffer: list,
                      noise_profile, brain=None):
    """Process recorded speech through ASR then LLM brain (or regex fallback).

    Flow:
        Audio → Whisper ASR → transcript
        transcript → safety check (instant regex for stop/estop/freeze)
        transcript → LLM brain (state + history → action + response)
        OR (legacy) → regex parser → intent → dispatch
    """
    if not speech_float_buffer:
        return

    # Combine audio
    audio_float = np.concatenate(speech_float_buffer)
    duration = len(audio_float) / SAMPLE_RATE

    # Reject very short clips (noise bursts, not speech)
    if duration < MIN_SPEECH_DURATION:
        print(f"[Too short ({duration:.2f}s) — skipping]")
        return

    # Apply noise reduction (batch, not real-time)
    if noise_profile is not None and NOISE_REDUCE_AVAILABLE:
        print("[Applying noise reduction...]")
        audio_float = apply_noise_reduction(audio_float, noise_profile)

    # Convert to PCM16 and send to ASR
    pcm_bytes = float32_to_pcm16(audio_float)
    print(f"[Sending {duration:.1f}s to ASR...]")

    transcript = send_to_asr(stub, pcm_bytes)

    if not transcript:
        print("[No speech recognized]")
        return

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
        return

    # ------------------------------------------------------------------
    # 1. Safety fast-path: stop/freeze/estop bypass LLM (zero latency)
    # ------------------------------------------------------------------
    safety = check_safety_command(clean)
    if safety:
        print(f"[SAFETY] {safety['intent']} — executing immediately")
        if execute_on_spot(safety):
            print(">>> SAFETY COMMAND EXECUTED")
        else:
            print(">>> SAFETY COMMAND FAILED")
        return

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
            print(f"Action: {cmd}" + (f" {params}" if params else ""))

            if execute_on_spot(intent):
                print(">>> SUCCESS")
            else:
                print(">>> FAILED")
        else:
            print("(No physical action — conversation only)")
        return

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


if __name__ == "__main__":
    main()
