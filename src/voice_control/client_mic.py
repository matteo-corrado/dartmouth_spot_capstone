"""
Voice control client for Spot with noise handling.

Uses energy-based gating combined with VAD to filter out constant fan/system noise
from Spot and Jetson. Applies noise reduction to the final audio before ASR.
"""
import sys
import pathlib

# Add parent to path for local imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

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
SILENCE_TIMEOUT = 0.8            # Seconds of silence to end utterance
MAX_UTTERANCE_SECONDS = 8        # Force-send after this duration
NOISE_CALIBRATION_SECONDS = 2    # Seconds to measure ambient noise
ENERGY_THRESHOLD_MULTIPLIER = 2.0  # Speech must be this many times louder than noise

# ============================================================================
# Audio Queue (filled by callback)
# ============================================================================
audio_queue = queue.Queue()


def audio_callback(indata, frames, time_info, status):
    """Sounddevice callback - converts stereo to mono PCM16."""
    if status:
        print(f"[Audio: {status}]")
    mono = indata.mean(axis=1).astype(np.float32)
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
        mono = indata.mean(axis=1).astype(np.float32)
        samples.append(mono.copy())

    try:
        with sd.InputStream(device=device, channels=1, samplerate=SAMPLE_RATE,
                           callback=callback, blocksize=FRAME_SAMPLES):
            start = time.time()
            while sum(len(s) for s in samples) < samples_needed:
                time.sleep(0.05)
                if time.time() - start > duration_sec + 1:
                    break

        noise_audio = np.concatenate(samples)[:samples_needed]
        noise_rms = np.sqrt(np.mean(noise_audio ** 2))
        print(f"[Noise floor RMS: {noise_rms:.5f}]")
        print(f"[Energy threshold: {noise_rms * ENERGY_THRESHOLD_MULTIPLIER:.5f}]")
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
            prop_decrease=0.75,
            stationary=True
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
        project_root = pathlib.Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.voice_control.spot_dispatch import dispatch_intent
        return dispatch_intent(intent)
    except Exception as e:
        print(f"[Spot error: {e}]")
        return False


def cleanup_spot():
    """Clean up Spot session on exit."""
    try:
        project_root = pathlib.Path(__file__).resolve().parents[2]
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.voice_control.spot_dispatch import close_spot_session
        close_spot_session()
    except Exception:
        pass


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
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    # ========================================================================
    # Startup
    # ========================================================================
    print("=" * 60)
    print("SPOT VOICE CONTROL")
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

    # Calibrate noise floor
    noise_rms, noise_profile = calibrate_noise_floor(NOISE_CALIBRATION_SECONDS, args.device)
    energy_threshold = noise_rms * args.energy_mult

    if NOISE_REDUCE_AVAILABLE and not args.no_noise_reduction:
        print("[Noise reduction: ENABLED]")
    else:
        print("[Noise reduction: DISABLED]")
        noise_profile = None

    # Connect to ASR
    print(f"\nConnecting to ASR server at {args.server}...")
    channel = grpc.insecure_channel(args.server)
    stub = ASRStub(channel)

    # Initialize VAD
    vad = webrtcvad.Vad(VAD_LEVEL)

    # Open audio stream
    try:
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

    print("\n" + "=" * 60)
    print("LISTENING - Commands: stand, sit, stop, turn left/right")
    print("=" * 60 + "\n")

    # ========================================================================
    # Main Loop State
    # ========================================================================
    window = b""
    speech_buffer = bytearray()
    speech_float_buffer = []

    is_speaking = False
    last_speech_time = None
    speech_start_time = None
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
                        speech_start_time = time.time()
                        speech_buffer.clear()
                        speech_float_buffer.clear()

                    last_speech_time = time.time()
                    speech_buffer.extend(frame)
                    speech_float_buffer.append(pcm16_to_float32(frame))

                    # Progress indicator
                    duration = time.time() - speech_start_time
                    if int(duration * 2) > int((duration - 0.03) * 2):  # Every 0.5s
                        print(f"    Recording: {duration:.1f}s")

                    # Max duration check
                    if duration > MAX_UTTERANCE_SECONDS:
                        print(f"\n>>> Max duration reached, processing...")
                        process_utterance(stub, speech_buffer, speech_float_buffer, noise_profile)
                        is_speaking = False
                        speech_buffer.clear()
                        speech_float_buffer.clear()

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
                                process_utterance(stub, speech_buffer, speech_float_buffer, noise_profile)
                                is_speaking = False
                                speech_buffer.clear()
                                speech_float_buffer.clear()

                # Periodic status when idle
                if not is_speaking and frame_count % 166 == 0:  # ~5 seconds
                    print("[Listening...]")

    except KeyboardInterrupt:
        print("\n\nShutting down...")
    finally:
        stream.stop()
        stream.close()
        cleanup_spot()


def process_utterance(stub, speech_buffer: bytearray, speech_float_buffer: list, noise_profile):
    """Process recorded speech: apply noise reduction, send to ASR, execute command."""
    if not speech_float_buffer:
        return

    # Combine audio
    audio_float = np.concatenate(speech_float_buffer)
    duration = len(audio_float) / SAMPLE_RATE

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

    # Parse and execute
    intent = parse_intent(transcript)

    if intent:
        cmd = intent['intent']
        params = intent.get('params', {})
        print(f"Command: {cmd}" + (f" {params}" if params else ""))

        if execute_on_spot(intent):
            print(">>> SUCCESS")
        else:
            print(">>> FAILED")
    else:
        print("(Not recognized - try: stand, sit, stop, turn left/right)")


if __name__ == "__main__":
    main()
