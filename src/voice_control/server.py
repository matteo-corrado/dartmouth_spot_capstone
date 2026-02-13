"""
ASR gRPC server using faster-whisper (CTranslate2 backend) with CUDA.

Receives streaming audio from clients, transcribes using Whisper, and returns
the transcript. Designed for voice control of Spot robot.

Migration from openai-whisper: CTranslate2 provides ~2-3x faster inference
with lower memory usage. CTranslate2 3.24.0 built from source for CUDA 11.4.
"""
import os

# Prevent threading issues on Jetson
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# CTranslate2 built from source — preload .so before import
import ctypes
_ct2_so = os.path.expanduser("~/local/lib/libctranslate2.so.3")
if os.path.exists(_ct2_so):
    ctypes.cdll.LoadLibrary(_ct2_so)

import io
import time
import numpy as np
from faster_whisper import WhisperModel
import grpc
from concurrent import futures

import asr_pb2 as pb
import asr_pb2_grpc as pbg

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
SERVER_PORT = 50055

# Whisper model settings - Jetson AGX Orin with CTranslate2 CUDA
# Options: "tiny", "base", "small", "medium", "large-v2", "large-v3"
# Memory: large-v3 ~3GB (float16 via CTranslate2, vs ~5GB with PyTorch)
WHISPER_MODEL = "large-v3"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE_TYPE = "float16"   # float16 on GPU, int8 for CPU


def build_initial_prompt():
    """Build Whisper initial_prompt with command vocabulary only.

    Location names are intentionally excluded — including them biases Whisper
    to hallucinate those words on white noise / motor noise.
    """
    words = ["Stand", "Sit", "Stop", "Walk", "Turn", "Strafe", "Go to"]
    prompt = ". ".join(words) + "."
    print(f"[Server] Initial prompt: {prompt}")
    return prompt


# Known Whisper hallucination phrases on silence/noise
HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thanks for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "see you next time",
    "bye", "goodbye", "you", "the end", ".", "...",
}

# Minimum audio duration (seconds) — lower because Silero VAD pre-filters noise
MIN_AUDIO_DURATION = 0.15


def _is_repetitive(text: str, threshold: float = 0.7) -> bool:
    """Detect repetitive hallucination (e.g., 'To. To. To. To...').

    Whisper produces these when it hears steady motor/fan noise.
    Returns True if any single word makes up more than `threshold` of all words.
    """
    words = text.strip().lower().split()
    if len(words) < 4:
        return False
    from collections import Counter
    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count / len(words) > threshold


# ============================================================================
# ASR Service Implementation
# ============================================================================
class ASRServicer(pbg.ASRServicer):
    """gRPC service for speech-to-text transcription."""

    def __init__(self, model_name=WHISPER_MODEL):
        print(f"[Server] Loading Whisper model '{model_name}'...")
        print(f"[Server] Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE_TYPE}")

        try:
            self.model = WhisperModel(
                model_name,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE_TYPE,
            )
            print(f"[Server] Model loaded on {WHISPER_DEVICE}")

            # Command vocabulary prompt (no location names — prevents hallucination)
            self.initial_prompt = build_initial_prompt()

            # CTranslate2 warmup — first transcription initializes CUDA context
            print("[Server] Warming up CUDA (first transcription)...")
            warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s silence
            segments, _ = self.model.transcribe(warmup_audio, language="en")
            # Must consume the generator to trigger actual computation
            for _ in segments:
                pass
            print("[Server] CUDA warm, ready for transcription")
        except Exception as e:
            print(f"[Server] Failed to load model: {e}")
            raise

    def StreamingRecognize(self, request_iterator, context):
        """Handle streaming audio transcription request."""
        print("\n[Server] New transcription request")

        # Collect audio from stream
        audio_buffer = io.BytesIO()
        language = "en"
        chunk_count = 0

        try:
            for request in request_iterator:
                if request.HasField("config"):
                    language = request.config.language_code or "en"
                    print(f"[Server] Language: {language}")
                elif request.HasField("audio"):
                    audio_buffer.write(request.audio.pcm16)
                    chunk_count += 1
        except Exception as e:
            print(f"[Server] Stream error: {e}")
            yield pb.StreamingResponse(is_final=True, transcript="")
            return

        # Convert PCM16 to float32
        pcm_bytes = audio_buffer.getvalue()
        if len(pcm_bytes) == 0:
            print("[Server] Empty audio received")
            yield pb.StreamingResponse(is_final=True, transcript="")
            return

        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        duration = len(audio) / SAMPLE_RATE

        print(f"[Server] Received {duration:.1f}s audio ({chunk_count} chunks)")

        # Reject clips too short to be real speech
        if duration < MIN_AUDIO_DURATION:
            print(f"[Server] Too short ({duration:.2f}s < {MIN_AUDIO_DURATION}s), skipping")
            yield pb.StreamingResponse(is_final=True, transcript="")
            return

        print("[Server] Transcribing...")

        # Transcribe with faster-whisper (CTranslate2 backend)
        try:
            start_time = time.time()

            transcribe_kwargs = dict(
                language=language,
                beam_size=1,                  # Greedy decode — beam_size=5 is 3.3s on ct2 3.24/CUDA 11.4 ARM64
                temperature=0.0,
                initial_prompt=self.initial_prompt,
                condition_on_previous_text=False,
                no_speech_threshold=0.7,      # 0.7 works well; 0.6 lets too many noise hallucinations through
                log_prob_threshold=-0.5,
                vad_filter=True,              # Silero VAD pre-filters fan/motor noise
                vad_parameters=dict(min_silence_duration_ms=500),
            )

            segments_gen, info = self.model.transcribe(audio, **transcribe_kwargs)

            # Consume segments and collect text + no_speech_prob
            segments = list(segments_gen)
            text = " ".join(s.text for s in segments).strip()
            elapsed = time.time() - start_time

            print(f"[Server] Done in {elapsed:.2f}s ({duration/elapsed:.1f}x realtime)")
            print(f"[Server] Raw result: \"{text}\"")

            # Filter hallucinations: check no_speech_prob on segments
            if segments:
                avg_no_speech = sum(s.no_speech_prob for s in segments) / len(segments)
                print(f"[Server] Avg no_speech_prob: {avg_no_speech:.3f}")
                if avg_no_speech > 0.5:
                    print(f"[Server] High no_speech_prob ({avg_no_speech:.3f}), likely noise — discarding")
                    text = ""

            # Filter known hallucination phrases
            if text and text.strip(".!?, ").lower() in HALLUCINATION_PHRASES:
                print(f"[Server] Known hallucination phrase \"{text}\" — discarding")
                text = ""

            # Filter repetitive hallucinations (motor noise → "To. To. To. To...")
            if text and _is_repetitive(text):
                print(f"[Server] Repetitive hallucination \"{text[:60]}...\" — discarding")
                text = ""

        except Exception as e:
            print(f"[Server] Transcription error: {e}")
            text = ""

        yield pb.StreamingResponse(is_final=True, transcript=text, stability=1.0)


# ============================================================================
# Server Entry Point
# ============================================================================
def serve():
    """Start the gRPC server."""
    import argparse
    parser = argparse.ArgumentParser(description="Spot ASR Server (faster-whisper / CTranslate2)")
    parser.add_argument("--model", type=str, default=WHISPER_MODEL,
                        help=f"Whisper model name (default: {WHISPER_MODEL})")
    parser.add_argument("--port", type=int, default=SERVER_PORT,
                        help=f"gRPC port (default: {SERVER_PORT})")
    args = parser.parse_args()

    print("=" * 60)
    print("SPOT ASR SERVER (faster-whisper / CTranslate2)")
    print("=" * 60)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_ASRServicer_to_server(ASRServicer(model_name=args.model), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    print(f"\n[Server] Listening on port {args.port}")
    print("[Server] Ready for connections\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.stop(5)


if __name__ == "__main__":
    serve()
