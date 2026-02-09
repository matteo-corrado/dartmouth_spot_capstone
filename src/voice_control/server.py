"""
ASR gRPC server using OpenAI Whisper with PyTorch CUDA backend.

Receives streaming audio from clients, transcribes using Whisper, and returns
the transcript. Designed for voice control of Spot robot.
"""
import os

# Prevent threading issues on Jetson
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import io
import time
import numpy as np
import torch
import whisper
import grpc
from concurrent import futures

import asr_pb2 as pb
import asr_pb2_grpc as pbg

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
SERVER_PORT = 50055

# Whisper model settings - Jetson AGX Orin with PyTorch CUDA
# Options: "tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"
# Memory requirements: medium ~3GB, large-v2 ~5GB, turbo ~3GB (optimized)
WHISPER_MODEL = "large-v3"   # Best accuracy for voice commands
WHISPER_DEVICE = "cuda"      # Use Orin GPU acceleration
WHISPER_FP16 = True          # Use FP16 for GPU (2x faster than FP32)


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
    "thank you", "thanks for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "see you next time",
    "bye", "goodbye", "you", "the end",
}

# Minimum audio duration (seconds) to consider as real speech
MIN_AUDIO_DURATION = 0.5


# ============================================================================
# ASR Service Implementation
# ============================================================================
class ASRServicer(pbg.ASRServicer):
    """gRPC service for speech-to-text transcription."""

    def __init__(self):
        print(f"[Server] Loading Whisper model '{WHISPER_MODEL}'...")
        print(f"[Server] Device: {WHISPER_DEVICE}, FP16: {WHISPER_FP16}")

        try:
            self.model = whisper.load_model(WHISPER_MODEL, device=WHISPER_DEVICE)
            device_name = torch.cuda.get_device_name(0) if WHISPER_DEVICE == "cuda" else "CPU"
            print(f"[Server] Model loaded on: {device_name}")

            # Command vocabulary prompt (no location names — prevents hallucination)
            self.initial_prompt = build_initial_prompt()

            # CUDA warmup - first transcription compiles kernels and is 5-10x slower
            print("[Server] Warming up CUDA (first transcription)...")
            warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)  # 1s silence
            self.model.transcribe(warmup_audio, language="en", fp16=WHISPER_FP16)
            print("[Server] ✓ CUDA warm, ready for transcription")
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

        # Transcribe with optimized settings
        try:
            start_time = time.time()

            result = self.model.transcribe(
                audio,
                language=language,
                beam_size=1,              # Greedy decode (fast)
                best_of=1,               # Single candidate (fast)
                temperature=0.0,          # Deterministic output
                fp16=WHISPER_FP16,
                initial_prompt=self.initial_prompt,
                condition_on_previous_text=False,
                no_speech_threshold=0.7,
                logprob_threshold=-0.5,
                verbose=False
            )
            text = result["text"].strip()
            elapsed = time.time() - start_time

            print(f"[Server] Done in {elapsed:.2f}s ({duration/elapsed:.1f}x realtime)")
            print(f"[Server] Raw result: \"{text}\"")

            # Filter hallucinations: check no_speech_prob on segments
            segments = result.get("segments", [])
            if segments:
                avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
                print(f"[Server] Avg no_speech_prob: {avg_no_speech:.3f}")
                if avg_no_speech > 0.5:
                    print(f"[Server] High no_speech_prob ({avg_no_speech:.3f}), likely noise — discarding")
                    text = ""

            # Filter known hallucination phrases
            if text and text.strip(".!?, ").lower() in HALLUCINATION_PHRASES:
                print(f"[Server] Known hallucination phrase \"{text}\" — discarding")
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
    print("=" * 60)
    print("SPOT ASR SERVER")
    print("=" * 60)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_ASRServicer_to_server(ASRServicer(), server)
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    server.start()

    print(f"\n[Server] Listening on port {SERVER_PORT}")
    print("[Server] Ready for connections\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.stop(5)


if __name__ == "__main__":
    serve()
