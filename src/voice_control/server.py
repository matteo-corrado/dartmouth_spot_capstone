"""
ASR gRPC server using Faster Whisper.

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
import grpc
from concurrent import futures
from faster_whisper import WhisperModel

import asr_pb2 as pb
import asr_pb2_grpc as pbg

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
SERVER_PORT = 50055

# Whisper model settings - adjust based on your hardware
# Options: "tiny", "base", "small", "medium", "large-v3-turbo"
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"      # "cpu" or "cuda"
WHISPER_COMPUTE = "int8"    # "int8" for CPU, "float16" for GPU


# ============================================================================
# ASR Service Implementation
# ============================================================================
class ASRServicer(pbg.ASRServicer):
    """gRPC service for speech-to-text transcription."""

    def __init__(self):
        print(f"[Server] Loading Whisper model '{WHISPER_MODEL}'...")
        print(f"[Server] Device: {WHISPER_DEVICE}, Compute: {WHISPER_COMPUTE}")

        try:
            self.model = WhisperModel(
                WHISPER_MODEL,
                device=WHISPER_DEVICE,
                compute_type=WHISPER_COMPUTE,
                num_workers=1,
                cpu_threads=4
            )
            print("[Server] Model loaded successfully")
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
        print("[Server] Transcribing...")

        # Transcribe
        try:
            start_time = time.time()
            segments, _ = self.model.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=False
            )
            # Consume generator
            text = "".join(seg.text for seg in segments).strip()
            elapsed = time.time() - start_time

            print(f"[Server] Done in {elapsed:.1f}s")
            print(f"[Server] Result: \"{text}\"")

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
