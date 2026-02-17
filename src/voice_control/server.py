"""
ASR gRPC bridge server — forwards audio to NVIDIA Riva ASR (Canary-Qwen-2.5B).

Receives streaming audio from clients via custom asr.proto (port 50055),
forwards to NVIDIA Riva server (port 50051 by default) for transcription,
applies post-processing filters, and returns the transcript.

Architecture:
    client_mic.py --[custom proto:50055]--> this server --[Riva:50051]--> Riva ASR

Prerequisites:
    1. Riva server running with Canary-Qwen-2.5B (see scripts/setup_riva.sh)
    2. pip install nvidia-riva-client
"""
import os
import time
import numpy as np
import grpc
from concurrent import futures

import riva.client

import asr_pb2 as pb
import asr_pb2_grpc as pbg

# ============================================================================
# Configuration
# ============================================================================
SAMPLE_RATE = 16000
SERVER_PORT = 50055
RIVA_URI = os.environ.get("RIVA_URI", "localhost:50051")

# Known hallucination phrases (Canary is cleaner than Whisper, but motor noise
# and silence can still produce these on rare occasions)
HALLUCINATION_PHRASES = {
    "thank you", "thanks", "thanks for watching", "thanks for listening",
    "please subscribe", "like and subscribe", "see you next time",
    "bye", "goodbye", "you", "the end", ".", "...",
}

# Minimum audio duration (seconds) — reject clicks/bumps that aren't speech
MIN_AUDIO_DURATION = 0.15


def _is_repetitive(text: str, threshold: float = 0.7) -> bool:
    """Detect repetitive hallucination (e.g., 'To. To. To. To...').

    Returns True if any single word makes up more than `threshold` of all words.
    """
    words = text.strip().lower().split()
    if len(words) < 4:
        return False
    from collections import Counter
    most_common_count = Counter(words).most_common(1)[0][1]
    return most_common_count / len(words) > threshold


# ============================================================================
# ASR Service Implementation (bridge to Riva)
# ============================================================================
class ASRServicer(pbg.ASRServicer):
    """gRPC service that bridges custom asr.proto to NVIDIA Riva ASR."""

    def __init__(self, riva_uri=RIVA_URI):
        print(f"[Server] Connecting to Riva ASR at {riva_uri}...")

        try:
            self.auth = riva.client.Auth(uri=riva_uri)
            self.asr_service = riva.client.ASRService(self.auth)

            # Riva recognition config for raw PCM16 mono audio
            self.riva_config = riva.client.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM,
                sample_rate_hertz=SAMPLE_RATE,
                language_code="en-US",
                max_alternatives=1,
                enable_automatic_punctuation=True,
                audio_channel_count=1,
            )

            # Warmup — first Riva request initializes the TensorRT engine
            print("[Server] Warming up Riva ASR (first request)...")
            warmup_audio = np.zeros(SAMPLE_RATE, dtype=np.int16).tobytes()  # 1s silence
            try:
                # Try offline first, fall back to streaming
                self.asr_service.offline_recognize(warmup_audio, self.riva_config)
                self._use_streaming = False
                print("[Server] Using offline recognition mode")
            except Exception as e:
                self._use_streaming = True
                print(f"[Server] Offline mode unavailable ({e})")
                print("[Server] Using streaming recognition mode")

            print("[Server] Riva ASR ready")

        except Exception as e:
            print(f"[Server] Failed to connect to Riva at {riva_uri}: {e}")
            print(f"[Server] Make sure Riva server is running (see scripts/setup_riva.sh)")
            raise

    def StreamingRecognize(self, request_iterator, context):
        """Handle streaming audio transcription request."""
        print("\n[Server] New transcription request")

        # Collect audio from stream
        audio_buffer = bytearray()
        language = "en"
        chunk_count = 0

        try:
            for request in request_iterator:
                if request.HasField("config"):
                    language = request.config.language_code or "en"
                    print(f"[Server] Language: {language}")
                elif request.HasField("audio"):
                    audio_buffer.extend(request.audio.pcm16)
                    chunk_count += 1
        except Exception as e:
            print(f"[Server] Stream error: {e}")
            yield pb.StreamingResponse(is_final=True, transcript="")
            return

        pcm_bytes = bytes(audio_buffer)
        if len(pcm_bytes) == 0:
            print("[Server] Empty audio received")
            yield pb.StreamingResponse(is_final=True, transcript="")
            return

        duration = len(pcm_bytes) / (2 * SAMPLE_RATE)  # 2 bytes per int16 sample
        print(f"[Server] Received {duration:.1f}s audio ({chunk_count} chunks)")

        # Reject clips too short to be real speech
        if duration < MIN_AUDIO_DURATION:
            print(f"[Server] Too short ({duration:.2f}s < {MIN_AUDIO_DURATION}s), skipping")
            yield pb.StreamingResponse(is_final=True, transcript="")
            return

        # Transcribe via Riva
        print("[Server] Transcribing via Riva...")
        try:
            start_time = time.time()

            # Build config with correct language code
            lang_code = f"{language}-US" if len(language) == 2 else language
            config = self.riva_config
            if config.language_code != lang_code:
                config = riva.client.RecognitionConfig(
                    encoding=riva.client.AudioEncoding.LINEAR_PCM,
                    sample_rate_hertz=SAMPLE_RATE,
                    language_code=lang_code,
                    max_alternatives=1,
                    enable_automatic_punctuation=True,
                    audio_channel_count=1,
                )

            text = ""
            if self._use_streaming:
                # Use streaming recognition (Conformer on Jetson only supports this)
                streaming_config = riva.client.StreamingRecognitionConfig(
                    config=config,
                    interim_results=False,
                )
                # Send audio as a single chunk in streaming mode
                def audio_generator():
                    # Send config first, then audio in chunks
                    chunk_size = SAMPLE_RATE * 2  # 1 second chunks (16-bit = 2 bytes/sample)
                    for i in range(0, len(pcm_bytes), chunk_size):
                        yield pcm_bytes[i:i + chunk_size]

                responses = self.asr_service.streaming_response_generator(
                    audio_chunks=audio_generator(),
                    streaming_config=streaming_config,
                )
                for resp in responses:
                    for result in resp.results:
                        if result.is_final and result.alternatives:
                            text += result.alternatives[0].transcript
            else:
                # Use offline recognition
                response = self.asr_service.offline_recognize(pcm_bytes, config)
                for result in response.results:
                    if result.alternatives:
                        text += result.alternatives[0].transcript
            text = text.strip()

            elapsed = time.time() - start_time
            rtf = duration / elapsed if elapsed > 0 else 0
            print(f"[Server] Done in {elapsed:.2f}s ({rtf:.1f}x realtime)")
            print(f'[Server] Raw result: "{text}"')

        except Exception as e:
            print(f"[Server] Riva transcription error: {e}")
            text = ""

        # Post-processing filters (kept from Whisper era — still useful as safety net)
        if text and text.strip(".!?, ").lower() in HALLUCINATION_PHRASES:
            print(f'[Server] Known hallucination phrase "{text}" — discarding')
            text = ""

        if text and _is_repetitive(text):
            print(f'[Server] Repetitive hallucination "{text[:60]}..." — discarding')
            text = ""

        yield pb.StreamingResponse(is_final=True, transcript=text, stability=1.0)


# ============================================================================
# Server Entry Point
# ============================================================================
def serve():
    """Start the gRPC bridge server."""
    import argparse
    parser = argparse.ArgumentParser(description="Spot ASR Server (Riva bridge)")
    parser.add_argument("--port", type=int, default=SERVER_PORT,
                        help=f"gRPC port (default: {SERVER_PORT})")
    parser.add_argument("--riva-uri", type=str, default=RIVA_URI,
                        help=f"Riva server URI (default: {RIVA_URI})")
    args = parser.parse_args()

    print("=" * 60)
    print("SPOT ASR SERVER (NVIDIA Riva / Canary-Qwen-2.5B)")
    print("=" * 60)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_ASRServicer_to_server(ASRServicer(riva_uri=args.riva_uri), server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()

    print(f"\n[Server] listening on port {args.port}")
    print("[Server] Ready for connections\n")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        server.stop(5)


if __name__ == "__main__":
    serve()
