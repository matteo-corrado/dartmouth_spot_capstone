import io, numpy as np, grpc, asyncio
"""
ASR gRPC server using Faster Whisper
-----------------------------------
This module implements a simple streaming automatic speech recognition (ASR)
gRPC server that accepts protobuf-defined streaming requests containing
configuration and PCM16 audio payloads, decodes and buffers the audio, and
produces a final transcription when the client closes the stream.
High-level behavior
- The server exposes a StreamingRecognize RPC that accepts a stream of
    requests. Each request may contain either:
        - config: an initial configuration message (e.g. language_code), or
        - audio: a chunk of raw PCM16 audio bytes (mono, 16 kHz expected).
    The implementation currently ignores repeated config messages after the
    first and concatenates all incoming PCM16 audio into an in-memory buffer.
- When the input stream closes, the buffered audio is converted from int16
    PCM to float32 in [-1.0, 1.0] and passed to a Faster Whisper model for
    single-shot transcription. The server yields exactly one final
    StreamingResponse containing the full transcript and is_final=True.
- This is a proof-of-concept single-shot implementation. For partial
    interim results or lower latency, transcribe on rolling windows instead.
Important constants and expectations
- SAMPLE_RATE = 16000
    The server expects incoming audio to be sampled at 16 kHz. If clients send
    audio at a different rate, results will be incorrect unless resampled
    client-side or server-side resampling is implemented.
Audio format
- Each audio chunk must be PCM16 (signed 16-bit little-endian), mono.
- The internal helper _bytes_to_float32 converts the raw int16 bytes into
    float32 samples in the range [-1.0, 1.0].
ASRServicer
- __init__(self)
    - Loads a WhisperModel from faster_whisper. The code auto-selects device
        ("auto") and uses compute_type="int8_float16" for reduced memory use
        while preserving fp16 compute where available.
    - Adjust model name / compute_type as appropriate for your hardware.
- _bytes_to_float32(self, pcm_bytes: bytes) -> numpy.ndarray
    - Utility to convert int16 PCM bytes to a float32 numpy array scaled to
        [-1, 1]. Returns an empty array if input bytes are empty.
- StreamingRecognize(self, request_iterator, context)
    - Implements the streaming RPC method. It:
            1. Reads the first config message (if present) to obtain language.
            2. Buffers audio.pcm16 from subsequent messages into memory.
            3. On stream close, converts buffered audio to float32 and runs
                 model.transcribe(...) once.
            4. Concatenates segment texts and yields a single final
                 pb.StreamingResponse(is_final=True, transcript=..., stability=1.0).
    - Notes and caveats:
            - This method is implemented as a blocking generator that waits for
                the client to close the stream before producing a result.
            - For streaming/partial transcripts, replace the single-shot logic
                with periodic transcribe calls on sliding windows of audio.
            - The implementation assumes the protobuf types defined in the
                asr_pb2 / asr_pb2_grpc modules, and that audio bytes are available
                on request.audio.pcm16 and config as request.config.language_code.
            - If no audio is received, a final empty transcript is returned.
- Starts a gRPC server on "[::]:50055" with a ThreadPoolExecutor
    (max_workers=2), registers the ASRServicer, and blocks until terminated.
- Performs a graceful shutdown on KeyboardInterrupt (5 second timeout).
- This is an insecure server (no TLS). Use secure credentials for
    production deployments.
Dependencies and runtime notes
- Requires:
        - faster_whisper
        - numpy
        - grpcio
        - the generated protobuf modules (asr_pb2, asr_pb2_grpc)
- GPU acceleration:
        - The WhisperModel is configured to auto-select GPU if available.
        - compute_type="int8_float16" reduces memory but requires hardware
            support; change to e.g. "float16" or "float32" if necessary.
- Resource considerations:
        - The current buffering strategy keeps the entire audio stream in
            memory, which is not suitable for long streaming sessions.
        - For production, stream processing or chunked transcription to
            limit memory consumption and latency is recommended.
- Thread-safety:
        - The faster_whisper model object is created per server instance. If
            multiple concurrent RPCs are anticipated, consider one of:
                - Creating a model pool
                - Serializing access to the model
                - Running multiple server processes
            to avoid contention or GPU memory overcommit.
Example usage (conceptual)
- Client streams a config (optional) then multiple audio chunks (PCM16).
- Server returns a single final StreamingResponse with the full transcript.
Security and hardening
- Replace add_insecure_port with a TLS-enabled server in production and
    validate or authenticate clients appropriately.
- Validate incoming config fields and reject unsupported audio formats.
"""
from concurrent import futures
from faster_whisper import WhisperModel

import asr_pb2 as pb
import asr_pb2_grpc as pbg

SAMPLE_RATE = 16000


class ASRServicer(pbg.ASRServicer):
    def __init__(self):
        # Auto-selects GPU acceleration if available
        self.model = WhisperModel(
            "large-v3-turbo",
            device="auto",
            compute_type="int8_float16"
        )

    def _bytes_to_float32(self, pcm_bytes: bytes) -> np.ndarray:
        # int16 PCM to float32 [-1, 1]
        audio_i16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        return (audio_i16.astype(np.float32) / 32768.0)

    def StreamingRecognize(self, request_iterator, context):
        cfg_seen = False
        buf = io.BytesIO()
        lang = "en"
        for req in request_iterator:
            if req.HasField("config"):
                if cfg_seen:
                    continue
                cfg_seen = True
                lang = req.config.language_code or "en"
            elif req.HasField("audio"):
                buf.write(req.audio.pcm16)

        # Transcribe once the stream closes
        audio = self._bytes_to_float32(buf.getvalue())
        if audio.size == 0:
            yield pb.StreamingResponse(is_final=True, transcript="")
            return

        # Single-shot (POC). For partials, call model.transcribe on rolling windows.
        segments, _ = self.model.transcribe(
            audio,
            beam_size=1,
            vad_filter=True,
            language=lang
        )
        text = "".join(s.text for s in segments).strip()

        # Send one final response
        yield pb.StreamingResponse(is_final=True, transcript=text, stability=1.0)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pbg.add_ASRServicer_to_server(ASRServicer(), server)
    server.add_insecure_port("[::]:50055")
    server.start()
    print("ASR server listening on :50055")
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("Shutting down gRPC server...")
        server.stop(5)

if __name__ == "__main__":
    serve()
