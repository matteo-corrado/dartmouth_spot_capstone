"""ASR gRPC bridge — dispatches to a configurable Backend.

Backends:
    SPOT_ASR_BACKEND=nemotron  (default) — NVIDIA Nemotron Speech Streaming 0.6B
    SPOT_ASR_BACKEND=parakeet           — NVIDIA Parakeet TDT 0.6B v3 (rollback)

The custom asr.proto (port 50055) is unchanged; client_mic.py uses it as-is.
"""
import os
import pathlib
import sys
from concurrent import futures

# Match client_mic.py bootstrap so bare `import asr_pb2` (cwd=src/voice_control)
# and packaged `from src.voice_control.asr` both resolve.
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
project_root = pathlib.Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import grpc

import asr_pb2 as pb
import asr_pb2_grpc as pbg

from src.voice_control.asr import get_backend

SERVER_PORT = 50055
SAMPLE_RATE = 16000


class ASRServicer(pbg.ASRServicer):
    def __init__(self):
        self.backend = get_backend()
        print(f"[Server] backend={self.backend.name}")

    def StreamingRecognize(self, request_iterator, context):
        pcm_chunks = []
        for req in request_iterator:
            if req.HasField("config"):
                continue
            pcm_chunks.append(req.audio.pcm16)

        pcm_bytes = b"".join(pcm_chunks)
        transcript = self.backend.transcribe(pcm_bytes)
        yield pb.StreamingResponse(is_final=True, transcript=transcript, stability=1.0)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    pbg.add_ASRServicer_to_server(ASRServicer(), server)
    server.add_insecure_port(f"[::]:{SERVER_PORT}")
    server.start()
    print(f"[Server] listening on :{SERVER_PORT}")
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
