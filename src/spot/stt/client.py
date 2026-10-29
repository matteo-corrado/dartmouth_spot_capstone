"""gRPC ASR client: stream one utterance's PCM to the ASR server.

Moved from the old client_mic.py (Task 10), unchanged.
"""
import src.spot.audio.capture as _cap
from src.spot.stt.asr_pb2 import StreamingRequest, StreamingConfig, AudioChunk


# ============================================================================
# ASR Communication
# ============================================================================
def send_to_asr(stub, pcm_bytes: bytes, timeout=None) -> str:
    """Stream audio to ASR server and return transcript (timeout: gRPC deadline, s)."""
    def request_generator():
        # Send config first
        yield StreamingRequest(config=StreamingConfig(
            language_code="en",
            sample_rate_hz=_cap.SAMPLE_RATE,
            enable_punctuation=True
        ))
        # Send audio chunks
        chunk_size = _cap.BYTES_PER_FRAME
        for i in range(0, len(pcm_bytes), chunk_size):
            yield StreamingRequest(audio=AudioChunk(pcm16=pcm_bytes[i:i+chunk_size]))

    try:
        responses = (stub.StreamingRecognize(request_generator()) if timeout is None
                     else stub.StreamingRecognize(request_generator(), timeout=timeout))
        for response in responses:
            if response.is_final:
                return response.transcript.strip()
        return ""
    except Exception as e:
        print(f"[ASR error: {e}]")
        return ""
