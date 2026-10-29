"""SpeechToText factory.

The client side of ASR is the gRPC bridge to the ASR server on
localhost:50055, as today. The model is picked by the SERVER process from
SPOT_ASR_BACKEND (src/spot/stt/backends); the client reads no backend env,
so this factory reads none either.
"""
import src.spot.stt.client as _stt

ASR_SERVER = "localhost:50055"


class GrpcSpeechToText:
    """SpeechToText over the ASR gRPC server. `stub` is the ASRStub the
    conversation loop still takes (Conversation.run(stub=...))."""

    def __init__(self, stub):
        self.stub = stub

    def transcribe(self, pcm16: bytes, timeout: float | None = None) -> str:
        return _stt.send_to_asr(self.stub, pcm16, timeout=timeout)  # looked up at call time


def make_stt(channel_factory=None, stub_cls=None, server: str = ASR_SERVER) -> GrpcSpeechToText:
    """gRPC channel + ASRStub to `server`. main passes its own grpc /
    ASRStub names (the golden harness fakes them on src.spot.app.main)."""
    if channel_factory is None:
        import grpc
        channel_factory = grpc.insecure_channel
    if stub_cls is None:
        from src.spot.stt.asr_pb2_grpc import ASRStub as stub_cls
    print(f"\nConnecting to ASR server at {server}...")
    return GrpcSpeechToText(stub_cls(channel_factory(server)))
