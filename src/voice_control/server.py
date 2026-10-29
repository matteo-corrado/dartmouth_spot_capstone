"""Launch stub: ``python src/voice_control/server.py`` runs the ASR gRPC
server, src.spot.stt.server.serve(). Kept because scripts/run_voice_control.py
launches this path and checks it with ``pgrep -f voice_control/server.py``.
It holds no state and re-exports nothing.
"""
import pathlib
import sys

_root = str(pathlib.Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

if __name__ == "__main__":
    from src.spot.stt.server import serve
    serve()
