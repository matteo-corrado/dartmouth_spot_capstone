"""Script shim: moved to src/spot/stt/server.py. Removed in Task 19."""
import pathlib
import sys

_root = str(pathlib.Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

if __name__ == "__main__":
    from src.spot.stt.server import serve
    serve()
