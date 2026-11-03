"""Launch stub: ``python src/voice_control/client_mic.py`` runs the voice
client, src.spot.app.main.main(). Kept because scripts/run_voice_control.py
launches this path; the code lives in src/spot (docs/architecture/package-map.md).
It holds no state and re-exports nothing.
"""
import pathlib
import sys

# Bootstrap sys.path BEFORE any `from src.*` import: run as a script,
# sys.path[0] is this stub's own directory, not the project root (src/ is
# appended as the old client_mic did).
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
_root = str(pathlib.Path(__file__).resolve().parents[2])
if _root not in sys.path:
    sys.path.insert(0, _root)

if __name__ == "__main__":
    from src.spot.app.main import main
    sys.exit(main())  # 1 on a refused start (C6 / safety KWS / robot still powered)
