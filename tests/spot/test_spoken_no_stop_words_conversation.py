"""Spoken lines in the code moved out of client_mic (Task 10) must never
contain stop/halt/freeze/e-stop (the always-on safety KWS hears Spot's own
voice). Same scanner as tests/voice_control/test_spoken_no_stop_words.py,
which scans client_mic.py by name (now a shim with no spoken lines)."""
import pathlib

from tests.voice_control.test_spoken_no_stop_words import BAD, _spoken_literals

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "spot"
FILES = sorted([*(SRC / "conversation").glob("*.py"), *(SRC / "app").glob("*.py"),
                SRC / "safety" / "listener.py", SRC / "operator" / "control.py",
                SRC / "audio" / "capture.py", SRC / "mic" / "ring_control.py",
                SRC / "stt" / "client.py"])


def test_scanner_sees_the_moved_client_lines():
    assert any("train of thought" in s for s in _spoken_literals(SRC / "conversation" / "loop.py"))
    assert any("let me think" in s for s in _spoken_literals(SRC / "conversation" / "filler.py"))


def test_no_stop_words_in_moved_client_spoken_lines():
    assert all(p.exists() for p in FILES)
    for path in FILES:
        bad = [s for s in _spoken_literals(path) if BAD.search(s)]
        assert bad == [], f"{path.relative_to(SRC)}: {bad}"
