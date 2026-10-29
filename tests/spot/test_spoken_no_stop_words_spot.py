"""Spoken lines in the moved robot core / nav code must never contain
stop/halt/freeze/e-stop (the always-on safety KWS hears Spot's own voice).
Same scanner as tests/voice_control/test_spoken_no_stop_words.py."""
import pathlib

from tests.voice_control.test_spoken_no_stop_words import BAD, _spoken_literals

SRC = pathlib.Path(__file__).resolve().parents[2] / "src" / "spot"
FILES = sorted(p for d in ("robot", "nav") for p in (SRC / d).rglob("*.py"))


def test_scanner_sees_the_moved_dispatcher_line():
    lines = _spoken_literals(SRC / "robot" / "dispatcher.py")
    assert any("tuck my arm" in s for s in lines)


def test_no_stop_words_in_robot_and_nav_spoken_lines():
    assert FILES
    for path in FILES:
        bad = [s for s in _spoken_literals(path) if BAD.search(s)]
        assert bad == [], f"{path.relative_to(SRC)}: {bad}"
