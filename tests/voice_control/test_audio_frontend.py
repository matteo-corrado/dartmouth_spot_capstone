import numpy as np
import pytest

from src.voice_control.audio_frontend import (
    Endpointer, FloorTracker, UtteranceStats, frame_dbfs, passes_gate)


def test_dbfs():
    assert frame_dbfs(b"\x00\x00" * 480) == -120.0
    full = np.full(480, 16384, dtype=np.int16).tobytes()
    assert frame_dbfs(full) == pytest.approx(-6.02, abs=0.1)


def test_floor_cannot_ratchet_fast():
    f = FloorTracker(5.0, 10.0, rise_db_per_s=1.0); f.seed(-60)
    for _ in range(100): f.update(-30, is_speech=False)
    assert f.floor_db <= -60 + 3.5


def test_floor_falls_immediately():
    f = FloorTracker(5.0, 10.0, 1.0); f.seed(-40); f.update(-70, False)
    assert f.floor_db == pytest.approx(-70, abs=0.5)


def test_floor_ignores_speech():
    f = FloorTracker(5.0, 10.0, 1.0); f.seed(-60); f.update(-70, True)
    assert f.floor_db == -60


def test_pause_inside_speech_does_not_end():
    e = Endpointer(3, end_silence_frames=50, max_frames=666)
    events = [e.feed(True, -30, -60) for _ in range(10)] + [e.feed(False, -62, -60) for _ in range(33)] + [e.feed(True, -30, -60)]
    assert "start" in events and "end" not in events


def test_end_after_silence():
    e = Endpointer(3, 50, 666)
    ev = [e.feed(True, -30, -60) for _ in range(10)] + [e.feed(False, -62, -60) for _ in range(50)]
    assert ev.count("start") == 1 and ev[-1] == "end"


def test_quiet_syllable_counts_as_speech_not_silence():
    e = Endpointer(3, 50, 666); [e.feed(True, -30, -60) for _ in range(5)]
    assert all(e.feed(False, -45, -60) is None for _ in range(60))


def test_cap():
    e = Endpointer(3, 50, max_frames=20)
    assert "cap" in [e.feed(True, -30, -60) for _ in range(25)]


def test_snr_gate():
    s = UtteranceStats(); [s.add(-35, True) for _ in range(20)]
    assert passes_gate(s, floor_db=-50, min_speech_dbfs=-50, min_snr_db=10) is True
    assert passes_gate(s, floor_db=-40, min_speech_dbfs=-50, min_snr_db=10) is False


def test_gate_no_speech_frames_fails():
    s = UtteranceStats(); s.add(-20, False)
    assert passes_gate(s, -60, -50, 10) is False
