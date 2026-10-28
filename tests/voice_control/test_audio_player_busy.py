"""AudioPlayer.is_speech_busy(): chimes ("beep:" labels) don't count as speech."""
import threading

import numpy as np
import pytest

from src.voice_control import audio_player as ap
from src.voice_control.response_gate import should_reopen_mic


class _BlockingStream:
    """No-op OutputStream whose write() blocks until released, so tasks stay in flight."""

    def __init__(self, release: threading.Event):
        self._release = release

    def write(self, _chunk):
        self._release.wait(timeout=5.0)

    def stop(self):
        pass

    def close(self):
        pass


@pytest.fixture
def fake_stream_player(monkeypatch):
    release = threading.Event()
    if ap.sd is None:
        monkeypatch.setattr(ap, "sd", object())
    monkeypatch.setattr(ap.AudioPlayer, "_open_stream",
                        lambda self, channels: _BlockingStream(release))
    player = ap.AudioPlayer(output_device=None, sample_rate=24000)
    yield player
    release.set()
    player.shutdown()


def test_beep_does_not_count_as_speech(fake_stream_player):
    p = fake_stream_player
    p.enqueue_raw(np.zeros(2400, np.float32), 24000, label="beep:listening")
    assert p.is_busy() is True and p.is_speech_busy() is False


def test_tts_counts_as_speech(fake_stream_player):
    p = fake_stream_player
    p.enqueue_raw(np.zeros(2400, np.float32), 24000, label="tts:hello")
    assert p.is_speech_busy() is True


def test_speech_busy_clears_after_playback(fake_stream_player):
    p = fake_stream_player
    p._stream._release.set()
    p.enqueue_raw(np.zeros(240, np.float32), 24000, label="tts:hi")
    p.wait()
    assert p.is_speech_busy() is False and p.is_busy() is False


def test_reopen_uses_tuned_delay():
    assert should_reopen_mic(True, reopen_at=10.0 + 0.2, now=10.25) is True
