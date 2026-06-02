# tests/voice_control/test_audio_feedback.py
"""Unit tests for audio feedback tone generation (Stage 2F P6)."""
import numpy as np
from src.voice_control.audio_feedback import AudioFeedback, _tone


class RecordingPlayer:
    """Stand-in AudioPlayer that records enqueued samples."""
    def __init__(self):
        self.calls = []
    def is_available(self):
        return True
    def enqueue_raw(self, samples, rate, label=""):
        self.calls.append((samples, rate, label))


def test_thinking_tone_enqueues_nonempty_float32():
    fb = AudioFeedback(player=RecordingPlayer(), volume=1.0)
    fb.thinking()
    samples, rate, label = fb._player.calls[0]
    assert label == "beep:thinking"
    assert rate == 48000
    assert samples.dtype == np.float32
    assert len(samples) > 0
    assert np.max(np.abs(samples)) > 0.0


def test_existing_tones_still_build():
    fb = AudioFeedback(player=RecordingPlayer(), volume=1.0)
    for name in ("wake_detected", "listening", "command_ok", "error",
                 "chain_next", "ready", "thinking"):
        getattr(fb, name)()
    assert len(fb._player.calls) == 7


def test_noop_without_player():
    fb = AudioFeedback(player=None)
    fb.thinking()  # must not raise
