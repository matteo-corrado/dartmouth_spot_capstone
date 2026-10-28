"""client_mic mic gating: chimes don't mute the mic, speech does (Task 8)."""
import numpy as np

import src.voice_control.client_mic as cm


class _Player:
    def __init__(self, busy, speech):
        self._b, self._s = busy, speech

    def is_busy(self):
        return self._b

    def is_speech_busy(self):
        return self._s


def _drain():
    while not cm.audio_queue.empty():
        cm.audio_queue.get_nowait()


def _feed():
    cm.audio_callback(np.zeros((480, 2), np.float32), 480, None, None)


def test_chime_only_does_not_mute(monkeypatch):
    monkeypatch.setattr(cm, "_audio_player", _Player(busy=True, speech=False))
    monkeypatch.setattr(cm, "_mic_gated", False)
    monkeypatch.setattr(cm, "_operator_muted", False)
    monkeypatch.setattr(cm, "_player_last_busy_at", 0.0)
    _drain()
    _feed()
    assert not cm.audio_queue.empty()
    _drain()


def test_speech_mutes(monkeypatch):
    monkeypatch.setattr(cm, "_audio_player", _Player(busy=True, speech=True))
    monkeypatch.setattr(cm, "_mic_gated", False)
    monkeypatch.setattr(cm, "_operator_muted", False)
    _drain()
    _feed()
    assert cm.audio_queue.empty()


def test_gate_mutes(monkeypatch):
    monkeypatch.setattr(cm, "_audio_player", None)
    monkeypatch.setattr(cm, "_mic_gated", True)
    monkeypatch.setattr(cm, "_operator_muted", False)
    _drain()
    _feed()
    assert cm.audio_queue.empty()


def test_gate_does_not_extend_speaker_tail(monkeypatch):
    # Reopen: the first frame after the gate lifts must be captured, not
    # eaten by a tail cooldown measured from the last *gated* frame.
    monkeypatch.setattr(cm, "_audio_player", _Player(busy=False, speech=False))
    monkeypatch.setattr(cm, "_operator_muted", False)
    monkeypatch.setattr(cm, "_player_last_busy_at", 0.0)
    monkeypatch.setattr(cm, "_mic_gated", True)
    _drain()
    _feed()
    assert cm.audio_queue.empty() and cm._player_last_busy_at == 0.0
    monkeypatch.setattr(cm, "_mic_gated", False)
    _feed()
    assert not cm.audio_queue.empty()
    _drain()
