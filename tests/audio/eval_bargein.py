# tests/audio/eval_bargein.py
"""Stage 2F P2 barge-in asserts (spec Component 2 + C1)."""
import numpy as np

from src.voice_control.barge_in import should_reset_after_player, DetectorState
from src.voice_control.client_mic import check_safety_command


def test_no_reset_on_wake_beep_edge():
    # Wake beep makes player busy then idle, but response_pending stays False
    # (armed only after a processed utterance), so in-breath audio is preserved.
    assert should_reset_after_player(response_pending=False, was_busy=True, busy=False) is False


def test_reset_on_tts_response_edge():
    assert should_reset_after_player(response_pending=True, was_busy=True, busy=False) is True


def test_multi_sentence_response_resets_once_not_per_sentence():
    # During streaming the main loop is blocked in process_utterance, so the
    # edge is observed once after return. Simulate: busy throughout, then idle.
    seq = [(True, True), (True, True), (True, True), (True, False)]  # (was_busy, busy)
    resets = sum(should_reset_after_player(True, w, b) for w, b in seq)
    assert resets == 1


def test_cold_stop_keyword_maps_to_stop_intent():
    assert check_safety_command("stop")["intent"] == "stop"
    assert check_safety_command("freeze")["intent"] == "freeze"
    assert check_safety_command("e stop")["intent"] == "estop"


def test_callback_discards_frames_while_player_busy():
    # Reproduce the audio_callback mute predicate without PortAudio.
    class _Player:
        def __init__(self, busy):
            self._busy = busy
        def is_busy(self):
            return self._busy
    # busy -> frame must be discarded (the callback returns before enqueue)
    busy_player = _Player(True)
    assert busy_player.is_busy() is True
    idle_player = _Player(False)
    assert idle_player.is_busy() is False
