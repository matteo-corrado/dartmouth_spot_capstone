# tests/voice_control/test_barge_in.py
"""Unit tests for the barge-in detector-state reset helper (P0a)."""
from src.voice_control.barge_in import should_reset_after_player, DetectorState


def test_edge_fires_only_on_busy_to_idle_with_response_pending():
    # busy -> idle, response pending => reset
    assert should_reset_after_player(response_pending=True, was_busy=True, busy=False) is True


def test_no_reset_without_response_pending():
    # wake-beep idle edge must NOT reset (preserves in-breath "Hey Spot stand up")
    assert should_reset_after_player(response_pending=False, was_busy=True, busy=False) is False


def test_no_reset_while_still_busy():
    assert should_reset_after_player(response_pending=True, was_busy=True, busy=True) is False


def test_no_reset_on_idle_to_busy():
    assert should_reset_after_player(response_pending=True, was_busy=False, busy=True) is False


def test_no_reset_when_steady_idle():
    assert should_reset_after_player(response_pending=True, was_busy=False, busy=False) is False


def test_reset_clears_all_detector_buffers_and_not_state():
    calls = {"vad": 0, "wake": 0, "drain": 0}
    st = DetectorState(
        preroll=[1, 2, 3],
        pending=[4, 5],
        consecutive=7,
        is_speaking=True,
        speech_buffer=bytearray(b"abc"),
        speech_float=[0.1, 0.2],
    )
    reset = st.reset(
        vad_reset=lambda: calls.__setitem__("vad", calls["vad"] + 1),
        wake_reset=lambda: calls.__setitem__("wake", calls["wake"] + 1),
        drain=lambda: calls.__setitem__("drain", calls["drain"] + 1),
    )
    assert st.preroll == []
    assert st.pending == []
    assert st.consecutive == 0
    assert st.is_speaking is False
    assert st.speech_buffer == bytearray()
    assert st.speech_float == []
    assert calls == {"vad": 1, "wake": 1, "drain": 1}
    # The helper returns nothing about VoiceState — it cannot touch `state`.
    assert reset is None
