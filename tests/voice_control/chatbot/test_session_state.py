import time
from src.voice_control.chatbot.session_state import SessionState


def test_defaults():
    state = SessionState()
    assert state.current_persona == "tour_guide"
    assert state.turn_index == 0
    assert state.last_action_taken is None
    assert state.last_comment_ts == 0.0


def test_as_dict_with_defaults():
    state = SessionState()
    d = state.as_dict()
    assert d["current_persona"] == "tour_guide"
    assert d["turn_index"] == 0
    assert d["last_action_taken"] == "none"  # human-readable for LLM
    # seconds_since_last_comment should be very large when ts=0
    assert d["seconds_since_last_comment"] > 1_000_000


def test_as_dict_with_set_fields():
    now = time.time()
    state = SessionState(
        current_persona="pirate",
        turn_index=5,
        last_action_taken="go_to lobby",
        last_comment_ts=now - 7,
    )
    d = state.as_dict()
    assert d["current_persona"] == "pirate"
    assert d["turn_index"] == 5
    assert d["last_action_taken"] == "go_to lobby"
    assert 7 <= d["seconds_since_last_comment"] <= 9  # tolerate small drift
