"""A13: the stop regex fires only for short or stop-led utterances."""
from src.voice_control.intent import check_safety_command, parse_intent


def test_stop_inside_question_not_safety():
    assert check_safety_command("what's the next stop on the tour") is None


def test_short_stop_is_safety():
    assert check_safety_command("stop")["intent"] == "stop"
    assert check_safety_command("spot stop now")["intent"] == "stop"


def test_leading_stop_long_is_safety():
    assert check_safety_command("stop walking towards the table please")["intent"] == "stop"


def test_estop_wins():
    assert check_safety_command("e-stop")["intent"] == "estop"
    assert check_safety_command("e stop")["intent"] == "estop"


def test_freeze_and_shape():
    assert check_safety_command("freeze") == {"intent": "freeze", "params": {}, "raw": "freeze"}


def test_regex_fallback_uses_same_rule():
    assert parse_intent("what's the next stop on the tour") is None
    assert parse_intent("e-stop")["intent"] == "estop"
