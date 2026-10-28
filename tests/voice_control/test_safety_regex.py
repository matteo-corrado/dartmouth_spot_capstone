"""A13: safety rule fires unless 'stop' is a noun use; estop always wins."""
import pytest

from src.voice_control.intent import check_safety_command, parse_intent

FIRE_STOP = [
    "stop", "STOP!", "stop.", "spot stop", "hey spot stop",
    "hey spot please stop now", "hey spot, stop right now",
    "spot could you please stop", "okay spot can you stop", "no no no stop it",
    "stop walking towards the table please", "Hey Spot, stop. Come back.",
    "halt",
    # fix round 2: articles are not exemptions; punctuation breaks exemption
    "come to a stop", "bring it to a stop", "come to a full stop",
    "press the stop button", "Spot, don't do that. Stop!", "next. Stop!",
    "the last one, stop", "uh, a stop",
]
FIRE_ESTOP = [
    "please do an emergency stop", "Emergency! Stop!", "e-stop",
    "e–stop", "e—stop", "estop", "e stop",
]
FIRE_FREEZE = ["freeze", "spot freeze now"]
NO_FIRE = [
    "what's the next stop on the tour", "is this the last stop",
    "where is the bus stop", "that was the first stop", "the robot stopped",
    "it stops here", "hello there", "", "one more pit stop",
]


@pytest.mark.parametrize("text", FIRE_STOP)
def test_fires_stop(text):
    assert check_safety_command(text)["intent"] == "stop"


@pytest.mark.parametrize("text", FIRE_ESTOP)
def test_fires_estop(text):
    assert check_safety_command(text)["intent"] == "estop"


def test_e_comma_stop_fires():
    assert check_safety_command("E, stop")["intent"] in ("estop", "stop")


@pytest.mark.parametrize("text", FIRE_FREEZE)
def test_fires_freeze(text):
    assert check_safety_command(text)["intent"] == "freeze"


@pytest.mark.parametrize("text", NO_FIRE)
def test_benign_does_not_fire(text):
    assert check_safety_command(text) is None


def test_estop_wins_over_stop_and_freeze():
    assert check_safety_command("stop stop emergency stop freeze")["intent"] == "estop"


def test_noun_and_command_stop_mixed_fires():
    assert check_safety_command("this is the last stop so stop")["intent"] == "stop"


@pytest.mark.parametrize("text", [None, "...", "!!", "   ", 42])
def test_never_raises_on_junk(text):
    assert check_safety_command(text) is None


def test_shape():
    assert check_safety_command("freeze") == {"intent": "freeze", "params": {}, "raw": "freeze"}


def test_regex_fallback_uses_same_rule():
    assert parse_intent("what's the next stop on the tour") is None
    assert parse_intent("e-stop")["intent"] == "estop"
    assert parse_intent("hey spot please stop now")["intent"] == "stop"
    # punctuation must reach the check: parse_intent strips "." for other rules
    assert parse_intent("next. Stop!")["intent"] == "stop"
