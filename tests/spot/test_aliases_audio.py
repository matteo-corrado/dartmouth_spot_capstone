import importlib

import pytest

PAIRS = [
    ("audio_player", "player"),
    ("audio_feedback", "chimes"),
    ("audio_frontend", "frontend"),
    ("barge_in", "barge_in"),
    ("response_gate", "response_gate"),
]


@pytest.mark.parametrize("old_name,new_name", PAIRS)
def test_alias_identity(old_name, new_name):
    old = importlib.import_module(f"src.voice_control.{old_name}")
    new = importlib.import_module(f"src.spot.audio.{new_name}")
    assert old is new
