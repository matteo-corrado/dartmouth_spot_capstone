import importlib

import pytest

PAIRS = [
    ("mic_array", "xvf3800"),
    ("ring_animator", "ring_animator"),
    ("state_feedback", "state_feedback"),
    ("spot_leds", "spot_leds"),
]


@pytest.mark.parametrize("old_name,new_name", PAIRS)
def test_alias_identity(old_name, new_name):
    old = importlib.import_module(f"src.voice_control.{old_name}")
    new = importlib.import_module(f"src.spot.mic.{new_name}")
    assert old is new
