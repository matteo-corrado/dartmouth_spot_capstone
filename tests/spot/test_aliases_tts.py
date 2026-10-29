import importlib

import pytest

PAIRS = [
    ("tts", "tts"),
    ("tts.kokoro", "tts.kokoro"),
    ("tts.elevenlabs", "tts.elevenlabs"),
    ("spot_tts", "tts.speaker"),
    ("text_segment", "tts.text_segment"),
    ("animation", "features.mouth"),
    ("animation.mouth", "features.mouth.driver"),
]


@pytest.mark.parametrize("old_name,new_name", PAIRS)
def test_alias_identity(old_name, new_name):
    old = importlib.import_module(f"src.voice_control.{old_name}")
    new = importlib.import_module(f"src.spot.{new_name}")
    assert old is new
