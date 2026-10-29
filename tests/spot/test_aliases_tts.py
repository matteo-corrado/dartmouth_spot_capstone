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


def test_statement_form_import_animation_mouth():
    import src.spot.features.mouth.driver as driver
    import src.voice_control.animation.mouth as m
    assert m is driver


def test_statement_form_import_tts_submodule():
    import src.spot.tts.kokoro as kokoro
    import src.voice_control.tts.kokoro as m
    assert m is kokoro
