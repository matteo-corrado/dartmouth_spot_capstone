import importlib
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]

PAIRS = [
    ("src.voice_control.wake", "src.spot.wake"),
    ("src.voice_control.wake.sherpa_onnx", "src.spot.wake.sherpa_onnx"),
    ("src.voice_control.wake.livekit", "src.spot.wake.livekit"),
    ("src.voice_control.wake.phrases", "src.spot.wake.phrases"),
    ("src.voice_control.wake.safety_kws", "src.spot.safety.kws"),
]


def test_alias_identity():
    for old_name, new_name in PAIRS:
        old = importlib.import_module(old_name)
        new = importlib.import_module(new_name)
        assert old is new, old_name


def test_path_constants_unchanged():
    sherpa = importlib.import_module("src.spot.wake.sherpa_onnx")
    kws = importlib.import_module("src.spot.safety.kws")
    phrases = importlib.import_module("src.spot.wake.phrases")
    assert sherpa._PROJECT_ROOT == ROOT
    assert kws._PROJECT_ROOT == ROOT
    assert phrases._PHRASES_FILE == ROOT / "config" / "wake_phrases.txt"
