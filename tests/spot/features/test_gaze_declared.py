import pytest

from src.spot.features import REGISTRY
from src.spot.features.gaze import FEATURE
from src.spot.features.gaze import handlers as gaze_h


def test_flags_match_legacy_sets():
    assert FEATURE.actions == ()  # no intent: nothing to mirror in the legacy sets
    assert FEATURE in REGISTRY.features
    assert FEATURE.start is None and FEATURE.stop is None  # worker stays in client_mic (T10)
    assert FEATURE.subscriptions == (("speaker_locked", gaze_h.on_speaker_locked),)


def test_speaker_locked_subscriber_needs_gen(monkeypatch):
    calls = []
    monkeypatch.setattr(gaze_h, "gaze_at_bearing", lambda b, g: calls.append((b, g)) or True)
    assert gaze_h.on_speaker_locked(30.0, None) is False       # no gen: skipped, as client_mic
    with pytest.raises(TypeError):
        gaze_h.on_speaker_locked(body_deg=30.0)                  # bad payload never moves the arm
    assert gaze_h.on_speaker_locked(30.0, 7) is True
    assert calls == [(30.0, 7)]
