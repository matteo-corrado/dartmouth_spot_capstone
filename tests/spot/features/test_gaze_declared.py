from src.spot.features import REGISTRY
from src.spot.features.gaze import FEATURE
from src.spot.features.gaze import handlers as gaze_h


def test_flags_match_legacy_sets():
    assert FEATURE.actions == ()  # no intent: nothing to mirror in the legacy sets
    assert FEATURE in REGISTRY.features
    assert FEATURE.start is None and FEATURE.stop is None  # worker stays in client_mic (T10)
    # CR #7: no bus subscription; the conversation loop calls gaze directly (C5)
    assert FEATURE.subscriptions == ()
    assert not hasattr(gaze_h, "on_speaker_locked")
