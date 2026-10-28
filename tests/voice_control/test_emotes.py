"""Unit tests for the emote/dance routine catalogue (Task 15)."""
import random

import pytest

from src.voice_control.emotes import DANCES, ROUTINES, prompt_section, resolve_routine


def test_resolve_exact_and_random():
    assert resolve_routine("tail_wag", None) == "tail_wag"
    r = resolve_routine("dance", last="hey_join_me", rng=random.Random(0))
    assert r in DANCES and r != "hey_join_me"


def test_random_never_repeats_last():
    rng = random.Random(1)
    for name in (None, "dance", "random"):
        for last in DANCES:
            for _ in range(20):
                r = resolve_routine(name, last, rng=rng)
                assert r in DANCES and r != last


def test_unknown_raises():
    with pytest.raises(ValueError):
        resolve_routine("signal_alert", None)


def test_catalogue_keys():
    assert set(ROUTINES) == {
        "hey_join_me", "performance_bow", "smooth_sailing", "happy_gesture",
        "tail_wag", "upright_sit", "wanna_play",
    }
    assert set(DANCES) <= set(ROUTINES)


def test_generic_triggers_not_on_specific_routines():
    from src.voice_control.emotes import _TRIGGERS
    for phrases in _TRIGGERS.values():
        assert "dance" not in phrases and "show off" not in phrases


def test_prompt_lists_all():
    s = prompt_section()
    assert all(n in s for n in ROUTINES) and "show off" in s
    for phrase in ("dance", "show off", "do a trick", "take a bow", "wag your tail",
                   "are you happy", "sit up nicely", "want to play"):
        assert phrase in s
