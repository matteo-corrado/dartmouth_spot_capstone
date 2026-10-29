from types import SimpleNamespace

from src.spot.features import REGISTRY
from src.spot.features.dance import FEATURE


def _legacy_flags(name: str) -> dict:
    """Membership in the hand-written intent sets each flag mirrors (spec 6.3),
    as captured on base (the live sets are generated since Task 16)."""
    import tests.spot.golden.legacy_literals as dispatcher
    intent = cm = dispatcher
    return {
        "locomotion": name in dispatcher.LOCOMOTION_INTENTS,
        "aborting": name in dispatcher.ABORTING_INTENTS,
        "unlocked": name in dispatcher._UNLOCKED_INTENTS,
        "info": name in dispatcher.INFO_INTENTS,
        "short_path_blocked": name in intent.SHORT_PATH_BLOCKED_INTENTS,
        "state_only": name in cm._STATE_ONLY_INTENTS,
        "persona": name in cm._PERSONA_INTENTS,
        "motion_free": name in cm._MOTION_FREE_INTENTS,
    }


def test_flags_match_legacy_sets():
    assert [a.name for a in FEATURE.actions] == ["emote"]
    for spec in FEATURE.actions:
        legacy = _legacy_flags(spec.name)
        assert {k: getattr(spec, k) for k in legacy} == legacy, spec.name
        assert spec.needs_session is True
        assert REGISTRY.action(spec.name) is spec


def test_prompt_line_verbatim_from_llm_brain():
    import tests.spot.golden.legacy_literals as intent
    llm_brain = SimpleNamespace(_EMOTE_ACTION_LINE=intent.EMOTE_ACTION_LINE)
    (spec,) = FEATURE.actions
    assert spec.prompt_line == llm_brain._EMOTE_ACTION_LINE
    assert spec.grammar_params is None  # the GBNF has no emote-specific params rule
    assert FEATURE.intent_rules == ()   # intent.py has no emote regex rule
    assert not any(i == "emote" for _, i, _ in intent.COMMANDS)


def test_last_routine_write_seen_through_spot_dispatch(monkeypatch):
    """R21: the moved global is read live through spot_dispatch (no stale copy)."""
    import src.spot.features.dance.handlers as dance_h
    import src.voice_control.spot_dispatch as d
    assert "_last_routine" not in vars(d)
    monkeypatch.setattr(dance_h, "_last_routine", "tail_wag")
    assert d._last_routine == "tail_wag"
