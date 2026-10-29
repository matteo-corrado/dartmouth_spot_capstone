"""Task 13: the follow feature's declarations match the hand-written base."""
from src.spot.features.follow import FEATURE
from tests.spot.features._legacy_sets import (flag_mismatches, missing_rules,
                                              prompt_lines_missing, rule_mismatches)


def test_flags_match_legacy_sets():
    assert [a.name for a in FEATURE.actions] == ["follow_me"]
    assert flag_mismatches(FEATURE) == []
    assert all(a.arm_policy is None and a.needs_session for a in FEATURE.actions)


def test_prompt_lines_verbatim_and_no_gbnf_params():
    assert prompt_lines_missing(FEATURE) == []
    assert all(a.grammar_params is None for a in FEATURE.actions)


def test_intent_rules_copied_from_intent_table():
    assert rule_mismatches(FEATURE) == []
    assert missing_rules(FEATURE) == []
