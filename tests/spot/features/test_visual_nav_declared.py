"""visual_nav FEATURE declarations equal the hand-written sets, regex table
rows and prompt lines they replace."""
from src.spot.brain.intent import COMMANDS
from src.spot.features.visual_nav import FEATURE
from tests.spot.features._legacy_sets_nav import legacy_flags, prompt_action_lines, spec_flags

NAMES = {"go_to_object", "check_obstacles", "describe"}


def test_declares_exactly_the_visual_nav_intents():
    assert {a.name for a in FEATURE.actions} == NAMES


def test_flags_match_legacy_sets():
    for spec in FEATURE.actions:
        assert spec_flags(spec) == legacy_flags(spec.name), spec.name
        assert spec.arm_policy is None and spec.grammar_params is None


def test_prompt_lines_verbatim():
    lines = prompt_action_lines()
    for spec in FEATURE.actions:
        assert spec.prompt_line in lines, spec.name


def test_intent_rules_copy_the_regex_table():
    rows = {i * 10: p for i, (p, n, t) in enumerate(COMMANDS) if n in NAMES}
    assert [r.priority for r in FEATURE.intent_rules] == sorted(rows)
    for rule in FEATURE.intent_rules:
        assert rule.pattern == rows[rule.priority]
