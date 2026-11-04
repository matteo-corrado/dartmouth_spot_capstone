"""Navigation FEATURE declarations equal the hand-written sets, regex table
rows and prompt lines they replace."""
import pathlib
import re

from src.spot.brain.intent import parse_intent
from tests.spot.golden.legacy_literals import COMMANDS  # base table (generated since Task 16)
from src.spot.features.navigation import FEATURE
from tests.spot.features._legacy_sets_nav import (
    grammar_action_names, legacy_flags, prompt_action_lines, spec_flags)

ROOT = pathlib.Path(__file__).resolve().parents[3]
COMMAND_PATTERNS = [(re.compile(p, re.IGNORECASE), n, t) for p, n, t in COMMANDS]
NAMES = {"walk_to", "go_to", "tour", "patrol", "come_back", "load_map",
         "save_location", "list_maps", "list_locations"}


def test_declares_exactly_the_navigation_intents():
    assert {a.name for a in FEATURE.actions} == NAMES


def test_flags_match_legacy_sets():
    for spec in FEATURE.actions:
        assert spec_flags(spec) == legacy_flags(spec.name), spec.name
        assert spec.arm_policy is None and spec.grammar_params is None
    no_session = {a.name for a in FEATURE.actions if not a.needs_session}
    assert no_session == {"list_maps", "list_locations"}


def test_prompt_lines_verbatim():
    lines = prompt_action_lines()
    for spec in FEATURE.actions:
        if spec.name in grammar_action_names(ROOT):
            assert spec.prompt_line in lines, spec.name
        else:  # walk_to: regex-only, not in the grammar or the prompt
            assert spec.prompt_line is None, spec.name


def test_intent_rules_copy_the_regex_table():
    rows = {i * 10: (p, n, t) for i, (p, n, t) in enumerate(COMMANDS) if n in NAMES}
    assert [r.priority for r in FEATURE.intent_rules] == sorted(rows)
    for rule in FEATURE.intent_rules:
        assert rule.pattern == rows[rule.priority][0]


def test_intent_rule_build_matches_parse():
    samples = ["go to the kitchen", "tour the map", "visit all places", "patrol the map",
               "keep patrolling", "come back", "save location my desk", "come here",
               "list locations", "load the jackson map", "load map thayer", "load map",
               "list maps"]
    for text in samples:
        hit = next((i, pat, n, t) for i, (pat, n, t) in enumerate(COMMAND_PATTERNS) if pat.search(text))
        rule = next(r for r in FEATURE.intent_rules if r.priority == hit[0] * 10)
        built = rule.build(hit[1].search(text))
        assert built["intent"] == hit[2]
        assert set(built["params"]) == set(hit[3])
        params = dict(built["params"])
        if isinstance(params.get("location"), str):  # parse_intent's post-processing
            params["location"] = re.sub(r"\s+", "_", params["location"].strip().lower())
        assert {"intent": built["intent"], "params": params} == \
            {k: v for k, v in parse_intent(text).items() if k != "raw"}, text
