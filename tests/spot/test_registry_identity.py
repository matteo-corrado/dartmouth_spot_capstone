"""Task 16: the grammar, prompt action list, regex intent table and intent sets
are generated from the core intents + the enabled features, and with every
feature enabled they equal the hand-written base values (golden copies in
tests/spot/golden, captured on base b70939a)."""
import pathlib
import re

import pytest

import src.spot.brain.intent as intent
from src.spot.brain import llm_brain
from src.spot.brain.grammar.generate import build_grammar, build_prompt_actions
from src.spot.core import core_intents
from src.spot.core.core_intents import intent_names, intent_rules
from src.spot.core.registry import Registry
from src.spot.features import REGISTRY
from src.spot.features.enabled import ENABLED
from tests.spot.golden import legacy_literals as golden

ROOT = pathlib.Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "spot" / "golden"
FLAGS = ("locomotion", "aborting", "unlocked", "short_path_blocked", "info",
         "state_only", "motion_free", "persona")


def _norm(text):
    return "\n".join(re.sub(r" {2,}", " ", l).rstrip() for l in text.splitlines())


def _registry(without=()):
    reg = Registry()
    reg.load([n for n in ENABLED if n not in without])
    return reg


def test_all_features_enabled():
    assert [f.name for f in REGISTRY.features] == ENABLED


# ---------------------------------------------------------------- grammar
def test_grammar_identical():
    gold = (GOLDEN / "spot_action.gbnf").read_text()
    assert _norm(build_grammar(REGISTRY)) == _norm(gold)
    assert build_grammar(REGISTRY) == gold  # byte-identical too


def test_committed_grammar_snapshot_is_the_generated_one():
    """src/spot/brain/grammar/spot_action.gbnf is kept (old paths and ported
    tests read it) but is never hand-edited: it must equal the generated text,
    which is what llama-server is actually sent."""
    from src.spot.brain.backends import llamacpp_backend
    assert pathlib.Path(llamacpp_backend.GRAMMAR_PATH).read_text() == build_grammar(REGISTRY)
    assert llamacpp_backend._grammar() == llm_brain._GRAMMAR_TEXT == build_grammar(REGISTRY)


def test_walk_to_absent_from_generated_grammar():
    """R23: walk_to is regex-only (relative x/y/yaw unclamped): the LLM must
    never be able to emit it."""
    assert REGISTRY.action("walk_to") is not None and REGISTRY.action("walk_to").prompt_line is None
    line = next(l for l in build_grammar(REGISTRY).splitlines() if l.startswith("action-name ::="))
    names = re.findall(r'\\"(\w+)\\"', line)
    assert "walk_to" not in names
    assert '"walk_to"' not in build_grammar(REGISTRY)
    assert "- walk_to:" not in build_prompt_actions(REGISTRY)


def test_grammar_action_names_one_line():
    lines = [l for l in build_grammar(REGISTRY).splitlines() if l.startswith("action-name ::=")]
    assert len(lines) == 1


# ---------------------------------------------------------------- prompt
def test_prompt_section_identical():
    gold = (GOLDEN / "prompt_actions.txt").read_text()
    assert build_prompt_actions(REGISTRY) == gold
    p = llm_brain.SYSTEM_PROMPT
    assert "AVAILABLE ACTIONS:\n" + gold + "\n\nRULES:" in p
    assert "{{" not in p


def test_emote_line_prefixes_set_backend():
    """Base had {{EMOTE_ACTION}}- set_backend on one source line: the emote
    bullet comes straight before set_backend."""
    assert golden.EMOTE_ACTION_LINE + "- set_backend:" in build_prompt_actions(REGISTRY)


# ---------------------------------------------------------------- sets
def test_sets_identical():
    assert core_intents.SAFETY_INTENTS == golden.SAFETY_INTENTS
    assert core_intents.SHUTDOWN_INTENTS == golden.SHUTDOWN_INTENTS
    assert core_intents.ABORTING_INTENTS == golden.ABORTING_INTENTS
    assert intent_names("unlocked") == golden._UNLOCKED_INTENTS
    assert intent_names("short_path_blocked") == golden.SHORT_PATH_BLOCKED_INTENTS
    assert intent_names("locomotion") == golden.LOCOMOTION_INTENTS
    assert intent_names("info") == golden.INFO_INTENTS
    assert intent_names("state_only") == golden._STATE_ONLY_INTENTS
    assert intent_names("persona") == set(golden._PERSONA_INTENTS)
    assert intent_names("motion_free") == golden._MOTION_FREE_INTENTS
    # every aborting intent is core: no feature can add one
    assert REGISTRY.names("aborting") == frozenset()
    assert intent_names("aborting") <= golden.ABORTING_INTENTS


def test_users_read_the_generated_sets():
    from src.spot.robot import arm, dispatcher
    assert dispatcher.ABORTING_INTENTS is arm.ABORTING_INTENTS is core_intents.ABORTING_INTENTS
    assert dispatcher.SAFETY_INTENTS is core_intents.SAFETY_INTENTS
    for mod, name in ((dispatcher, "_UNLOCKED_INTENTS"), (dispatcher, "INFO_INTENTS"),
                      (arm, "LOCOMOTION_INTENTS"), (intent, "SHORT_PATH_BLOCKED_INTENTS"),
                      (intent, "COMMANDS")):
        assert not hasattr(mod, name), f"hand-written {name} still in {mod.__name__}"
    assert not hasattr(dispatcher, "LEGACY_FEATURE_DISPATCH")


def test_core_intents_survive_disabling_every_feature():
    empty = Registry()
    assert intent_names("unlocked", empty) >= core_intents.SAFETY_INTENTS
    assert intent_names("unlocked", empty) == golden.SAFETY_INTENTS | {"status", "battery_status"}
    assert intent_names("short_path_blocked", empty) == {"power_off"}
    g = build_grammar(empty)
    for name in ("stop", "freeze", "estop", "stand", "sit", "selfright",
                 "battery_status", "status", "power_off"):
        assert f'"\\"{name}\\""' in g
    built = {r.build(None)["intent"] for r in intent_rules(empty)}
    assert built == {"stand", "sit", "selfright", "battery_status", "status", "power_off"}
    from src.spot.robot import dispatcher
    assert dispatcher.CORE_INTENTS == {"stop", "freeze", "estop", "stand", "sit", "power_off",
                                       "stow", "selfright", "status", "battery_status"}


# ---------------------------------------------------------------- intent table
class _Stub:
    """A match whose every group is "5": enough to resolve any template."""
    def group(self, key):
        return "5"


def test_intent_table_identical():
    rules = intent_rules()
    assert len(rules) == len(golden.COMMANDS)
    for i, (rule, (pat, name, template)) in enumerate(zip(rules, golden.COMMANDS)):
        assert rule.priority == i * 10, (i, rule.pattern)
        assert rule.pattern == pat, i
        want = {"intent": name, "params": {k: intent._resolve_param(v, _Stub())
                                           for k, v in template.items()}}
        assert rule.build(_Stub()) == want, i
    assert [r for _, r in intent.command_patterns()] == rules


def _golden_parse(text):
    """parse_intent at base: the safety check, then the hand-written table."""
    safety = intent.check_safety_command(text)
    if safety is not None:
        return safety
    t = re.sub(r"\s+", " ", re.sub(r"\.\s*", " ", text).strip())
    for pat, name, template in golden.COMMANDS:
        m = re.search(pat, t, re.IGNORECASE)
        if not m:
            continue
        out = {k: intent._resolve_param(v, m) for k, v in template.items()}
        if "deg" in out and isinstance(out["deg"], (int, float)):
            deg = max(0, min(360, out["deg"]))
            out["deg"] = int(deg) if isinstance(deg, float) and deg.is_integer() else deg
        if "location" in out and isinstance(out["location"], str):
            out["location"] = re.sub(r"\s+", "_", out["location"].strip().lower())
        return {"intent": name, "params": out, "raw": text}
    return None


@pytest.mark.parametrize("text", [
    "walk forward 2 meters", "go back 3", "turn left 45 degrees", "turn right 400",
    "go to the Kitchen Area", "save location front door", "come here", "battery",
    "status check", "power off", "load the jackson map", "load map thayer", "load map",
    "list maps", "list locations", "find the red chair", "follow me", "stand up", "sit",
    "self right", "crouch", "slow down", "patrol the map", "keep patrolling", "come back",
    "visit all places", "tour the map", "stop", "e-stop", "freeze", "hello there",
    "what is the next stop", "speed up", "normal height", "step right", "strafe left 1.5 m",
    "Go to. Corner.", "how are you", "turn around", "shut down", "recover",
])
def test_parse_intent_matches_base_table(text):
    assert intent.parse_intent(text) == _golden_parse(text)


# ---------------------------------------------------------------- disabling
def test_disabling_dance_removes_emote_everywhere():
    reg = _registry(without=("dance",))
    assert reg.action("emote") is None
    assert "emote" not in build_grammar(reg)
    assert "emote" not in build_prompt_actions(reg)
    for flag in FLAGS:
        assert "emote" not in intent_names(flag, reg)
    assert all(r.build(_Stub())["intent"] != "emote" for r in intent_rules(reg))
    grammar, prompt = llm_brain.build_system_prompt(reg, "EMOTES:\n- tail_wag")
    assert "emote" not in grammar
    assert "- emote:" not in prompt and "EMOTES:" not in prompt
    # with dance enabled the same section is advertised
    _, full = llm_brain.build_system_prompt(REGISTRY, "EMOTES:\n- tail_wag")
    assert "- emote:" in full and full.endswith("\n\nEMOTES:\n- tail_wag")


def test_disabling_walk_removes_its_rules_and_sets():
    reg = _registry(without=("walk",))
    names = {r.build(_Stub())["intent"] for r in intent_rules(reg)}
    assert not names & {"walk", "strafe", "turn", "body_height", "set_speed"}
    assert not intent_names("locomotion", reg) & {"walk", "strafe", "turn"}
    assert '"\\"walk\\""' not in build_grammar(reg)
    assert intent.parse_intent("walk forward") is not None  # live registry untouched
