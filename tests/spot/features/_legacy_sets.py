"""Shared check for the Wave 4 test_<X>_declared tests: every ActionSpec flag
equals membership in the hand-written legacy sets at base (Task 16: the
golden literals in tests/spot/golden, since the sets are now generated)."""
import re
import pathlib

import src.spot.brain.intent as intent
import tests.spot.golden.legacy_literals as golden  # base sets and COMMANDS
from src.spot.robot.arm import arm_policy_for

_GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "golden"

LEGACY = {
    "locomotion": golden.LOCOMOTION_INTENTS,
    "aborting": golden.ABORTING_INTENTS,
    "unlocked": golden._UNLOCKED_INTENTS,
    "info": golden.INFO_INTENTS,
    "short_path_blocked": golden.SHORT_PATH_BLOCKED_INTENTS,
    "state_only": golden._STATE_ONLY_INTENTS,
    "persona": golden._PERSONA_INTENTS,
    "motion_free": golden._MOTION_FREE_INTENTS,
}


def flag_mismatches(feature):
    bad = []
    for a in feature.actions:
        for flag, legacy in LEGACY.items():
            if getattr(a, flag) != (a.name in legacy):
                bad.append((a.name, flag, getattr(a, flag)))
    return bad


def prompt_lines_missing(feature):
    lines = (_GOLDEN / "prompt_actions.txt").read_text().splitlines()
    return [a.name for a in feature.actions if a.prompt_line not in lines]


def rule_mismatches(feature):
    """Each rule is golden.COMMANDS[priority // 10]: same pattern, and build()
    gives that entry's intent + resolved params on a match."""
    bad = []
    for r in feature.intent_rules:
        pat, name, template = golden.COMMANDS[r.priority // 10]
        if r.priority % 10 or r.pattern != pat:
            bad.append((r.priority, r.pattern))
            continue
        for m in [re.search(pat, s, re.IGNORECASE) for s in _SAMPLES]:
            if m is None:
                continue
            want = {"intent": name, "params": {k: intent._resolve_param(v, m)
                                               for k, v in template.items()}}
            if r.build(m) != want:
                bad.append((r.priority, r.build(m), want))
    return bad


def missing_rules(feature):
    """COMMANDS entries for this feature's intents that no rule covers."""
    names = {a.name for a in feature.actions}
    have = {r.priority // 10 for r in feature.intent_rules}
    return [i for i, (_, n, _) in enumerate(golden.COMMANDS) if n in names and i not in have]


_SAMPLES = ["walk forward 2 meters", "go back 3", "walk forward", "move backward", "back up",
            "strafe left 1.5 m", "move right 2", "step left", "strafe right",
            "turn left 45 degrees", "turn right 30", "turn left", "turn right", "turn around",
            "slow down", "normal speed", "speed up", "crouch", "stand tall", "normal height",
            "follow me", "come with me", "tag along"]


# --- T12 API (mouth/settings tests): flag dicts incl. needs_session ---
LLM_PROMPT_SOURCE = "tests/spot/golden/prompt_actions.txt"  # base prompt action lines


def legacy_flags(name: str) -> dict:
    flags = {k: name in v for k, v in LEGACY.items()}
    # legacy _dispatch_core ran exactly the state-only intents before
    # bringing up a session (plus list_maps / list_locations: navigation)
    flags["needs_session"] = name not in golden._STATE_ONLY_INTENTS | {"list_maps", "list_locations"}
    return flags


def declared_flags(spec) -> dict:
    return {k: getattr(spec, k) for k in legacy_flags(spec.name)}


__all__ = ["LEGACY", "arm_policy_for", "flag_mismatches", "prompt_lines_missing",
           "rule_mismatches", "missing_rules", "LLM_PROMPT_SOURCE", "legacy_flags", "declared_flags"]
