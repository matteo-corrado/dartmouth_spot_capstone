"""Shared check for the Wave 4 test_<X>_declared tests: every ActionSpec flag
equals membership in the hand-written legacy sets at base."""
import re

import src.spot.brain.intent as intent
import src.spot.brain.llm_brain as llm_brain
import src.spot.robot.dispatcher as dispatcher
import src.voice_control.client_mic as client_mic
from src.spot.robot.arm import arm_policy_for

LEGACY = {
    "locomotion": dispatcher.LOCOMOTION_INTENTS,
    "aborting": dispatcher.ABORTING_INTENTS,
    "unlocked": dispatcher._UNLOCKED_INTENTS,
    "info": dispatcher.INFO_INTENTS,
    "short_path_blocked": intent.SHORT_PATH_BLOCKED_INTENTS,
    "state_only": client_mic._STATE_ONLY_INTENTS,
    "persona": client_mic._PERSONA_INTENTS,
    "motion_free": client_mic._MOTION_FREE_INTENTS,
}


def flag_mismatches(feature):
    bad = []
    for a in feature.actions:
        for flag, legacy in LEGACY.items():
            if getattr(a, flag) != (a.name in legacy):
                bad.append((a.name, flag, getattr(a, flag)))
    return bad


def prompt_lines_missing(feature):
    lines = llm_brain.SYSTEM_PROMPT.splitlines()
    return [a.name for a in feature.actions if a.prompt_line not in lines]


def rule_mismatches(feature):
    """Each rule is intent.COMMANDS[priority // 10]: same pattern, and build()
    gives that entry's intent + resolved params on a match."""
    bad = []
    for r in feature.intent_rules:
        pat, name, template = intent.COMMANDS[r.priority // 10]
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
    return [i for i, (_, n, _) in enumerate(intent.COMMANDS) if n in names and i not in have]


_SAMPLES = ["walk forward 2 meters", "go back 3", "walk forward", "move backward", "back up",
            "strafe left 1.5 m", "move right 2", "step left", "strafe right",
            "turn left 45 degrees", "turn right 30", "turn left", "turn right", "turn around",
            "slow down", "normal speed", "speed up", "crouch", "stand tall", "normal height",
            "follow me", "come with me", "tag along"]


# --- T12 API (mouth/settings tests): flag dicts incl. needs_session ---
LLM_PROMPT_SOURCE = "src/spot/brain/llm_brain.py"


def legacy_flags(name: str) -> dict:
    flags = {k: name in v for k, v in LEGACY.items()}
    # legacy _dispatch_core ran exactly the state-only intents before
    # bringing up a session (plus list_maps / list_locations: navigation)
    flags["needs_session"] = name not in client_mic._STATE_ONLY_INTENTS | {"list_maps", "list_locations"}
    return flags


def declared_flags(spec) -> dict:
    return {k: getattr(spec, k) for k in legacy_flags(spec.name)}


__all__ = ["LEGACY", "arm_policy_for", "flag_mismatches", "prompt_lines_missing",
           "rule_mismatches", "missing_rules", "LLM_PROMPT_SOURCE", "legacy_flags", "declared_flags"]
