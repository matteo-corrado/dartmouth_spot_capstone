"""Shared helper for the feature-declaration tests: membership of an action
in the hand-written legacy sets (spec 6.3), read from the golden base
literals since Task 16 generates the live sets."""
import pathlib
import re

import tests.spot.golden.legacy_literals as golden

cm = dispatcher = intent = golden
GBNF = "tests/spot/golden/spot_action.gbnf"
SYSTEM_PROMPT = (pathlib.Path(__file__).resolve().parents[1] / "golden" / "prompt_actions.txt").read_text()


def legacy_flags(name):
    return {
        "locomotion": name in dispatcher.LOCOMOTION_INTENTS,
        "aborting": name in dispatcher.ABORTING_INTENTS,
        "unlocked": name in dispatcher._UNLOCKED_INTENTS,
        "info": name in dispatcher.INFO_INTENTS,
        "short_path_blocked": name in intent.SHORT_PATH_BLOCKED_INTENTS,
        "state_only": name in cm._STATE_ONLY_INTENTS,
        "motion_free": name in cm._MOTION_FREE_INTENTS,
        "persona": name in cm._PERSONA_INTENTS,
    }


def spec_flags(spec):
    return {k: getattr(spec, k) for k in legacy_flags(spec.name)}


def grammar_action_names(root):
    text = (root / GBNF).read_text()
    line = next(l for l in text.splitlines() if l.startswith("action-name ::="))
    return re.findall(r'\\"(\w+)\\"', line)


def prompt_action_lines():
    return [l for l in SYSTEM_PROMPT.splitlines() if re.match(r"- \w+: ", l)]
