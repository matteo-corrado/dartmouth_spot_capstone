"""Shared helper for the feature-declaration tests: membership of an action
in the hand-written legacy sets (spec 6.3)."""
import re

import src.voice_control.client_mic as cm
from src.spot.brain import intent
from src.spot.brain.llm_brain import SYSTEM_PROMPT
from src.spot.robot import dispatcher

GBNF = "src/voice_control/grammar/spot_action.gbnf"


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
