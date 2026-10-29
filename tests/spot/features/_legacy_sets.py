"""The hand-written intent sets a feature's ActionSpec flags must mirror."""
import src.spot.robot.dispatcher as dispatcher
import src.voice_control.client_mic as cm
from src.voice_control.intent import SHORT_PATH_BLOCKED_INTENTS

LLM_PROMPT_SOURCE = "src/spot/brain/llm_brain.py"


def legacy_flags(name: str) -> dict:
    return {
        "locomotion": name in dispatcher.LOCOMOTION_INTENTS,
        "aborting": name in dispatcher.ABORTING_INTENTS,
        "unlocked": name in dispatcher._UNLOCKED_INTENTS,
        "short_path_blocked": name in SHORT_PATH_BLOCKED_INTENTS,
        "info": name in dispatcher.INFO_INTENTS,
        "state_only": name in cm._STATE_ONLY_INTENTS,
        "motion_free": name in cm._MOTION_FREE_INTENTS,
        "persona": name in cm._PERSONA_INTENTS,
        # legacy _dispatch_core ran exactly the state-only intents before
        # bringing up a session (plus list_maps / list_locations: navigation)
        "needs_session": name not in cm._STATE_ONLY_INTENTS | {"list_maps", "list_locations"},
    }


def declared_flags(spec) -> dict:
    return {k: getattr(spec, k) for k in legacy_flags(spec.name)}
