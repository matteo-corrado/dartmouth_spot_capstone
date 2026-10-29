"""Visual-nav feature: go_to_object (saved location / world object / YOLO),
check_obstacles (LocalGrid read), and describe (declared only: client_mic's
VLM branch runs it; a dispatched describe behaves as the legacy fall-through).

Flags copy the hand-written sets (arm.LOCOMOTION_INTENTS, dispatcher
_UNLOCKED_INTENTS / INFO_INTENTS, intent.SHORT_PATH_BLOCKED_INTENTS,
client_mic._MOTION_FREE_INTENTS). Prompt lines are verbatim from
llm_brain.py; the GBNF has no per-action params rules (grammar_params None).
arm_policy stays None: the dispatcher still applies arm_policy_for by name.
"""
from src.spot.brain.intent import _resolve_param
from src.spot.core.registry import ActionSpec, Feature, IntentRule
from src.spot.features.visual_nav import handlers as _h


def _rule(priority, pattern, intent_name, template):
    """IntentRule whose build gives the same intent/params parse_intent makes
    for this table row (parse_intent still applies its deg/location post-processing)."""
    def build(m):
        return {"intent": intent_name,
                "params": {k: _resolve_param(v, m) for k, v in template.items()}}
    return IntentRule(priority, pattern, build)


FEATURE = Feature(
    name="visual_nav",
    actions=(
        ActionSpec("go_to_object", _h.handle_go_to_object, None,
                   '- go_to_object: Walk toward any visible object Spot can recognise. Params: {"description": "<what to find>"}. Spot uses YOLO/VLM to identify the target in its surroundings — anything you can describe in a few words. Works for ordinary objects (chairs, tables, posters, backpacks, people), architectural features (doors, exits, doorways, windows), and pick-up targets ("the cup on the floor"). Only for things you can SEE in the camera right now — use go_to for saved map locations.',
                   locomotion=True, short_path_blocked=True),
        ActionSpec("describe", _h.handle_describe, None,
                   '- describe: Take a photo and describe what you see. Params: {"camera": "front"|"left"|"right"|"back", "query": "<specific object to look for, if any>"}. Default camera "front". Omit query for general "what do you see" questions.',
                   motion_free=True),
        ActionSpec("check_obstacles", _h.handle_check_obstacles, None,
                   '- check_obstacles: Query Spot\'s built-in obstacle map for nearby obstacles. No params. Use for "is there anything in front of you", "is this room cluttered", "what direction is clearest", "can you move forward safely". Returns distances and bearings without using cameras or YOLO — it\'s a direct read of Spot\'s footstep-planning grid.',
                   unlocked=True, info=True, motion_free=True),
    ),
    intent_rules=(
        _rule(230, r"\b(?:find|look for|search for|locate)\s+(?:the\s+|a\s+|my\s+)?(?P<desc>.+?)(?:\s*\.|$)", "go_to_object", {"description": "$desc"}),
    ),
)
