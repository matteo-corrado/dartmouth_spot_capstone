"""Navigation feature: GraphNav trips (go_to, tour, patrol, come_back), map
and saved-location management, and the regex-only walk_to.

Flags copy the hand-written sets (arm.LOCOMOTION_INTENTS, dispatcher
_UNLOCKED_INTENTS / INFO_INTENTS, intent.SHORT_PATH_BLOCKED_INTENTS,
client_mic._MOTION_FREE_INTENTS). Prompt lines are verbatim from
llm_brain.py; the GBNF has no per-action params rules (grammar_params None).
arm_policy stays None: the locomotion policy depends on the live pose, and the
dispatcher still applies arm_policy_for by name.
"""
from src.spot.brain.intent import _resolve_param
from src.spot.core.registry import ActionSpec, Feature, IntentRule
from src.spot.features.navigation import handlers as _h


def _rule(priority, pattern, intent_name, template):
    """IntentRule whose build gives the same intent/params parse_intent makes
    for this table row (parse_intent still applies its deg/location post-processing)."""
    def build(m):
        return {"intent": intent_name,
                "params": {k: _resolve_param(v, m) for k, v in template.items()}}
    return IntentRule(priority, pattern, build)


FEATURE = Feature(
    name="navigation",
    actions=(
        ActionSpec("walk_to", _h.handle_walk_to, None, None,
                   locomotion=True, short_path_blocked=True),
        ActionSpec("go_to", _h.handle_go_to, None,
                   '- go_to: Navigate to saved location. Params: {"location": "<name>"}. Use lowercase_with_underscores.',
                   locomotion=True, grammar_rank=120, prompt_rank=120),
        ActionSpec("tour", _h.handle_tour, None,
                   '- tour: Visit locations in sequence (one pass). Params: {"locations": ["loc1", "loc2"]} for specific stops, or {"locations": "all"} to visit all. Use for "loop the map", "visit everywhere".',
                   locomotion=True, grammar_rank=130, prompt_rank=130),
        ActionSpec("patrol", _h.handle_patrol, None,
                   '- patrol: Loop through locations continuously until stopped. Params: same as tour. Use for "patrol", "keep looping", "keep patrolling".',
                   locomotion=True, short_path_blocked=True, grammar_rank=140, prompt_rank=140),
        ActionSpec("come_back", _h.handle_come_back, None,
                   '- come_back: Return to position before last navigation. No params. Use for "come back", "go home", "return".',
                   locomotion=True, short_path_blocked=True, grammar_rank=150, prompt_rank=150),
        ActionSpec("save_location", _h.handle_save_location, None,
                   '- save_location: Save current position. Params: {"location": "<name>"}.', grammar_rank=160, prompt_rank=160),
        ActionSpec("list_locations", _h.handle_list_locations, None,
                   '- list_locations: List saved locations. No params.',
                   unlocked=True, info=True, motion_free=True, needs_session=False, grammar_rank=170, prompt_rank=170),
        ActionSpec("load_map", _h.handle_load_map, None,
                   '- load_map: Switch to a different GraphNav map. Params: {"map": "<name>"} (optional — omit to reload the last-used map). The available maps are listed in the robot state under "available_maps". Loading a map disrupts any in-progress navigation and the robot will need to re-localize (best with a fiducial visible).', grammar_rank=180, prompt_rank=180),
        ActionSpec("list_maps", _h.handle_list_maps, None,
                   '- list_maps: List the GraphNav maps available on disk. No params.',
                   unlocked=True, info=True, motion_free=True, needs_session=False, grammar_rank=190, prompt_rank=190),
    ),
    intent_rules=(
        _rule(250, r"\b(?:go to|navigate to|drive to|walk to)\s+(?P<location>[\w\s]+?)(?:\s*\.|$|\s+stop|\s+halt|\s+freeze)", "go_to", {"location": "$location"}),
        _rule(260, r"\b(?:loop|tour)\s+(?:the\s+)?map\b", "tour", {"locations": "all"}),
        _rule(270, r"\bvisit\s+all\s+(?:locations|waypoints|places)\b", "tour", {"locations": "all"}),
        _rule(280, r"\bpatrol\s+(?:the\s+)?map\b", "patrol", {"locations": "all"}),
        _rule(290, r"\b(?:keep\s+)?patrol(?:ling)?\b", "patrol", {"locations": "all"}),
        _rule(300, r"\b(?:come\s+back|go\s+home|return\s+(?:to\s+)?(?:start|home|base))\b", "come_back", {}),
        _rule(310, r"\b(?:save location|remember location|save this as|mark this as)\s+(?P<location>[\w\s]+?)(?:\s*\.|$|\s+stop|\s+halt|\s+freeze)", "save_location", {"location": "$location"}),
        _rule(320, r"\b(?:come here|come to me)\b", "walk_to", {"relative": [0.0, -1.0, 0.0]}),
        _rule(360, r"\b(?:list locations|what locations|show locations|saved locations)\b", "list_locations", {}),
        _rule(370, r"\bload\s+(?:the\s+)?(?P<map>\w+)\s+map\b", "load_map", {"map": "$map"}),
        _rule(380, r"\bload\s+map\s+(?P<map>\w+)\b", "load_map", {"map": "$map"}),
        _rule(390, r"\bload\s+map\b(?!\s+\w)", "load_map", {}),
        _rule(400, r"\b(?:list|show|what)\s+(?:are\s+)?(?:the\s+)?(?:available\s+)?maps\b", "list_maps", {}),
    ),
)
