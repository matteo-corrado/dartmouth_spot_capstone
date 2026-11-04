"""Golden literals captured on base b70939a (before the Task 16 registry
cut-over): the hand-written intent table and intent sets. The generated
values must equal these exactly."""

# src/spot/brain/intent.py COMMANDS at base, verbatim.
COMMANDS = [
    # === POSTURE ===
    (r"\b(?:stand|stand up|get up)\b",             "stand", {}),
    (r"\b(?:sit|sit down|lay down)\b",             "sit", {}),
    (r"\b(?:self[\s-]?right|get up from fall|recover)\b", "selfright", {}),

    # === BODY HEIGHT ===
    (r"\b(?:crouch|lower body|get low|duck)\b",    "body_height", {"height": -0.15}),
    (r"\b(?:stand tall|raise body|stretch up|max height)\b", "body_height", {"height": 0.1}),
    (r"\b(?:normal height|default height|regular height)\b", "body_height", {"height": 0.0}),

    # === WALKING WITH DISTANCE ===
    (r"\b(?:walk|move|go)\s+forward\s+(?P<dist>\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)?\b", "walk", {"direction": "forward", "distance": "$dist"}),
    (r"\b(?:walk|move|go)\s+(?:backward|back)\s+(?P<dist>\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)?\b", "walk", {"direction": "backward", "distance": "$dist"}),
    (r"\b(?:walk|move|go)\s+forward\b",            "walk", {"direction": "forward", "distance": 1.0}),
    (r"\b(?:walk|move|go)\s+(?:backward|back)\b",  "walk", {"direction": "backward", "distance": 1.0}),
    (r"\bback\s*up\b",                             "walk", {"direction": "backward", "distance": 1.0}),

    # === STRAFING ===
    (r"\b(?:strafe|move|step)\s+left\s+(?P<dist>\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)?\b", "strafe", {"direction": "left", "distance": "$dist"}),
    (r"\b(?:strafe|move|step)\s+right\s+(?P<dist>\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)?\b", "strafe", {"direction": "right", "distance": "$dist"}),
    (r"\b(?:strafe|step)\s+left\b",                "strafe", {"direction": "left", "distance": 0.5}),
    (r"\b(?:strafe|step)\s+right\b",               "strafe", {"direction": "right", "distance": 0.5}),

    # === TURNING ===
    (r"\bturn\s+left\s+(?P<deg>\d+(?:\.\d+)?)\s*(?:deg(?:rees)?)?\b",  "turn", {"deg": "$deg", "dir": "left"}),
    (r"\bturn\s+right\s+(?P<deg>\d+(?:\.\d+)?)\s*(?:deg(?:rees)?)?\b", "turn", {"deg": "$deg", "dir": "right"}),
    (r"\bturn\s+left\b",                           "turn", {"deg": 90, "dir": "left"}),
    (r"\bturn\s+right\b",                          "turn", {"deg": 90, "dir": "right"}),
    (r"\b(?:turn around|spin around|about face|one eighty|180)\b", "turn", {"deg": 180, "dir": "left"}),

    # === SPEED CONTROL ===
    (r"\b(?:slow\s+(?:down|mode)|walk\s+slowly|go\s+slow)\b", "set_speed", {"speed": "slow"}),
    (r"\b(?:normal\s+speed|regular\s+speed|default\s+speed)\b", "set_speed", {"speed": "normal"}),
    (r"\b(?:fast\s+(?:mode)?|walk\s+fast|go\s+fast|speed\s+up)\b", "set_speed", {"speed": "fast"}),

    # === VISUAL NAVIGATION ===
    (r"\b(?:find|look for|search for|locate)\s+(?:the\s+|a\s+|my\s+)?(?P<desc>.+?)(?:\s*\.|$)", "go_to_object", {"description": "$desc"}),
    (r"\b(?:follow me|come with me|tag along)\b",  "follow_me", {}),

    # === NAVIGATION ===
    (r"\b(?:go to|navigate to|drive to|walk to)\s+(?P<location>[\w\s]+?)(?:\s*\.|$|\s+stop|\s+halt|\s+freeze)",  "go_to", {"location": "$location"}),
    (r"\b(?:loop|tour)\s+(?:the\s+)?map\b",        "tour", {"locations": "all"}),
    (r"\bvisit\s+all\s+(?:locations|waypoints|places)\b", "tour", {"locations": "all"}),
    (r"\bpatrol\s+(?:the\s+)?map\b",               "patrol", {"locations": "all"}),
    (r"\b(?:keep\s+)?patrol(?:ling)?\b",            "patrol", {"locations": "all"}),
    (r"\b(?:come\s+back|go\s+home|return\s+(?:to\s+)?(?:start|home|base))\b", "come_back", {}),
    (r"\b(?:save location|remember location|save this as|mark this as)\s+(?P<location>[\w\s]+?)(?:\s*\.|$|\s+stop|\s+halt|\s+freeze)", "save_location", {"location": "$location"}),
    (r"\b(?:come here|come to me)\b",              "walk_to", {"relative": [0.0, -1.0, 0.0]}),

    # === STATUS & POWER ===
    (r"\b(?:battery|battery status|how much battery|charge level|power level)\b", "battery_status", {}),
    (r"\b(?:status|status check|how are you|robot status)\b", "status", {}),
    (r"\b(?:power off|shut\s*down|turn off)\b",    "power_off", {}),

    # === OTHER ===
    (r"\b(?:list locations|what locations|show locations|saved locations)\b", "list_locations", {}),

    # === MAP MANAGEMENT ===
    # Order: most specific first. The parser stops on the first match.
    # 1) "load the jackson map" / "load thayer map" / "load the lab_map"
    (r"\bload\s+(?:the\s+)?(?P<map>\w+)\s+map\b", "load_map", {"map": "$map"}),
    # 2) "load map jackson"
    (r"\bload\s+map\s+(?P<map>\w+)\b", "load_map", {"map": "$map"}),
    # 3) Bare "load map" — negative lookahead so it does NOT match "load map jackson".
    #    Empty params dict → handler defaults to last-used map.
    (r"\bload\s+map\b(?!\s+\w)", "load_map", {}),
    # 4) "list maps" / "show maps" / "what are the available maps" / etc.
    (r"\b(?:list|show|what)\s+(?:are\s+)?(?:the\s+)?(?:available\s+)?maps\b", "list_maps", {}),
]


# The hand-written sets at base.
LOCOMOTION_INTENTS = frozenset({
    "walk_to", "turn", "walk", "strafe", "go_to", "tour", "patrol",
    "come_back", "go_to_object", "follow_me",
})
SAFETY_INTENTS = frozenset({"stop", "freeze", "estop"})
SHUTDOWN_INTENTS = frozenset({"power_off", "shutdown", "sit"})
ABORTING_INTENTS = SAFETY_INTENTS | SHUTDOWN_INTENTS | {"stow"}
_UNLOCKED_INTENTS = SAFETY_INTENTS | frozenset({
    "set_persona", "add_persona", "set_volume", "set_backend",
    "battery_status", "status", "check_obstacles", "list_locations", "list_maps",
})
INFO_INTENTS = frozenset({"battery_status", "status", "check_obstacles",
                          "list_locations", "list_maps"})
SHORT_PATH_BLOCKED_INTENTS = frozenset({"walk", "turn", "walk_to", "follow_me", "patrol",
                                        "come_back", "go_to_object", "power_off"})
_PERSONA_INTENTS = ("set_persona", "add_persona")
_MOTION_FREE_INTENTS = frozenset({
    "describe", "stop", "freeze", "estop",
    "set_persona", "add_persona", "set_volume", "set_backend",
    "battery_status", "status", "check_obstacles", "list_locations", "list_maps",
})
_STATE_ONLY_INTENTS = {"set_persona", "add_persona", "set_volume", "set_backend"}

# src/spot/brain/llm_brain.py _EMOTE_ACTION_LINE at base, verbatim.
EMOTE_ACTION_LINE = (
    '- emote: Perform a fun built-in routine (dance, bow, wag, etc.). '
    'Params: {"name": "<routine name>"}. Use only a routine name from the '
    'EMOTES list below.\n'
)
