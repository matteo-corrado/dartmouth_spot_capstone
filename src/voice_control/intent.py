"""Intent parser: map free-text to structured robot intents.

This module provides a small rules-based parser. Improvements made here:
- more flexible regexes (case-insensitive, named groups, optional units)
- precompiled patterns for performance
- automatic numeric conversion (int/float) and validation (clamp degrees)
- supports callable params for more advanced extraction
"""

import re
from typing import Any, Dict, List, Tuple, Pattern

# Tuples are (pattern, intent_name, params_template)
# params_template values can be:
# - literal values (kept as-is)
# - strings of the form "$name" to substitute a named capture group
# - callables(match) to compute a value from the match
# Safety commands (stop / freeze / estop) are NOT in this table: they go
# through check_safety_command() below, the single source of truth that
# client_mic also uses for its zero-latency bypass.
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

# Safety commands that bypass the LLM for zero-latency execution.
# ORDER MATTERS: estop must be checked before stop so "e stop" / "e-stop"
# matches the more-specific estop pattern, not the bare stop pattern.
SAFETY_PATTERNS = [
    (re.compile(r"emergency\W+stop|\be\W?stop\b|\bestop\b", re.IGNORECASE), "estop"),
    (re.compile(r"\b(?:stop|halt)\b", re.IGNORECASE), "stop"),
    (re.compile(r"\bfreeze\b", re.IGNORECASE), "freeze"),
]

_ESTOP_RE = SAFETY_PATTERNS[0][0]
_HALT_RE = re.compile(r"\bhalt\b", re.IGNORECASE)
_FREEZE_RE = SAFETY_PATTERNS[2][0]
# "stop" only as a whole word: "stops" / "stopped" / "stopping" don't match.
_STOP_WORD_RE = re.compile(r"\bstop\b", re.IGNORECASE)
_PREV_WORD_RE = re.compile(r"(\w+)(\W*)$")
# Only unambiguous noun preceders ("the next stop", "bus stop"). Articles
# etc. are deliberately absent: "come to a stop" / "uh, stop" (ASR "a stop")
# are real commands.
_STOP_NOUN_PRECEDERS = frozenset({
    "next", "last", "bus", "pit", "first", "final", "another",
})
_SENTENCE_PUNCT = set(".,!?;:")
_DASHES = str.maketrans({"–": "-", "—": "-", "‑": "-"})


def _has_command_stop(text: str) -> bool:
    """True if any whole-word "stop" is NOT a noun use. A "stop" at the start
    of the text or after sentence punctuation ("next. Stop!") is never exempt."""
    for m in _STOP_WORD_RE.finditer(text):
        prev = _PREV_WORD_RE.search(text[:m.start()])
        if (prev is None
                or _SENTENCE_PUNCT.intersection(prev.group(2))
                or prev.group(1).lower() not in _STOP_NOUN_PRECEDERS):
            return True
    return False


def check_safety_command(text: str):
    """Return {"intent", "params", "raw"} for a safety command, else None.

    Errs toward stopping ("fire unless benign"):
    - estop ("emergency stop", "e-stop", "estop") always fires and wins.
    - halt (-> stop) and freeze fire anywhere.
    - stop fires anywhere unless every "stop" is a noun use ("the next stop",
      "bus stop"), so "what's the next stop on the tour" doesn't halt Spot.
    Never raises; None for empty / non-text input.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    t = text.translate(_DASHES)
    if _ESTOP_RE.search(t):
        intent_name = "estop"
    elif _HALT_RE.search(t) or _has_command_stop(t):
        intent_name = "stop"
    elif _FREEZE_RE.search(t):
        intent_name = "freeze"
    else:
        return None
    return {"intent": intent_name, "params": {}, "raw": text}


FILLER_WORDS = frozenset({"uh", "um", "er", "ah", "hmm", "like", "so", "okay"})
_WORD_RE = re.compile(r"[a-z0-9']+")


def is_filler_only(text: str) -> bool:
    """True only if no word remains after removing filler words (one-word
    commands like "sit" or "hello" still go through)."""
    words = [w for w in _WORD_RE.findall((text or "").lower()) if w not in FILLER_WORDS]
    return len(words) == 0


# Precompile with IGNORECASE for more robust matching
COMMAND_PATTERNS: List[Tuple[Pattern, str, Dict[str, Any]]] = [
    (re.compile(pat, re.IGNORECASE), intent, params) for pat, intent, params in COMMANDS
]


def _convert_numeric(s: Any):
    """Convert a string number to int or float when appropriate.

    Returns the original value if conversion isn't possible.
    """
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return s
    s = str(s).strip()
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    if re.fullmatch(r"[+-]?\d*\.\d+", s):
        return float(s)
    # fallback: return original string
    return s


def _resolve_param(value, match: re.Match):
    """Resolve a param template against a regex match.

    - If value is callable, call it with the match and return result.
    - If value is a string starting with '$', treat the rest as a group name
      and return that group's contents (converted to number when appropriate).
    - If value is a list/dict, resolve recursively.
    - Otherwise return value as-is.
    """
    if callable(value):
        return value(match)
    if isinstance(value, str) and value.startswith("$"):
        key = value[1:]
        # prefer named group, fall back to numeric index if provided
        grp = None
        if key.isdigit():
            try:
                grp = match.group(int(key))
            except Exception:
                grp = None
        else:
            try:
                grp = match.group(key)
            except Exception:
                grp = None
        return _convert_numeric(grp)
    if isinstance(value, dict):
        return {k: _resolve_param(v, match) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_param(v, match) for v in value]
    return value


def parse_intent(text: str):
    """Parse `text` and return a structured intent or None.

    Returned shape on match: {"intent": str, "params": dict, "raw": original_text}
    """
    if not text:
        return None
    # Safety check runs on the ORIGINAL text: punctuation matters ("next. Stop!").
    safety = check_safety_command(text)
    if safety is not None:
        return safety
    # Clean up Whisper punctuation artifacts (e.g. "Go to. Corner." → "Go to Corner")
    t = re.sub(r'\.\s*', ' ', text).strip()
    t = re.sub(r'\s+', ' ', t)
    for pat, intent_name, params in COMMAND_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        out_params: Dict[str, Any] = {}
        for k, v in params.items():
            out_params[k] = _resolve_param(v, m)

        # Post-processing: clamp/normalize degrees when present
        if "deg" in out_params and isinstance(out_params["deg"], (int, float)):
            deg = out_params["deg"]
            # clamp to [0, 360]
            deg = max(0, min(360, deg))
            # make integral degrees ints
            if isinstance(deg, float) and deg.is_integer():
                deg = int(deg)
            out_params["deg"] = deg
        
        # Post-processing: normalize location names (strip, lowercase, replace spaces with underscores)
        if "location" in out_params and isinstance(out_params["location"], str):
            location = out_params["location"].strip().lower()
            # Replace spaces with underscores for consistency
            location = re.sub(r'\s+', '_', location)
            out_params["location"] = location

        return {"intent": intent_name, "params": out_params, "raw": text}

    return None

