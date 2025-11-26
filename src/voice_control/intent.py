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
COMMANDS = [
    (r"\b(?:stop|halt)\b",                          "stop", {}),
    (r"\bfreeze\b",                                 "freeze", {}),
    (r"\b(?:stand|stand up|get up)\b",             "stand", {}),
    (r"\b(?:sit|sit down)\b",                      "sit", {}),
    (r"\b(?:go to|navigate to|drive to)\s+(?P<location>\w+)\b",  "go_to", {"location": "$location"}),
    (r"\b(?:save location|remember location|save this as)\s+(?P<location>\w+)\b", "save_location", {"location": "$location"}),
    (r"\b(?:follow me|follow)\b",                 "follow", {}),
    (r"\b(?:come here|come to me)\b",             "walk_to", {"relative": [0.0, -1.0, 0.0]}),
    (r"\bturn\s+left\s+(?P<deg>\d+(?:\.\d+)?)\s*(?:deg(?:rees)?)?\b",  "turn", {"deg": "$deg", "dir": "left"}),
    (r"\bturn\s+right\s+(?P<deg>\d+(?:\.\d+)?)\s*(?:deg(?:rees)?)?\b", "turn", {"deg": "$deg", "dir": "right"}),
    (r"\b(?:look at me|look here)\b",             "ptz_aim", {"target": "speaker"}),
]

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
    t = text.strip()
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

        return {"intent": intent_name, "params": out_params, "raw": text}

    return None

