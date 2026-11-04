"""Intent parser: map free-text to structured robot intents.

This module provides a small rules-based parser. Improvements made here:
- more flexible regexes (case-insensitive, named groups, optional units)
- precompiled patterns for performance
- automatic numeric conversion (int/float) and validation (clamp degrees)
- supports callable params for more advanced extraction
"""

import re
from typing import Any, Dict, List, Tuple, Pattern

# The regex table is generated (Task 16): core rows (src/spot/core/core_intents.py)
# plus each enabled feature's IntentRules, by priority (priority // 10 is the
# old COMMANDS index). A rule's build(match) gives {"intent", "params"}; param
# templates are resolved by _resolve_param below:
# - literal values (kept as-is)
# - strings of the form "$name" to substitute a named capture group
# - callables(match) to compute a value from the match
# Safety commands (stop / freeze / estop) are NOT in this table: they go
# through check_safety_command() below, the single source of truth that
# client_mic also uses for its zero-latency bypass.

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


# Answers, not filler, once Spot is talking with someone ("Want to see a dance?" "Okay.")
_CONVERSATION_ACKS = frozenset({"okay", "ok"})


def is_filler_only(text: str, mid_conversation: bool = False) -> bool:
    """True only if no word remains after removing filler words (one-word
    commands like "sit" or "hello" still go through). mid_conversation=True
    keeps "okay" as a real answer."""
    filler = FILLER_WORDS - _CONVERSATION_ACKS if mid_conversation else FILLER_WORDS
    words = [w for w in _WORD_RE.findall((text or "").lower()) if w not in filler]
    return len(words) == 0


_compiled_cache: Dict[Any, List[Tuple[Pattern, Any]]] = {}


def command_patterns(registry=None) -> List[Tuple[Pattern, Any]]:
    """[(compiled pattern, IntentRule)] for the current registry, in priority
    order. Compiled with IGNORECASE once per registry/rule set."""
    from src.spot.core.core_intents import intent_rules
    rules = intent_rules(registry)
    key = tuple(id(r) for r in rules)
    hit = _compiled_cache.get(key)
    if hit is None:
        hit = [(re.compile(r.pattern, re.IGNORECASE), r) for r in rules]
        _compiled_cache.clear()  # only the live table is kept
        _compiled_cache[key] = hit
    return hit


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
    for pat, rule in command_patterns():
        m = pat.search(t)
        if not m:
            continue
        built = rule.build(m)
        intent_name = built["intent"]
        out_params: Dict[str, Any] = dict(built["params"])

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


# Short-utterance path (client_mic._finish_utterance): a short utterance that
# fails the loudness gate still goes to ASR, but only a safety command, a
# direct robot command or (mid-conversation) a one-word answer is let through.
# Commands the regex table doesn't cover but the brain handles in one word:
SHORT_COMMAND_WORDS = frozenset({"dance", "stow", "wave", "bow", "nod", "spin",
                                 "sit", "stand", "stop", "freeze"})
# Quiet short utterances may not start travel or power off: these need a
# normal-volume utterance that passes the loudness gate (safety review I-2).
# The blocked set is generated: intent_names("short_path_blocked").
# No "thank you"/"you": those are classic ASR hallucinations on noise.
SHORT_ANSWER_WORDS = frozenset({"yes", "yeah", "yep", "yup", "no", "nope", "okay", "ok", "sure"})


def short_utterance_ok(text: str, in_conversation: bool = False) -> bool:
    """True if a short, quiet utterance's transcript is worth acting on:
    a safety command, a regex-known command, a known one-word command, or
    (only in an active LISTENING conversation) a short yes/no/okay answer."""
    if not text or not text.strip():
        return False
    if check_safety_command(text) is not None:
        return True
    parsed = parse_intent(text)
    from src.spot.core.core_intents import intent_names
    if parsed is not None and parsed.get("intent") not in intent_names("short_path_blocked"):
        return True
    words = _WORD_RE.findall(text.lower())
    if not words:
        return False
    if len(words) <= 3 and any(w in SHORT_COMMAND_WORDS for w in words):
        return True
    return in_conversation and all(w in SHORT_ANSWER_WORDS for w in words)

