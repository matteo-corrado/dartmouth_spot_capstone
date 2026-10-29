"""Spoken-text guard: Spot must never say stop, halt, freeze or e-stop.

The always-on safety listener hears Spot's own voice, so any word inserted
into a spoken line from a parameter, a detector label or a VLM answer goes
through ``safe_words`` first ("stop sign" -> "road sign"; other such words
are dropped).
"""
from __future__ import annotations

import re

_SIGN_RE = re.compile(r"\bstop[\s-]+sign(s?)\b", re.IGNORECASE)
_BAD_RE = re.compile(r"\b(?:e-?stop\w*|estop\w*|stop\w*|halt\w*|freez\w*|froze\w*)\b", re.IGNORECASE)


def safe_words(text) -> str:
    """``text`` with "stop sign" -> "road sign" and any stop/halt/freeze/e-stop word removed."""
    s = _SIGN_RE.sub(lambda m: "road sign" + m.group(1), str(text or ""))
    s = _BAD_RE.sub("", s)
    s = re.sub(r"([,.;:!?])(?:\s+[,.;:!?])+", r"\1", s)  # "sign. ." left by a dropped word
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r" +([,.;:!?])", r"\1", s)
    return s.strip()
