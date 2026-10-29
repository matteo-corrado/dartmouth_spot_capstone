"""Go-to phrase parsing: is a description a person, and which detector label is it?

Pure text helpers (no robot, no models). The heavy modules (people, vocab) are
imported lazily so the dispatcher can call these without loading the package's
GPU/VLM dependencies at import time.
"""
from __future__ import annotations

import re

PERSON_RE = re.compile(
    r"^(?:me|myself|us)$"
    r"|\b(?:persons?|people|someone|somebody|anyone|anybody|humans?|man|woman|men|women|"
    r"guys?|girls?|boys?|lady|ladies|gentleman|gentlemen|kids?|child|children|folks?|"
    r"wearing|dressed)\b"
    r"|\bone (?:in|on|with|wearing|at|to|behind|dressed)\b")
CLOTHES_RE = re.compile(
    r"\b(?:shirt|t-shirt|tshirt|jacket|hoodie|sweater|jumper|coat|dress|hat|cap|jeans|"
    r"pants|trousers|shorts|skirt|top|scarf|glasses|uniform|vest|suit|blouse)s?\b")
LABEL_SPLIT_RE = re.compile(r"\b(?:on|in|at|by|near|next to|beside|behind|under|with|over)\b")


def clean_phrase(text) -> str:
    text = re.sub(r"[^\w\s-]", " ", str(text or "").lower())
    return re.sub(r"\s+", " ", text).strip()


def is_person_phrase(description) -> bool:
    """"me", "person in red shirt", "the kids", "the one on the left" -> True; objects -> False."""
    text = clean_phrase(description)
    if PERSON_RE.search(text):
        return True
    from .people import parse_person_description
    clothing, _side = parse_person_description(text)
    return bool(clothing and CLOTHES_RE.search(clothing))


def object_label(description):
    """("coco"|"yoloe", label) for a go-to object phrase, or None: the whole
    phrase, then the part before a preposition and its shorter tails
    ("red chair" -> chair, "the cup on the floor" -> cup)."""
    from .vocab import resolve_label
    text = clean_phrase(description)
    words = LABEL_SPLIT_RE.split(text, maxsplit=1)[0].split()
    for cand in [text] + [" ".join(words[i:]) for i in range(len(words))]:
        hit = resolve_label(cand) if cand else None
        if hit:
            return hit
    return None


def wants_person(description) -> bool:
    """True if a go-to description must take the person path (R17/R19): a
    person phrase, or any phrase whose detector label resolves to ``person``."""
    if is_person_phrase(description):
        return True
    hit = object_label(description)
    return bool(hit and hit[1] == "person")
