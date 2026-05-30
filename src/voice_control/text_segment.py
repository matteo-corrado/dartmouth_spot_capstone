# src/voice_control/text_segment.py
"""Abbreviation-safe sentence segmentation for streaming TTS (Stage 2F P1).

Swaps the old `[.!?]\\s` regex. Uses pysbd for detection (blingfire has no
working aarch64 build on this Jetson), guards against known abbreviation
false-splits, and falls back to a clause split when a buffer grows long with
no sentence end so streaming latency stays bounded.

Pure + synchronous: returns (complete_sentences, remainder). The chunker keeps
`remainder` buffered until more tokens arrive.
"""
from __future__ import annotations
import re

# Backend choice (eval 2026-05-30): pysbd beats the alternatives for per-char
# streaming on the pinned aarch64 Jetson venv — blingfire (no aarch64 build),
# nltk-punkt (splits "e.g."), syntok / sentence-splitter (compiled regex dep),
# spaCy (263 MB, 28 deps, numpy-2 pull), wtpsplit/SaT (11-13 ms/call, not viable).
# pysbd is pure-Python, zero-dep, MIT, sub-ms, and passes all abbreviation cases.
_BACKEND = "pysbd"

# Lowercased tokens that, when they immediately precede a split point, indicate
# a false boundary. Compared against the last whitespace-delimited token of a
# candidate sentence, stripped of the trailing period.
_ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "inc", "ltd", "co", "e.g", "i.e", "u.s", "u.k", "a.m", "p.m",
    "no", "fig", "approx", "dept", "gen", "lt", "col", "sgt",
}
_CLAUSE_MAX = 80
_CLAUSE_RE = re.compile(r"[,;:]\s")


def _raw_split(text: str) -> list[str]:
    if _BACKEND == "pysbd":
        import pysbd
        return [s.strip() for s in pysbd.Segmenter(language="en", clean=False).segment(text) if s.strip()]
    import blingfire
    out = blingfire.text_to_sentences(text)
    return [s.strip() for s in out.split("\n") if s.strip()]


def _last_token(sentence: str) -> str:
    toks = sentence.split()
    if not toks:
        return ""
    return toks[-1].rstrip(".").lower()


def _merge_abbreviation_falsesplits(parts: list[str]) -> list[str]:
    """Join a part back to the next when it ends in a known abbreviation."""
    merged: list[str] = []
    i = 0
    while i < len(parts):
        cur = parts[i]
        while (i + 1 < len(parts)
               and cur.endswith(".")
               and _last_token(cur) in _ABBREVIATIONS):
            cur = cur + " " + parts[i + 1]
            i += 1
        merged.append(cur)
        i += 1
    return merged


def split_sentences(text: str) -> tuple[list[str], str]:
    """Return (complete_sentences, remainder).

    A sentence is 'complete' only when followed by more text — the final
    segment is always returned as `remainder` (it may still be growing). If a
    single unpunctuated segment exceeds _CLAUSE_MAX chars, emit up to the last
    clause boundary so streaming latency is bounded.

    Pure + synchronous: returns (complete_sentences, remainder). The chunker
    keeps `remainder` buffered until more tokens arrive.
    """
    text = text.strip()
    if not text:
        return [], ""

    parts = _merge_abbreviation_falsesplits(_raw_split(text))

    # Whether the input ends with terminal punctuation decides if the last part
    # is complete or a still-growing remainder.
    ends_terminal = bool(re.search(r"[.!?][\"')\]]?$", text))
    if len(parts) <= 1 and not ends_terminal:
        # No sentence boundary yet — try the clause fallback for long buffers.
        if len(text) > _CLAUSE_MAX:
            m = list(_CLAUSE_RE.finditer(text))
            if m:
                cut = m[-1].end()
                return [text[:cut].strip()], text[cut:].strip()
        return [], text

    if ends_terminal:
        return parts, ""
    # Last part is still growing — keep it as remainder.
    return parts[:-1], parts[-1]
