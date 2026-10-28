"""Wake phrase list, fuzzy wake-phrase stripping, and peak-safe KWS gain."""

import difflib
import pathlib
import re

import numpy as np

_PHRASES_FILE = pathlib.Path(__file__).resolve().parents[3] / "config" / "wake_phrases.txt"
_DEFAULT = ["HEY SPOT"]
_FUZZY_RATIO = 0.75
_FILLERS = {"hey", "hi", "a", "say", "hay", "hello", "stay", "heh"}  # ASR variants of a leading "hey"
_WORD = re.compile(r"[A-Za-z0-9']+")

MAX_GAIN = 8.0


def load_wake_phrases(path=_PHRASES_FILE) -> list:
    """One phrase per line; blanks and '#' comments ignored; default HEY SPOT."""
    try:
        lines = pathlib.Path(path).read_text().splitlines()
    except OSError:
        return list(_DEFAULT)
    phrases = [ln.strip().upper() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    return phrases or list(_DEFAULT)


WAKE_PHRASES = load_wake_phrases()


def _close(word: str, target: str) -> bool:
    return difflib.SequenceMatcher(None, word, target).ratio() >= _FUZZY_RATIO


def _match_at(words, start, target) -> bool:
    """True if words[start:start+len(target)] fuzzy-matches the target words."""
    if start + len(target) > len(words):
        return False
    for i, tw in enumerate(target):
        w = words[start + i]
        if i == 0 and tw in ("hey", "hi") and w in _FILLERS:
            continue
        if not _close(w, tw):
            return False
    return True


def strip_wake_phrase(text: str, phrases: list) -> str:
    """Remove a leading wake phrase, tolerating ASR variants.

    Allows one optional leading filler (hey/hi/a/say) and a fuzzy match per
    word (difflib ratio >= 0.75). Only strips at the start of the text.
    """
    spans = [(m.group().lower(), m.end()) for m in _WORD.finditer(text)]
    words = [w for w, _ in spans]
    targets = sorted((p.lower().split() for p in phrases if p.strip()), key=len, reverse=True)
    for target in targets:
        for start in (0, 1):
            if start == 1 and not (words and words[0] in _FILLERS):
                continue
            if _match_at(words, start, target):
                end = spans[start + len(target) - 1][1]
                return text[end:].lstrip(" \t,.!?;:-")
    return text


def peak_safe_gain(frame: np.ndarray, target_peak: float) -> float:
    """Gain bringing the frame peak to target_peak, capped at MAX_GAIN."""
    peak = float(np.max(np.abs(frame))) if len(frame) else 0.0
    return min(MAX_GAIN, target_peak / max(peak, 1e-6))
