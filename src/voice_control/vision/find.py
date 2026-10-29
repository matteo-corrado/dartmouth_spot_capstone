"""'Do you see X?' - detector first, VLM yes/no as the fallback."""
from __future__ import annotations

import logging

from .cameras import BODY_SOURCES, HAND_SOURCE, capture, distance_words, sector
from .describe import FRONT_SOURCES, front_images, _article
from .types import ActionResult
from .vlm import ask_vlm
from .vocab import resolve_label
from .world import snapshot

logger = logging.getLogger(__name__)

YES_LINE = "Yes, there's {art} {label} {where}{dist}."
NEAR_DIST = ", right next to me"
FAR_DIST = ", {words} away"
NO_LINE = "I don't see one right now."
MAYBE_LINE = "I think I see {art} {phrase} in front of me, but I'm not certain."
FIND_PROMPT = "Answer yes or no first: do you see a {phrase}?"


def _vlm_check(robot, phrase: str) -> ActionResult:
    frames = capture(robot, FRONT_SOURCES + [HAND_SOURCE])
    images = front_images(frames)
    answer = ask_vlm(FIND_PROMPT.format(phrase=phrase), images, max_tokens=40) if images else None
    if answer and answer.strip().lower().lstrip("\"' ").startswith("yes"):
        return ActionResult(MAYBE_LINE.format(art=_article(phrase), phrase=phrase), True,
                            {"detections": [], "uncertain": True})
    return ActionResult(NO_LINE, False, {"detections": []})


def find(robot, phrase: str) -> ActionResult:
    hit = resolve_label(phrase)
    if hit is not None:
        kind, name = hit
        model = "yoloe-26l-seg" if kind == "yoloe" else "yolo26l-seg"
        try:
            dets = snapshot(robot, list(BODY_SOURCES) + [HAND_SOURCE], model=model,
                            labels={name}, with_depth=True)
        except Exception as e:
            logger.warning("find: detector unavailable (%s: %s)", type(e).__name__, e)
            dets = []
        dets = sorted((d for d in dets if d.label == name), key=lambda d: d.conf, reverse=True)
        if dets:
            best = dets[0]
            if best.distance_m is None:
                dist = ""
            elif best.distance_m < 1.0:
                dist = NEAR_DIST
            else:
                dist = FAR_DIST.format(words=distance_words(best.distance_m))
            text = YES_LINE.format(art=_article(name), label=name,
                                   where=sector(best.bearing_deg), dist=dist)
            return ActionResult(text, True, {"detections": dets})
    return _vlm_check(robot, phrase)
