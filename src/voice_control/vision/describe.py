"""Spoken scene description: front (stitch + gripper), a single side, or all around."""
from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .cameras import HAND_SOURCE, capture, sector
from .stitch import stitch_front
from .types import ActionResult, CameraFrame, Detection
from .vlm import ask_vlm, encode_jpeg
from .world import dedupe, detect_frames

logger = logging.getLogger(__name__)

FRONT_SOURCES = ["frontright_fisheye_image", "frontleft_fisheye_image"]
SIDE_SOURCES = {
    "left": "left_fisheye_image",
    "right": "right_fisheye_image",
    "back": "back_fisheye_image",
}
SIDE_LEAD = {
    "front": "In front, ",
    "left": "On my left, ",
    "right": "On my right, ",
    "back": "Behind me, ",
}

NOTHING_LINE = "I can't see anything right now."
UNCLEAR_LINE = "I can't make out much right now."
OFFER_LINE = "I can look around if you like."
ALSO_LINE = "I can also spot {items}."
DESCRIBE_PROMPT = "What do you see in front of you? Describe it in two or three sentences."
HINT_PROMPT = " Things I can pick out: {labels}."

FRONT_SECTORS = ("in front", "front-left", "front-right")  # covered by the stitched view
PITCH_MIN_DEG = -60.0
YAW_MAX_DEG = 60.0


def gripper_faces_forward(frame: CameraFrame) -> bool:
    """Optical axis within +-60 deg yaw of body +x and pitch above -60 deg."""
    axis = np.asarray(frame.body_T_sensor, dtype=float)[:3, 2]
    n = float(np.linalg.norm(axis))
    if n == 0:
        return False
    axis = axis / n
    yaw = math.degrees(math.atan2(axis[1], axis[0]))
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, axis[2]))))
    return abs(yaw) <= YAW_MAX_DEG and pitch > PITCH_MIN_DEG


def article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def also_line(dets: list[Detection], max_items: int = 3) -> str:
    """'I can also spot a backpack behind me and a chair on my left.' or the offer."""
    best: dict[str, Detection] = {}
    for d in sorted(dets, key=lambda d: d.conf, reverse=True):
        best.setdefault(d.label, d)
    best = {k: d for k, d in best.items() if sector(d.bearing_deg) not in FRONT_SECTORS}
    ranked = sorted(best.values(), key=lambda d: (d.label != "person", -d.conf))[:max_items]
    if not ranked:
        return OFFER_LINE
    items = [f"{article(d.label)} {d.label} {sector(d.bearing_deg)}" for d in ranked]
    joined = items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
    return ALSO_LINE.format(items=joined)


def front_images(frames: dict[str, CameraFrame]) -> list[bytes]:
    """JPEG of the stitched front (grey) plus the gripper colour image when it faces forward."""
    fr, fl = frames.get(FRONT_SOURCES[0]), frames.get(FRONT_SOURCES[1])
    images: list[bytes] = []
    if fr is not None and fl is not None:
        images.append(encode_jpeg(stitch_front(fr, fl)))
    elif fr is not None or fl is not None:
        images.append(encode_jpeg((fr or fl).image))
    hand = frames.get(HAND_SOURCE)
    if hand is not None and gripper_faces_forward(hand):
        images.append(encode_jpeg(hand.image))
    return images


def _safe_detect(frames: dict[str, CameraFrame], model: str = "yolo26l-seg") -> list[Detection]:
    try:
        return detect_frames(frames, model=model)
    except Exception as e:
        logger.warning("describe: detection unavailable (%s: %s)", type(e).__name__, e)
        return []


def _others(robot) -> list[Detection]:
    frames = capture(robot, list(SIDE_SOURCES.values()))
    return _safe_detect(frames) if frames else []


def _describe_front(robot, question: str, with_also: bool) -> tuple[str | None, str]:
    frames = capture(robot, FRONT_SOURCES + [HAND_SOURCE])
    images = front_images(frames)
    if not images:
        return None, ""
    pool = ThreadPoolExecutor(max_workers=1) if with_also else None
    also = pool.submit(_others, robot) if pool else None
    try:
        front_dets = dedupe(_safe_detect({k: v for k, v in frames.items() if k in FRONT_SOURCES}))
        prompt = question or DESCRIBE_PROMPT
        if front_dets:
            labels = ", ".join(dict.fromkeys(d.label for d in front_dets))
            prompt += HINT_PROMPT.format(labels=labels)
        text = ask_vlm(prompt, images)
        line = ""
        if also is not None:
            try:
                line = also_line(dedupe(also.result()))
            except Exception as e:
                logger.warning("describe: side detection failed (%s)", e)
                line = OFFER_LINE
    finally:
        if pool:
            pool.shutdown(wait=True)
    return text, line


def _describe_side(robot, side: str, question: str) -> str | None:
    src = SIDE_SOURCES[side]
    frames = capture(robot, [src])
    frame = frames.get(src)
    if frame is None:
        return None
    return ask_vlm(question or DESCRIBE_PROMPT, [encode_jpeg(frame.image)])


_KEEP_CAPS = ("I", "I'm", "I'll", "I've", "I'd")


def _lower_first(s: str) -> str:
    first = s.split(" ", 1)[0].rstrip(",.;:!?")
    if not s or first in _KEEP_CAPS:
        return s
    return s[0].lower() + s[1:]


def describe(robot, side: str = "front", question: str = "") -> ActionResult:
    side = (side or "front").lower()
    if side == "around":
        parts = []
        text, _ = _describe_front(robot, question, with_also=False)
        if text:
            parts.append(SIDE_LEAD["front"] + _lower_first(text))
        for s in ("left", "right", "back"):
            t = _describe_side(robot, s, question)
            if t:
                parts.append(SIDE_LEAD[s] + _lower_first(t))
        if not parts:
            return ActionResult(UNCLEAR_LINE, False)
        return ActionResult(" ".join(parts), True)
    if side in SIDE_SOURCES:
        t = _describe_side(robot, side, question)
        if t is None:
            return ActionResult(UNCLEAR_LINE, False)
        return ActionResult(t, True)
    text, line = _describe_front(robot, question, with_also=True)
    if text is None and not line:
        return ActionResult(NOTHING_LINE, False)
    if not text:
        return ActionResult(UNCLEAR_LINE, False)
    return ActionResult(f"{text} {line}".strip(), True)
