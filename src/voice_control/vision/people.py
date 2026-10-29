"""Person selection: who does "come to me", "the person in red" or "the one on the left" mean?

Follow-ups are single-turn: when Spot needs more information it returns
``(None, message)`` and the caller speaks the message and ends the action. The
one exception is the raised-hand question, which is spoken here and then
watched for with the pose model for up to 5 s.

No motion is commanded from this module; ``gaze`` and ``speak`` are injected.
"""
from __future__ import annotations

import logging
import math
import re
import time
from typing import Callable

import numpy as np

from .cameras import BODY_SOURCES, HAND_SOURCE, capture, sector
from .speaker import speaker_bearing
from .types import Detection
from .vlm import ask_vlm, encode_jpeg
from .world import _angle_diff, dedupe, detect_frames, snapshot

logger = logging.getLogger(__name__)

POSE_MODEL = "yolo26l-pose"
SPEAKER_WINDOW_DEG = 30.0
TRACK_TOL_DEG = 10.0
GAZE_LIMIT_DEG = 60.0      # gaze_at_bearing shoulder-yaw clamp; candidates beyond it are skipped
MAX_CLOTHING_CANDIDATES = 3
GAZE_SETTLE_S = 2.5        # gaze path is rate-limited (~2 s); let the arm arrive before capturing

# COCO keypoint indices.
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 5, 6, 9, 10

# Spoken lines (never stop/halt/freeze/e-stop).
ASK_HAND = "Can you raise your hand?"
MSG_FEW_PEOPLE = "I can see a few people. Tell me what you're wearing and I'll come to you."
MSG_CANT_SEE_YOU = "I can't see you. Could you step in front of me?"
MSG_NOT_SURE = "I'm not sure who you mean."
MSG_NO_MATCH = "I can't see anyone who matches."
MSG_WHICH_SIDE = "Do you mean the person {a} or {b}?"
TEMPLATES = (ASK_HAND, MSG_FEW_PEOPLE, MSG_CANT_SEE_YOU, MSG_NOT_SURE, MSG_NO_MATCH,
             MSG_WHICH_SIDE)

# How a sector reads after "the person ...".
_WHERE = {"in front": "in front of me", "behind me": "behind me",
          "on my left": "on my left", "on my right": "on my right"}


def _where(sec: str) -> str:
    return _WHERE.get(sec, f"to my {sec}")

WEARING_PROMPT = "Answer yes or no: is this person wearing {clothing}?"
YES_NO_SYSTEM = (
    "You are Spot, a robot, looking at a person through your own camera eyes. "
    "Reply with one word: yes or no."
)

# A side phrase covers its own sector plus the two diagonals next to it, because
# people say "on the left" for someone front-left too. Diagonals match exactly.
SIDE_SECTORS: dict[str, frozenset[str]] = {
    "on my left": frozenset({"front-left", "on my left", "behind-left"}),
    "on my right": frozenset({"front-right", "on my right", "behind-right"}),
    "in front": frozenset({"front-left", "in front", "front-right"}),
    "behind me": frozenset({"behind-left", "behind me", "behind-right"}),
}

_SIDE_PATTERNS: tuple[tuple[re.Pattern, Callable[[re.Match], str]], ...] = (
    (re.compile(r"\b(?:(?:in|on|at|to) (?:the|my|your) )?(front|behind|back)[ -](left|right)\b"),
     lambda m: f"{'front' if m.group(1) == 'front' else 'behind'}-{m.group(2)}"),
    (re.compile(r"\bin front(?: of (?:me|you|us|spot))?\b"), lambda m: "in front"),
    (re.compile(r"\b(?:behind(?: (?:me|you|us|spot))?|(?:at|in) the back)\b"), lambda m: "behind me"),
    (re.compile(r"\b(?:(?:on|to|at) (?:the|my|your) )?(left|right)(?: side)?\b"),
     lambda m: f"on my {m.group(1)}"),
)
_CLOTHING_RE = re.compile(r"\b(?:wearing|dressed in|in|with)\s+(?:(?:a|an|the|some)\s+)?(.+)$")


def parse_person_description(phrase: str) -> tuple[str | None, str | None]:
    """``"person in the red shirt on my left"`` -> ``("red shirt", "on my left")``; ``"me"`` -> (None, None)."""
    text = re.sub(r"[^\w\s-]", " ", (phrase or "").lower())
    text = re.sub(r"\s+", " ", text).strip()
    side = None
    for pat, fmt in _SIDE_PATTERNS:
        m = pat.search(text)
        if m:
            side = fmt(m)
            text = (text[:m.start()] + " " + text[m.end():]).strip()
            text = re.sub(r"\s+", " ", text)
            break
    clothing = None
    m = _CLOTHING_RE.search(text)
    if m:
        clothing = re.sub(r"\s+one$", "", m.group(1)).strip() or None
    return clothing, side


def hand_raised(kps: np.ndarray | None, min_conf: float = 0.5) -> bool:
    """Wrist above the same-side shoulder (upright image, smaller y = higher), both keypoints confident."""
    if kps is None:
        return False
    k = np.asarray(kps)
    if k.ndim != 2 or k.shape[0] < 11 or k.shape[1] < 3:
        return False
    for sh, wr in ((L_SHOULDER, L_WRIST), (R_SHOULDER, R_WRIST)):
        if k[sh, 2] >= min_conf and k[wr, 2] >= min_conf and k[wr, 1] < k[sh, 1]:
            return True
    return False


def wait_for_raised_hand(robot, sources, timeout_s: float = 5.0, frames_needed: int = 3, *,
                         accept: Callable[[Detection], bool] | None = None,
                         clock: Callable[[], float] = time.monotonic,
                         sleep: Callable[[float], None] = time.sleep) -> Detection | None:
    """Watch ``sources`` with the pose model; the one person with a hand up for
    ``frames_needed`` consecutive frames, or None (nobody, several, or timeout)."""
    tracks: list[dict] = []  # {"bearing", "streak", "det"}
    start = clock()
    while clock() - start < timeout_s:
        try:
            dets = snapshot(robot, list(sources), model=POSE_MODEL, labels={"person"},
                            with_depth=False)
        except Exception as e:
            logger.warning("raised hand: pose snapshot failed (%s: %s)", type(e).__name__, e)
            sleep(0.1)
            continue
        matched: set[int] = set()
        # Overlapping cameras can see one person twice; keep one detection per person.
        dets = dedupe([d for d in dets if d.label == "person"
                       and (accept is None or accept(d))])
        for d in dets:
            best, best_off = None, TRACK_TOL_DEG
            for i, t in enumerate(tracks):
                off = _angle_diff(t["bearing"], d.bearing_deg)
                if i not in matched and off <= best_off:
                    best, best_off = i, off
            up = hand_raised(d.keypoints)
            if best is None:
                tracks.append({"bearing": d.bearing_deg, "streak": 1 if up else 0, "det": d})
                matched.add(len(tracks) - 1)
            else:
                t = tracks[best]
                t.update(bearing=d.bearing_deg, det=d, streak=t["streak"] + 1 if up else 0)
                matched.add(best)
        for i, t in enumerate(tracks):
            if i not in matched:
                t["streak"] = 0  # unseen this frame: not consecutive any more
        winners = [t for t in tracks if t["streak"] >= frames_needed]
        if len(winners) > 1:
            return None
        if len(winners) == 1:
            return winners[0]["det"]
    return None


def _yes_no(answer: str | None) -> bool | None:
    if not answer:
        return None
    a = re.sub(r"^[\W_]+", "", answer.strip().lower())
    if re.match(r"yes\b", a):
        return True
    if re.match(r"no\b", a):
        return False
    return None


def _gripper_crop(robot) -> bytes | None:
    """JPEG of the person nearest the gripper image centre, or None if no person."""
    frames = capture(robot, [HAND_SOURCE])
    frame = frames.get(HAND_SOURCE)
    if frame is None:
        return None
    dets = [d for d in detect_frames(frames, labels={"person"}) if d.label == "person"]
    if not dets:
        return None
    h, w = frame.image.shape[:2]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    def off(d):
        x0, y0, x1, y1 = d.box
        return math.hypot((x0 + x1) / 2.0 - cx, (y0 + y1) / 2.0 - cy)
    x0, y0, x1, y1 = min(dets, key=off).box
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    if x1 <= x0 or y1 <= y0:
        return None
    return encode_jpeg(frame.image[y0:y1, x0:x1])


def wearing(robot, person: Detection, clothing: str, gaze: Callable[[float], bool], *,
            sleep: Callable[[float], None] = time.sleep) -> bool | None:
    """Aim the gripper camera at ``person`` and ask the VLM whether they wear ``clothing``."""
    if not gaze(person.bearing_deg):
        return None
    sleep(GAZE_SETTLE_S)
    try:
        crop = _gripper_crop(robot)
    except Exception as e:
        logger.warning("wearing: gripper crop failed (%s: %s)", type(e).__name__, e)
        return None
    if crop is None:
        return None
    answer = ask_vlm(WEARING_PROMPT.format(clothing=clothing), [crop],
                     system=YES_NO_SYSTEM, max_tokens=8)
    return _yes_no(answer)


def _people(robot) -> list[Detection]:
    dets = snapshot(robot, list(BODY_SOURCES), labels={"person"})
    return dedupe([d for d in dets if d.label == "person"])


def _sources(dets: list[Detection]) -> list[str]:
    return list(dict.fromkeys(d.source for d in dets))


def _dist_key(d: Detection) -> float:
    return d.distance_m if d.distance_m is not None else math.inf


def _near_any(dets: list[Detection], tol: float) -> Callable[[Detection], bool]:
    return lambda x: any(_angle_diff(x.bearing_deg, d.bearing_deg) <= tol for d in dets)


def _in_side(d: Detection, side: str) -> bool:
    return sector(d.bearing_deg) in SIDE_SECTORS.get(side, frozenset({side}))


def _ask_raised_hand(robot, candidates, speak, clock, sleep, accept) -> Detection | None:
    speak(ASK_HAND)
    return wait_for_raised_hand(robot, _sources(candidates), accept=accept,
                                clock=clock, sleep=sleep)


def select_person(robot, phrase: str, speak: Callable[[str], None],
                  gaze: Callable[[float], bool], *,
                  clock: Callable[[], float] = time.monotonic,
                  sleep: Callable[[float], None] = time.sleep,
                  ) -> tuple[Detection | None, str | None]:
    """Pick the person ``phrase`` refers to: (target, None) or (None, message to speak)."""
    clothing, side = parse_person_description(phrase)
    sb = speaker_bearing()
    everyone = _people(robot)

    if clothing is None and side is None:  # "me"
        if sb is None:
            cands = everyone  # no locked speaker: anyone visible could be them
            accept = None
        else:
            cands = [d for d in everyone if _angle_diff(d.bearing_deg, sb) <= SPEAKER_WINDOW_DEG]
            accept = lambda d: _angle_diff(d.bearing_deg, sb) <= SPEAKER_WINDOW_DEG  # noqa: E731
        if not cands:
            return None, MSG_CANT_SEE_YOU
        if len(cands) == 1:
            return cands[0], None
        hit = _ask_raised_hand(robot, cands, speak, clock, sleep, accept)
        return (hit, None) if hit is not None else (None, MSG_FEW_PEOPLE)

    pool = [d for d in everyone if side is None or _in_side(d, side)]

    if clothing is None:  # side only
        if not pool:
            return None, MSG_NOT_SURE
        pool.sort(key=lambda d: (_dist_key(d), abs(d.bearing_deg)))
        sectors = list(dict.fromkeys(sector(d.bearing_deg) for d in pool))
        if len(sectors) == 1:
            return pool[0], None
        return None, MSG_WHICH_SIDE.format(a=_where(sectors[0]), b=_where(sectors[1]))

    # Clothing (optionally narrowed to a side).
    ref = sb if sb is not None else 0.0
    if not pool:
        return None, MSG_NO_MATCH
    pool.sort(key=lambda d: (_angle_diff(d.bearing_deg, ref), _dist_key(d)))
    cands = [d for d in pool if abs(d.bearing_deg) <= GAZE_LIMIT_DEG][:MAX_CLOTHING_CANDIDATES]
    refused = []

    def gaze_once(bearing: float) -> bool:
        ok = bool(gaze(bearing))
        if not ok:
            refused.append(bearing)
        return ok

    yes = []
    for c in cands:
        if wearing(robot, c, clothing, gaze_once, sleep=sleep):
            yes.append(c)
        if refused:
            break
    if refused:  # arm disabled or the gaze refused: fall back to a raised hand
        hit = _ask_raised_hand(robot, pool, speak, clock, sleep,
                               _near_any(pool, SPEAKER_WINDOW_DEG))
        return (hit, None) if hit is not None else (None, MSG_NOT_SURE)
    if len(yes) == 1:
        return yes[0], None
    return None, MSG_NOT_SURE
