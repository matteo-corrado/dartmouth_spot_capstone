"""Speaker localisation: refine the mic-array DoA bearing with camera person detections."""
from __future__ import annotations

import logging
from typing import Callable

from .runtime import load_model
from .world import _angle_diff, cameras_covering, snapshot

logger = logging.getLogger(__name__)

_getter: Callable[[], float | None] | None = None


def set_speaker_bearing_getter(fn: Callable[[], float | None]) -> None:
    global _getter
    _getter = fn


def speaker_bearing() -> float | None:
    """Current speaker bearing from the registered getter; None if unset or it raises."""
    if _getter is None:
        return None
    try:
        return _getter()
    except Exception:
        logger.debug("speaker bearing getter failed", exc_info=True)
        return None


def _gpu_ok() -> bool:
    try:
        return load_model("yolo26l-seg").provider in ("trt", "cuda")
    except Exception:
        return False


def refine_speaker_bearing(robot, doa_deg: float | None, window_deg: float = 30.0) -> float | None:
    """Snap ``doa_deg`` to the nearest detected person within the window; else return it unchanged."""
    if doa_deg is None:
        return None
    try:
        if not _gpu_ok():
            return doa_deg
        dets = snapshot(robot, cameras_covering(doa_deg), labels={"person"}, with_depth=False)
        near = [(round(_angle_diff(d.bearing_deg, doa_deg), 6), -d.conf, d.bearing_deg)
                for d in dets if d.label == "person"
                and _angle_diff(d.bearing_deg, doa_deg) <= window_deg]
        return min(near)[2] if near else doa_deg
    except Exception:
        logger.warning("speaker refinement failed", exc_info=True)
        return doa_deg
