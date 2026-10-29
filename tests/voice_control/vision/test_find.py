from unittest import mock

import numpy as np

from src.voice_control.vision import describe as D
from src.voice_control.vision import find as F
from src.voice_control.vision.types import Detection

from .test_describe import _capture


def _det(label, bearing, dist):
    return Detection(label, 0.9, "x", (0, 0, 1, 1), None, None, bearing, dist, "depth")


def _find(phrase, dets, vlm_text="No."):
    with mock.patch.object(F, "snapshot", return_value=dets), \
         mock.patch.object(F, "capture", _capture()), \
         mock.patch.object(D, "stitch_front", lambda a, b: np.zeros((8, 8), np.uint8)), \
         mock.patch.object(F, "ask_vlm", return_value=vlm_text) as vlm:
        return F.find(None, phrase), vlm


def test_find_coco_hit_says_direction_and_distance():
    res, vlm = _find("a backpack", [_det("backpack", 170, 2.1)])
    assert res.text == "Yes, there's a backpack behind me, about two metres away."
    assert res.ok and len(res.data["detections"]) == 1 and vlm.call_count == 0


def test_find_close_object():
    res, _ = _find("backpack", [_det("backpack", 0, 0.5)])
    assert res.text == "Yes, there's a backpack in front, right next to me."


def test_find_falls_back_to_vlm_no():
    res, vlm = _find("backpack", [])
    assert res.text == "I don't see one right now." and vlm.call_count == 1


def test_find_vlm_none_is_no():
    res, _ = _find("backpack", [], vlm_text=None)
    assert res.text == "I don't see one right now."


def test_find_vlm_yes_is_uncertain():
    res, _ = _find("zorblax", [], vlm_text="Yes, I do.")
    assert res.text == "I think I see a zorblax in front of me, but I'm not certain."
