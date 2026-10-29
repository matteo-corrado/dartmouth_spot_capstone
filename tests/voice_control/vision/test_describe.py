import re
from unittest import mock

import numpy as np

import importlib

# importlib: the package re-exports a describe function under the module name.
D = importlib.import_module("src.voice_control.vision.describe")
from src.voice_control.vision.types import CameraFrame, Detection


def _T(pitch_down=False):
    T = np.eye(4)
    # camera z axis = optical axis
    T[:3, 2] = [0, 0, -1] if pitch_down else [1, 0, 0]
    return T


def _frame(src, down=False):
    img = np.zeros((20, 30, 3), np.uint8)
    return CameraFrame(src, img, 20, 30, 0.0, 10, 10, 15, 10, _T(down), None, b"", 0.0)


def _capture(down=False):
    def cap(robot, sources, with_depth=False):
        return {s: _frame(s, down and s == D.HAND_SOURCE) for s in sources}
    return cap


def _det(label, conf, bearing):
    return Detection(label, conf, "x", (0, 0, 1, 1), None, None, bearing, None, None)


def _run(fn, down=False, dets=()):
    with mock.patch.object(D, "capture", _capture(down)), \
         mock.patch.object(D, "stitch_front", lambda a, b: np.zeros((8, 8), np.uint8)), \
         mock.patch.object(D, "detect_frames", lambda f, **k: list(dets)), \
         mock.patch.object(D, "ask_vlm", return_value="A hallway.") as vlm:
        return fn(), vlm


def test_describe_front_sends_stitch_and_gripper():
    res, vlm = _run(lambda: D.describe(None))
    assert vlm.call_count == 1 and len(vlm.call_args[0][1]) == 2
    assert res.ok and res.text.startswith("A hallway.")


def test_describe_front_drops_gripper_when_facing_down():
    res, vlm = _run(lambda: D.describe(None), down=True)
    assert len(vlm.call_args[0][1]) == 1


def test_describe_front_survives_missing_model():
    with mock.patch.object(D, "capture", _capture()), \
         mock.patch.object(D, "stitch_front", lambda a, b: np.zeros((8, 8), np.uint8)), \
         mock.patch.object(D, "detect_frames", side_effect=FileNotFoundError("x")), \
         mock.patch.object(D, "ask_vlm", return_value="A room."):
        res = D.describe(None)
    assert res.ok and D.OFFER_LINE in res.text


def test_also_line_people_first_max_three():
    dets = [_det("chair", .9, 90), _det("backpack", .8, 170), _det("person", .4, 135),
            _det("cup", .3, -90)]
    line = D.also_line(dets)
    assert "cup" not in line
    assert line == ("I can also spot a person behind-left, a chair on my left "
                    "and a backpack behind me.")


def test_also_line_drops_front_sectors():
    dets = [_det("person", .9, 0), _det("cat", .8, 40), _det("dog", .8, -40),
            _det("chair", .5, 90), _det("bed", .4, 180)]
    line = D.also_line(dets)
    assert line == "I can also spot a chair on my left and a bed behind me."
    assert D.also_line([_det("person", .9, 0)]) == D.OFFER_LINE


def test_lower_first():
    assert D._lower_first("A hallway.") == "a hallway."
    assert D._lower_first("I'm in a hall.") == "I'm in a hall."
    assert D._lower_first("I see a door.") == "I see a door."


def test_describe_front_vlm_none_is_honest():
    with mock.patch.object(D, "capture", _capture()), \
         mock.patch.object(D, "stitch_front", lambda a, b: np.zeros((8, 8), np.uint8)), \
         mock.patch.object(D, "detect_frames", lambda f, **k: []), \
         mock.patch.object(D, "ask_vlm", return_value=None):
        res = D.describe(None)
    assert not res.ok and res.text == D.UNCLEAR_LINE


def test_describe_front_missing_camera_and_no_cameras():
    def cap_one(robot, sources, with_depth=False):
        return {s: _frame(s) for s in sources if s == D.FRONT_SOURCES[0]}
    with mock.patch.object(D, "capture", cap_one), \
         mock.patch.object(D, "detect_frames", lambda f, **k: []), \
         mock.patch.object(D, "ask_vlm", return_value="A door.") as vlm:
        res = D.describe(None)
    assert res.ok and len(vlm.call_args[0][1]) == 1
    with mock.patch.object(D, "capture", lambda r, s, with_depth=False: {}), \
         mock.patch.object(D, "ask_vlm") as vlm:
        res = D.describe(None)
    assert not res.ok and res.text == D.NOTHING_LINE and vlm.call_count == 0


def test_also_line_empty_offer():
    assert D.also_line([]) == "I can look around if you like."


def test_describe_side_one_call():
    res, vlm = _run(lambda: D.describe(None, "left"))
    assert vlm.call_count == 1 and len(vlm.call_args[0][1]) == 1


def test_side_prompts_name_the_side():
    for side, words in (("left", "on your left"), ("right", "on your right"),
                        ("back", "behind you")):
        _, vlm = _run(lambda: D.describe(None, side))
        prompt = vlm.call_args[0][0]
        assert words in prompt and "in front" not in prompt, side
    _, vlm = _run(lambda: D.describe(None, "around"))
    prompts = [c[0][0] for c in vlm.call_args_list]
    assert "in front of you" in prompts[0]
    assert [p for p in prompts[1:] if "in front" in p] == []
    _, vlm = _run(lambda: D.describe(None, "left", "is there a door?"))
    assert vlm.call_args[0][0] == "is there a door?"   # the user's question wins


def test_describe_around_four_calls():
    res, vlm = _run(lambda: D.describe(None, "around"))
    assert vlm.call_count == 4
    assert res.text.startswith("In front, a hallway.") and "On my left, " in res.text


def test_templates_have_no_stop_words():
    bad = re.compile(r"\b(stop|halt|freez\w*|e-?stop|estop)\b", re.IGNORECASE)
    for name in ("NOTHING_LINE", "UNCLEAR_LINE", "OFFER_LINE", "ALSO_LINE",
                 "DESCRIBE_PROMPT", "HINT_PROMPT"):
        assert not bad.search(getattr(D, name)), name
    for p in D.SIDE_PROMPTS.values():
        assert not bad.search(p)
    F = importlib.import_module("src.voice_control.vision.find")
    for name in ("YES_LINE", "NEAR_DIST", "FAR_DIST", "NO_LINE", "MAYBE_LINE", "FIND_PROMPT"):
        assert not bad.search(getattr(F, name)), name
