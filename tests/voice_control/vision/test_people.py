import re

import numpy as np
import pytest

from src.voice_control.vision import people
from src.voice_control.vision.types import CameraFrame, Detection

BAD = re.compile(r"\b(stop|halt|freez\w*|e-?stop|estop)\b", re.IGNORECASE)
FL = "frontleft_fisheye_image"
FR = "frontright_fisheye_image"


def _kps(raised_right=False, raised_left=False, conf=0.9):
    k = np.zeros((17, 3), dtype=float)
    k[:, 2] = conf
    k[5, :2] = (40, 100)   # left shoulder
    k[6, :2] = (60, 100)   # right shoulder
    k[9, :2] = (40, 80 if raised_left else 150)    # left wrist
    k[10, :2] = (60, 80 if raised_right else 150)  # right wrist
    return k


def _p(bearing, dist=2.0, src=FL, conf=0.8, kps=None, box=(0, 0, 10, 10)):
    return Detection("person", conf, src, box, None, kps, bearing, dist, "depth")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


@pytest.fixture
def clk():
    return Clock()


def _snap_seq(monkeypatch, seq, clk=None, step=0.2):
    """seq: dets for the first (candidate) snapshot, then pose frames in order."""
    calls = []
    it = iter(seq)

    def fake(robot, sources=None, model="yolo26l-seg", labels=None, with_depth=True):
        calls.append({"sources": list(sources) if sources else sources, "model": model})
        if clk is not None and model == "yolo26l-pose":
            clk.t += step
        try:
            return list(next(it))
        except StopIteration:
            return []
    monkeypatch.setattr(people, "snapshot", fake)
    return calls


def _bearing(monkeypatch, b):
    monkeypatch.setattr(people, "speaker_bearing", lambda: b)


class Spy:
    def __init__(self, ret=True):
        self.calls = []
        self.ret = ret

    def __call__(self, x):
        self.calls.append(x)
        return self.ret


# ---------------------------------------------------------------- hand_raised

def test_hand_raised_rules():
    assert people.hand_raised(_kps(raised_right=True))
    assert people.hand_raised(_kps(raised_left=True))
    assert not people.hand_raised(_kps())
    assert not people.hand_raised(_kps(raised_right=True, conf=0.4))  # conf below 0.5
    assert not people.hand_raised(None)


def test_hand_raised_opposite_shoulder_only_is_false():
    k = _kps()
    k[6, 1] = 60    # right shoulder high
    k[5, 1] = 120   # left shoulder low
    k[10, 1] = 100  # right wrist: above left shoulder (120) but below right shoulder (60)
    k[9, 1] = 150
    assert not people.hand_raised(k)


def test_hand_raised_low_conf_on_one_side_uses_other():
    k = _kps(raised_left=True)
    k[10, 2] = 0.1  # right wrist unreliable
    assert people.hand_raised(k)
    k[9, 2] = 0.1   # left wrist unreliable too
    assert not people.hand_raised(k)


# ---------------------------------------------------------------- parse

@pytest.mark.parametrize("phrase,expected", [
    ("person in the red shirt on my left", ("red shirt", "on my left")),
    ("me", (None, None)),
    ("the one on the right", (None, "on my right")),
    ("person in red shirt", ("red shirt", None)),
    ("the person wearing a blue jacket in front of me", ("blue jacket", "in front")),
    ("the person behind you", (None, "behind me")),
    ("person on the left", (None, "on my left")),
    ("Person in a green hat.", ("green hat", None)),
])
def test_parse_person_description(phrase, expected):
    assert people.parse_person_description(phrase) == expected


# ---------------------------------------------------------------- wait_for_raised_hand

def test_wait_for_raised_hand_one_person(monkeypatch, clk):
    a_up, b = _p(0, kps=_kps(raised_right=True)), _p(30, kps=_kps())
    _snap_seq(monkeypatch, [[a_up, b]] * 5, clk)
    got = people.wait_for_raised_hand(None, [FL], clock=clk, sleep=clk.sleep)
    assert got is not None and got.bearing_deg == 0


def test_wait_for_raised_hand_needs_consecutive(monkeypatch, clk):
    up, down = _p(0, kps=_kps(raised_right=True)), _p(0, kps=_kps())
    _snap_seq(monkeypatch, [[up], [up], [down], [up], [up]] + [[down]] * 30, clk)
    assert people.wait_for_raised_hand(None, [FL], clock=clk, sleep=clk.sleep) is None


def test_wait_for_raised_hand_tracks_small_motion(monkeypatch, clk):
    frames = [[_p(b, kps=_kps(raised_right=True))] for b in (0, 4, 8)]
    _snap_seq(monkeypatch, frames, clk)
    got = people.wait_for_raised_hand(None, [FL], clock=clk, sleep=clk.sleep)
    assert got is not None and got.bearing_deg == 8


def test_wait_for_raised_hand_same_person_two_cameras(monkeypatch, clk):
    a = _p(0, src=FL, conf=0.9, kps=_kps(raised_right=True))
    b = _p(2, src=FR, conf=0.7, kps=_kps(raised_right=True))
    _snap_seq(monkeypatch, [[a, b]] * 5, clk)
    got = people.wait_for_raised_hand(None, [FL, FR], clock=clk, sleep=clk.sleep)
    assert got is not None and got.source == FL


def test_wait_for_raised_hand_two_people_none(monkeypatch, clk):
    a, b = _p(0, kps=_kps(raised_right=True)), _p(40, kps=_kps(raised_left=True))
    _snap_seq(monkeypatch, [[a, b]] * 5, clk)
    assert people.wait_for_raised_hand(None, [FL], clock=clk, sleep=clk.sleep) is None


def test_wait_for_raised_hand_times_out(monkeypatch, clk):
    calls = _snap_seq(monkeypatch, [[_p(0, kps=_kps())]] * 1000, clk)
    assert people.wait_for_raised_hand(None, [FL], timeout_s=5.0, clock=clk, sleep=clk.sleep) is None
    assert clk.t <= 5.0 + 0.2 + 1e-9
    assert all(c["model"] == "yolo26l-pose" for c in calls)


def test_wait_for_raised_hand_accept_filter(monkeypatch, clk):
    outside = _p(90, kps=_kps(raised_right=True))
    _snap_seq(monkeypatch, [[outside]] * 40, clk)
    got = people.wait_for_raised_hand(None, [FL], clock=clk, sleep=clk.sleep,
                                      accept=lambda d: abs(d.bearing_deg) <= 30)
    assert got is None


def test_wait_for_raised_hand_survives_snapshot_error(monkeypatch, clk):
    def boom(*a, **k):
        raise RuntimeError("camera")
    monkeypatch.setattr(people, "snapshot", boom)
    assert people.wait_for_raised_hand(None, [FL], clock=clk, sleep=clk.sleep) is None


# ---------------------------------------------------------------- wearing

def _hand_frame():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    return CameraFrame(people.HAND_SOURCE, img, 480, 640, 0.0, 1, 1, 0, 0,
                       np.eye(4), None, b"", 0.0)


@pytest.fixture
def gripper(monkeypatch):
    state = {"dets": [_p(0, src=people.HAND_SOURCE, box=(300, 200, 340, 300))],
             "answer": "Yes.", "prompts": [], "crops": []}
    monkeypatch.setattr(people, "capture", lambda robot, sources, with_depth=False:
                        {people.HAND_SOURCE: _hand_frame()})
    monkeypatch.setattr(people, "detect_frames", lambda frames, **k: list(state["dets"]))

    def fake_encode(img):
        state["crops"].append(img.shape)
        return b"jpg"
    monkeypatch.setattr(people, "encode_jpeg", fake_encode)

    def fake_vlm(prompt, images, **k):
        state["prompts"].append(prompt)
        return state["answer"]
    monkeypatch.setattr(people, "ask_vlm", fake_vlm)
    return state


@pytest.mark.parametrize("answer,expected", [
    ("Yes.", True), ("**Yes**, they are.", True), ('"no"', False), ("No, a blue one.", False),
    ("I see a person.", None), (None, None), ("Nope", None),
])
def test_wearing_parses_yes_no(gripper, clk, answer, expected):
    gripper["answer"] = answer
    gaze = Spy(True)
    assert people.wearing(None, _p(20), "red shirt", gaze, sleep=clk.sleep) is expected
    assert gaze.calls == [20]
    assert clk.t == pytest.approx(2.5)
    assert gripper["prompts"] == ["Answer yes or no: is this person wearing red shirt?"]


def test_wearing_gaze_refused(gripper, clk):
    assert people.wearing(None, _p(20), "red shirt", Spy(False), sleep=clk.sleep) is None
    assert gripper["prompts"] == [] and clk.t == 0


def test_wearing_no_person_in_gripper(gripper, clk):
    gripper["dets"] = []
    assert people.wearing(None, _p(20), "red shirt", Spy(True), sleep=clk.sleep) is None
    assert gripper["prompts"] == []


def test_wearing_crops_person_nearest_centre(gripper, clk):
    gripper["dets"] = [_p(0, src=people.HAND_SOURCE, box=(0, 0, 50, 60)),
                       _p(0, src=people.HAND_SOURCE, box=(300, 200, 340, 300))]
    people.wearing(None, _p(20), "red shirt", Spy(True), sleep=clk.sleep)
    assert gripper["crops"] == [(100, 40, 3)]


# ---------------------------------------------------------------- select_person: me

def test_select_me_single_candidate_no_question(monkeypatch, clk):
    _bearing(monkeypatch, 10.0)
    _snap_seq(monkeypatch, [[_p(5), _p(120)]])
    speak = Spy(None)
    target, msg = people.select_person(None, "me", speak, Spy(True), clock=clk, sleep=clk.sleep)
    assert target.bearing_deg == 5 and msg is None
    assert speak.calls == []


def test_select_me_same_person_two_cameras_is_one(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(2, src=FL, conf=0.9), _p(4, src=FR, conf=0.7)]])
    speak = Spy(None)
    target, msg = people.select_person(None, "me", speak, Spy(True), clock=clk, sleep=clk.sleep)
    assert target is not None and msg is None and speak.calls == []


def test_select_me_two_candidates_asks_hand(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    a = _p(-15, src=FL, kps=_kps())
    b = _p(15, src=FR, kps=_kps(raised_right=True))
    calls = _snap_seq(monkeypatch, [[a, _p(15, src=FR)]] + [[a, b]] * 5, clk)
    speak = Spy(None)
    target, msg = people.select_person(None, "me", speak, Spy(True), clock=clk, sleep=clk.sleep)
    assert speak.calls == ["Can you raise your hand?"]
    assert target.bearing_deg == 15 and msg is None
    assert set(calls[1]["sources"]) == {FL, FR}


def test_select_me_ambiguous_ends_with_clothing_prompt(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(-15, src=FL), _p(15, src=FR, dist=4.0)]]
              + [[_p(-15, kps=_kps())]] * 100, clk)
    speak = Spy(None)
    target, msg = people.select_person(None, "me", speak, Spy(True), clock=clk, sleep=clk.sleep)
    assert target is None
    assert msg == "I can see a few people. Tell me what you're wearing and I'll come to you."
    assert speak.calls == ["Can you raise your hand?"]


def test_select_me_raised_hand_outside_window_ignored(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(-15), _p(15, dist=4.0)]]
              + [[_p(90, kps=_kps(raised_right=True))]] * 100, clk)
    target, msg = people.select_person(None, "me", Spy(None), Spy(True), clock=clk, sleep=clk.sleep)
    assert target is None and msg.startswith("I can see a few people")


def test_select_me_nobody(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[]])
    assert people.select_person(None, "me", Spy(None), Spy(True), clock=clk, sleep=clk.sleep) == \
        (None, "I can't see you. Could you step in front of me?")


def test_select_me_people_only_outside_window(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(90), _p(-120)]])
    assert people.select_person(None, "me", Spy(None), Spy(True), clock=clk, sleep=clk.sleep) == \
        (None, "I can't see you. Could you step in front of me?")


def test_select_me_window_wraps(monkeypatch, clk):
    _bearing(monkeypatch, 175.0)
    _snap_seq(monkeypatch, [[_p(-170, src="back_fisheye_image")]])
    target, msg = people.select_person(None, "me", Spy(None), Spy(True), clock=clk, sleep=clk.sleep)
    assert target.bearing_deg == -170


def test_select_me_no_speaker_bearing_uses_everyone(monkeypatch, clk):
    _bearing(monkeypatch, None)
    _snap_seq(monkeypatch, [[_p(90)]])
    target, msg = people.select_person(None, "me", Spy(None), Spy(True), clock=clk, sleep=clk.sleep)
    assert target.bearing_deg == 90 and msg is None


# ---------------------------------------------------------------- select_person: clothing

def _wearing_by_bearing(monkeypatch, answers):
    """answers: bearing -> True/False/None; records the bearings asked, calls gaze like the real one."""
    asked = []

    def fake(robot, person, clothing, gaze, sleep=None, abort=None):
        if not gaze(person.bearing_deg):
            return None
        asked.append(person.bearing_deg)
        return answers.get(person.bearing_deg)
    monkeypatch.setattr(people, "wearing", fake)
    return asked


def test_select_clothing_max_three_gazes(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(b) for b in (50, -40, 30, -20, 10)]])
    asked = _wearing_by_bearing(monkeypatch, {})
    gaze = Spy(True)
    target, msg = people.select_person(None, "person in red shirt", Spy(None), gaze,
                                       clock=clk, sleep=clk.sleep)
    assert gaze.calls == [10, -20, 30]  # nearest the speaker bearing first, at most three
    assert (target, msg) == (None, "I'm not sure who you mean.")
    assert asked == [10, -20, 30]


def test_select_clothing_exactly_one_yes(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(-30), _p(10), _p(40)]])
    _wearing_by_bearing(monkeypatch, {10: False, -30: True, 40: None})
    target, msg = people.select_person(None, "person in red shirt", Spy(None), Spy(True),
                                       clock=clk, sleep=clk.sleep)
    assert target.bearing_deg == -30 and msg is None


def test_select_clothing_two_yes_not_sure(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(-30), _p(10)]])
    _wearing_by_bearing(monkeypatch, {10: True, -30: True})
    assert people.select_person(None, "person in red shirt", Spy(None), Spy(True),
                                clock=clk, sleep=clk.sleep) == (None, "I'm not sure who you mean.")


def test_select_clothing_skips_beyond_gaze_clamp(monkeypatch, clk):
    _bearing(monkeypatch, 90.0)
    _snap_seq(monkeypatch, [[_p(95), _p(55)]])
    gaze = Spy(True)
    _wearing_by_bearing(monkeypatch, {55: True})
    target, msg = people.select_person(None, "person in red shirt", Spy(None), gaze,
                                       clock=clk, sleep=clk.sleep)
    assert gaze.calls == [55] and target.bearing_deg == 55


def test_select_clothing_with_side_filters_sector(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(5), _p(-50), _p(50)]])
    gaze = Spy(True)
    _wearing_by_bearing(monkeypatch, {50: True})
    target, msg = people.select_person(None, "person in red shirt on my left", Spy(None), gaze,
                                       clock=clk, sleep=clk.sleep)
    assert gaze.calls == [50] and target.bearing_deg == 50


def test_select_clothing_arm_disabled(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    a = _p(-15, src=FL, kps=_kps())
    b = _p(15, src=FR, kps=_kps(raised_left=True))
    _snap_seq(monkeypatch, [[a, _p(15, src=FR)]] + [[a, b]] * 5, clk)
    _wearing_by_bearing(monkeypatch, {})
    speak = Spy(None)
    target, msg = people.select_person(None, "person in red shirt", speak, Spy(False),
                                       clock=clk, sleep=clk.sleep)
    assert speak.calls == ["Can you raise your hand?"]
    assert target.bearing_deg == 15 and msg is None


def test_select_clothing_arm_disabled_nobody_raises(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(-15), _p(15)]] + [[]] * 100, clk)
    _wearing_by_bearing(monkeypatch, {})
    assert people.select_person(None, "person in red shirt", Spy(None), Spy(False),
                                clock=clk, sleep=clk.sleep) == (None, "I'm not sure who you mean.")


def test_select_clothing_nobody_visible(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[]])
    speak, gaze = Spy(None), Spy(False)
    assert people.select_person(None, "person in red shirt", speak, gaze,
                                clock=clk, sleep=clk.sleep) == \
        (None, "I can't see anyone who matches.")
    assert speak.calls == [] and gaze.calls == []


def test_select_clothing_duplicate_person_gazed_once(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(2, src=FL, conf=0.9), _p(4, src=FR, conf=0.7)]])
    gaze = Spy(True)
    _wearing_by_bearing(monkeypatch, {2: True})
    target, msg = people.select_person(None, "person in red shirt", Spy(None), gaze,
                                       clock=clk, sleep=clk.sleep)
    assert gaze.calls == [2] and target.bearing_deg == 2


# ---------------------------------------------------------------- select_person: side

def test_select_side_duplicate_person_no_question(monkeypatch, clk):
    _bearing(monkeypatch, None)
    _snap_seq(monkeypatch, [[_p(66, src=FR, conf=0.9), _p(68, src="left_fisheye_image", conf=0.6)]])
    target, msg = people.select_person(None, "person on my left", Spy(None), Spy(True),
                                       clock=clk, sleep=clk.sleep)
    assert target.bearing_deg == 66 and msg is None


def test_select_side_plain_left_right_wording():
    assert people.MSG_WHICH_SIDE.format(a=people._where("on my left"), b=people._where("on my right")) \
        == "Do you mean the person on my left or on my right?"

def test_select_side_single(monkeypatch, clk):
    _bearing(monkeypatch, None)
    _snap_seq(monkeypatch, [[_p(0), _p(85)]])
    speak, gaze = Spy(None), Spy(True)
    target, msg = people.select_person(None, "the one on the left", speak, gaze,
                                       clock=clk, sleep=clk.sleep)
    assert target.bearing_deg == 85 and msg is None
    assert speak.calls == [] and gaze.calls == []


def test_select_side_same_sector_takes_nearest(monkeypatch, clk):
    _bearing(monkeypatch, None)
    _snap_seq(monkeypatch, [[_p(80, dist=4.0), _p(100, dist=1.5)]])
    target, msg = people.select_person(None, "person on my left", Spy(None), Spy(True),
                                       clock=clk, sleep=clk.sleep)
    assert target.bearing_deg == 100 and msg is None


def test_select_side_spans_sectors_asks(monkeypatch, clk):
    _bearing(monkeypatch, None)
    _snap_seq(monkeypatch, [[_p(50, dist=2.0), _p(95, dist=3.0)]])
    assert people.select_person(None, "person on my left", Spy(None), Spy(True),
                                clock=clk, sleep=clk.sleep) == \
        (None, "Do you mean the person to my front-left or on my left?")


def test_select_side_nobody_there(monkeypatch, clk):
    _bearing(monkeypatch, None)
    _snap_seq(monkeypatch, [[_p(-90)]])
    assert people.select_person(None, "person on my left", Spy(None), Spy(True),
                                clock=clk, sleep=clk.sleep) == (None, "I'm not sure who you mean.")


# ---------------------------------------------------------------- templates

def test_spoken_templates_have_no_stop_words():
    lines = [t.format(a="front-left", b="on my left") for t in people.TEMPLATES]
    assert len(lines) >= 5
    for line in lines:
        assert not BAD.search(line), line


# ---------------------------------------------------------------- final fix: I4(a), M5

def test_wearing_abort_after_settle_skips_crop_and_vlm(gripper, clk, monkeypatch):
    monkeypatch.setattr(people, "capture", lambda *a, **k: pytest.fail("no crop after a stop"))
    stopped = {"v": False}

    def sleep(s):
        clk.sleep(s)
        stopped["v"] = True  # the stop lands during the gaze settle
    assert people.wearing(None, _p(20), "red shirt", Spy(True), sleep=sleep,
                          abort=lambda: stopped["v"]) is None
    assert gripper["prompts"] == []


def test_wearing_vlm_timeout_is_short(gripper, clk, monkeypatch):
    seen = {}

    def fake_vlm(prompt, images, **k):
        seen.update(k)
        return "Yes."
    monkeypatch.setattr(people, "ask_vlm", fake_vlm)
    assert people.wearing(None, _p(20), "red shirt", Spy(True), sleep=clk.sleep) is True
    assert seen["timeout"] == pytest.approx(people.YES_NO_TIMEOUT_S) and seen["timeout"] <= 5.0


def test_select_clothing_abort_stops_asking(monkeypatch, clk):
    _bearing(monkeypatch, 0.0)
    _snap_seq(monkeypatch, [[_p(b) for b in (10, -20, 30)]])
    asked = _wearing_by_bearing(monkeypatch, {10: False, -20: True, 30: True})
    def abort():
        return len(asked) >= 1  # a stop arrives after the first candidate
    speak = Spy(None)
    assert people.select_person(None, "person in red shirt", speak, Spy(True), clock=clk,
                                sleep=clk.sleep, abort=abort) == (None, None)
    assert asked == [10] and speak.calls == []


def test_wait_for_raised_hand_pauses_between_snapshots(monkeypatch, clk):
    calls = _snap_seq(monkeypatch, [[_p(0)]] * 100, clk=clk, step=0.0)
    slept = []

    def sleep(s):
        slept.append(s)
        clk.sleep(s)
    assert people.wait_for_raised_hand(None, [FL], timeout_s=1.0, clock=clk, sleep=sleep) is None
    assert slept and all(0 < s <= people.HAND_POLL_S + 1e-9 for s in slept)
    assert slept[0] == pytest.approx(people.HAND_POLL_S)
    assert len(calls) <= int(1.0 / people.HAND_POLL_S) + 1
