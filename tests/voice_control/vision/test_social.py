"""Friendly Spot social mode: comfort port, COCO pose rules, loop safety (no robot, no GPU)."""
import importlib
import math
import threading

import numpy as np
import pytest

S = importlib.import_module("src.voice_control.vision.social")
from src.voice_control.vision.comfort import BehaviorLabel, ComfortModel, PerceptionInput
from src.voice_control.vision.social_pose import (
    LEFT_SHOULDER, LEFT_WRIST, NOSE, RIGHT_SHOULDER, RIGHT_WRIST, PoseClassifier, normalise)
from src.voice_control.vision.types import Detection


# -- comfort model (original values) -------------------------------------------
def test_comfort_original_sample():
    # friendly_spot planner.__main__ sample; worked by hand from the original weights
    p = PerceptionInput(current_action="waiting", distance_m=1.0, face_label="sally",
                        emotion_label="happy", pose_label="waving", gesture_label="thumbs_up")
    comfort, b = ComfortModel().predict_behavior(p)
    # 0.5 +0.03 +0.12 +0 (thumbs_up not a key) +0.4 +0.09 +0.12 -> clamp 1.0
    assert comfort == 1.0 and b is BehaviorLabel.GO_CLOSE


def test_comfort_stranger_neutral_sits():
    p = PerceptionInput(distance_m=1.2)  # idle, unknown face, neutral, standing, no gesture
    comfort, b = ComfortModel().predict_behavior(p)
    assert comfort == pytest.approx(0.5 + 0.05 + 0.1 + 0.09 - 0.18)
    assert b is BehaviorLabel.SIT


def test_comfort_angry_close_backs_away():
    p = PerceptionInput(distance_m=0.3, emotion_label="angry", gesture_label="closed_fist")
    assert ComfortModel().predict_behavior(p)[1] is BehaviorLabel.BACK_AWAY


def test_comfort_bilateral_gesture_doubles():
    m = ComfortModel()
    assert m._score_gesture("Left: Thumb_Up, Right: Thumb_Up", m.hp.gesture_weights) == pytest.approx(0.4)
    assert m._score_gesture("Left: Thumb_Up, Right: Thumb_Down", m.hp.gesture_weights) == pytest.approx(-0.25)


# -- pose rules ------------------------------------------------------------------
def _kps(over=None):
    """Standing person facing the camera in a 0..200 px tall box (x, y px)."""
    base = {0: (50, 10), 5: (65, 40), 6: (35, 40), 7: (68, 70), 8: (32, 70), 9: (68, 100),
            10: (32, 100), 11: (60, 110), 12: (40, 110), 13: (60, 150), 14: (40, 150),
            15: (62, 195), 16: (38, 195), 1: (52, 8), 2: (48, 8), 3: (55, 10), 4: (45, 10)}
    base.update(over or {})
    k = np.zeros((17, 3))
    for i, (x, y) in base.items():
        k[i] = (x, y, 0.9)
    return k


BOX = (0, 0, 100, 200)


def test_pose_standing():
    assert PoseClassifier().classify(normalise(_kps(), BOX)) == "standing"


def test_pose_raised_hand():
    assert PoseClassifier().classify(normalise(_kps({9: (70, 0)}), BOX)) == "raising_hand"


def test_pose_waving_needs_motion_history():
    pc = PoseClassifier()
    labels = [pc.classify(normalise(_kps({9: (70 + (30 if i % 2 else 0), 20)}), BOX))
              for i in range(8)]
    assert labels[-1] == "waving"


def test_pose_low_conf_keypoints_ignored():
    k = _kps({9: (70, 0)})
    k[LEFT_WRIST, 2] = 0.1  # the raised wrist is not trusted
    assert PoseClassifier().classify(normalise(k, BOX)) != "raising_hand"


def test_pose_arms_crossed():
    k = _kps({LEFT_WRIST: (38, 60), RIGHT_WRIST: (62, 62)})
    assert PoseClassifier().classify(normalise(k, BOX)) == "arms_crossed"


def test_pose_motion_compares_previous_frame():
    pc = PoseClassifier()
    for _ in range(3):
        pc.classify(normalise(_kps(), BOX))
    assert pc._speed(normalise(_kps(), BOX)) == 0.0
    moved = _kps()
    moved[:, 0] += 3
    assert pc._speed(normalise(moved, BOX)) > 0.0


# -- loop --------------------------------------------------------------------------
class Clock:
    def __init__(self):
        self.t = 0.0


class FakeIO:
    def __init__(self, clock):
        self.clock = clock
        self.pose = (0.0, 0.0, 0.0)
        self.moves, self.vels, self.postures = [], [], []
        self.halts = 0
        self._aborted = False

    def odom_pose(self):
        return self.pose

    def move_body(self, x, y, yaw):
        self.moves.append((x, y, yaw))
        return not self._aborted

    def velocity(self, vx, vy, vrot):
        self.vels.append((vx, vy, vrot))
        return not self._aborted

    def posture(self, h):
        self.postures.append(h)
        return not self._aborted

    def halt(self):
        self.halts += 1

    def aborted(self):
        return self._aborted


def _person(bearing=0.0, dist=3.0, source="frontleft_fisheye_image", method="depth"):
    return Detection(label="person", conf=0.9, source=source, box=(0, 0, 100, 200), mask=None,
                     keypoints=_kps(), bearing_deg=bearing, distance_m=dist,
                     distance_method=method)


class FixedModel:
    def __init__(self, behavior):
        self.behavior = behavior

    def predict_behavior(self, p):
        return 0.5, self.behavior


from types import SimpleNamespace


def _back(depth_m=3.0):
    return SimpleNamespace(depth=np.full((120, 160), depth_m), t=None)


def _run(behavior, dets, cycles=6, io_hook=None, frames=None, **kw):
    clock = Clock()
    io = FakeIO(clock)
    ev = threading.Event()
    aims = []

    def wait(s):
        clock.t += s
        if io_hook:
            io_hook(io, clock)

    res = S.social(io, lambda: (frames or {}, dets() if callable(dets) else dets), ev,
                   aim=aims.append, model=FixedModel(behavior),
                   now=lambda: clock.t, wait=wait, max_duration_s=cycles / S.RATE_HZ, **kw)
    return res, io, aims


def test_go_close_steps_toward_person_but_not_too_close():
    res, io, aims = _run(BehaviorLabel.GO_CLOSE, [_person(10.0, 3.0)], cycles=4)
    assert io.moves, "should approach"
    x, y, yaw = io.moves[0]
    assert math.hypot(x, y) <= 0.5 + 1e-9 and yaw == pytest.approx(math.radians(10.0))
    assert 10.0 in aims


def test_go_close_never_without_depth_range():
    _, io, _ = _run(BehaviorLabel.GO_CLOSE, [_person(0.0, 3.0, method="ground")])
    assert io.moves == []


def test_go_close_stops_at_min_gap():
    _, io, _ = _run(BehaviorLabel.GO_CLOSE, [_person(0.0, 0.9)])
    assert io.moves == []


def test_back_away_is_slow_and_short():
    _, io, _ = _run(BehaviorLabel.BACK_AWAY, [_person(0.0, 0.5)], cycles=3,
                    frames={"back_fisheye_image": _back()})
    moving = [v for v in io.vels if v != (0.0, 0.0, 0.0)]
    assert moving and all(v == (-0.2, 0.0, 0.0) for v in moving)
    assert io.vels[-1] == (0.0, 0.0, 0.0)
    # commanded travel plus the last command's coast stays within the approved 0.3 m
    assert len(moving) * 0.2 * 0.2 + 0.2 * 0.5 <= 0.3 + 1e-9


def test_back_away_blocked_by_someone_behind():
    behind = _person(180.0, 1.0, source="back_fisheye_image")
    _, io, _ = _run(BehaviorLabel.BACK_AWAY, [_person(0.0, 0.5), behind],
                    frames={"back_fisheye_image": _back()})
    assert io.vels == []


def test_back_away_blocked_without_back_camera_or_by_wall():
    _, io, _ = _run(BehaviorLabel.BACK_AWAY, [_person(0.0, 0.5)])
    assert io.vels == []
    _, io, _ = _run(BehaviorLabel.BACK_AWAY, [_person(0.0, 0.5)],
                    frames={"back_fisheye_image": _back(0.8)})
    assert io.vels == []


def test_step_gap_checks_bystanders():
    bystander = _person(20.0, 1.3)
    assert not S.step_gap_ok([_person(0.0, 3.0), bystander], 0.0, 0.5)
    assert S.step_gap_ok([_person(0.0, 3.0), _person(80.0, 1.0)], 0.0, 0.5)


def test_sit_crouches_and_stands_back_up_at_end():
    res, io, aims = _run(BehaviorLabel.SIT, [_person(0.0, 1.2)])
    assert io.postures[0] == S.CROUCH_M and io.postures[-1] == 0.0
    assert aims[-1] is None and res.ok


def test_debounce_needs_two_cycles():
    seq = iter([BehaviorLabel.GO_CLOSE, BehaviorLabel.STAY, BehaviorLabel.GO_CLOSE, BehaviorLabel.STAY])

    class Alt:
        def predict_behavior(self, p):
            return 0.5, next(seq, BehaviorLabel.STAY)
    clock = Clock()
    io = FakeIO(clock)
    S.social(io, lambda: ({}, [_person(0.0, 3.0)]), threading.Event(), model=Alt(),
             now=lambda: clock.t, wait=lambda s: setattr(clock, "t", clock.t + s),
             max_duration_s=4 / S.RATE_HZ)
    assert io.moves == []


def test_stop_sends_nothing_more():
    def hook(io, clock):
        io._aborted = True
    res, io, aims = _run(BehaviorLabel.SIT, [_person(0.0, 1.2)], io_hook=hook)
    assert io.postures in ([], [S.CROUCH_M])  # nothing after the abort
    assert io.halts == 0 and not res.ok and res.data["outcome"] == "paused"


def test_radius_limit_ends():
    def hook(io, clock):
        io.pose = (5.0, 0.0, 0.0)
    res, io, _ = _run(BehaviorLabel.STAY, [_person()], io_hook=hook)
    assert res.data["outcome"] == "limit"


def test_prefers_gripper_view_of_target():
    fish = _person(5.0, 2.0)
    hand = _person(8.0, 2.1, source="hand_color_image")
    assert S.best_view([fish, hand], fish) is hand
    assert S.best_range([fish, hand], fish) == (2.0, "depth")


def test_tracks_same_person():
    a, b = _person(0.0, 3.0), _person(60.0, 1.0)
    assert S.pick_target([a, b], None) is b          # nearest first
    assert S.pick_target([a, b], 5.0) is a           # then stick with them


# -- wiring --------------------------------------------------------------------------
def test_regex_and_grammar():
    from src.voice_control.intent import parse_intent  # noqa: F401  (import check)
    import pathlib
    g = pathlib.Path("src/voice_control/grammar/spot_action.gbnf").read_text()
    assert '"\\"social_mode\\""' in g
    from src.voice_control import spot_dispatch as sd
    assert "social_mode" in sd.LOCOMOTION_INTENTS


def test_gaze_tracking_only_inside_social_nav(monkeypatch):
    from src.voice_control import spot_dispatch as sd
    monkeypatch.setattr(sd, "is_navigating", lambda: True)
    monkeypatch.setattr(sd, "_nav_mode", "follow")
    assert not sd._gaze_nav_ok(False) and not sd._gaze_nav_ok(True)
    monkeypatch.setattr(sd, "_nav_mode", "social")
    assert sd._gaze_nav_ok(True) and not sd._gaze_nav_ok(False)
    monkeypatch.setattr(sd, "is_navigating", lambda: False)
    assert sd._gaze_nav_ok(False) and not sd._gaze_nav_ok(True)


def test_regex_social_mode():
    from src.voice_control.intent import parse_intent
    for t in ("switch to social mode", "friendly mode please"):
        r = parse_intent(t)
        assert r and r["intent"] == "social_mode", t


def test_keypoint_range_reads_torso_not_background():
    dep = np.full((200, 100), 4.0)       # wall behind
    dep[40:115, 30:70] = 1.2             # torso
    f = SimpleNamespace(depth=dep, raw_rows=200, raw_cols=100, rot_deg=0.0, t=None)
    assert S.keypoint_range(f, _kps()) == pytest.approx(1.2)


# -- face / gesture decode -------------------------------------------------------
def test_gesture_decode_dedupes_and_drops_no_gesture():
    from src.voice_control.vision import social_models as M
    out = np.zeros((1, 300, 6), np.float32)
    out[0, 0] = (10, 10, 50, 50, 0.9, 16)   # like
    out[0, 1] = (11, 11, 50, 50, 0.8, 16)   # duplicate
    out[0, 2] = (100, 10, 140, 50, 0.7, 33)  # no_gesture
    out[0, 3] = (200, 10, 240, 50, 0.6, 14)  # fist
    hands = M.decode_gestures(out)
    assert [h[0] for h in hands] == ["like", "fist"]
    assert M.format_gestures(hands, 120.0) == "Left: thumb_up, Right: closed_fist"
    assert M.format_gestures([], 0) == "none"


def test_gesture_labels_score_in_comfort_model():
    m = ComfortModel()
    assert m._score_gesture("Hand: thumb_up", m.hp.gesture_weights) == pytest.approx(0.2)
    assert m._score_gesture("Left: iloveyou, Right: iloveyou", m.hp.gesture_weights) == pytest.approx(0.8)


def test_emotion_labels_match_comfort_weights():
    from src.voice_control.vision import social_models as M
    assert set(M.TO_DEEPFACE.values()) <= set(ComfortModel().hp.emotion_weights)
