"""Per-camera IoU + constant-velocity tracker and cross-camera hand-off."""
import math

import pytest

from src.voice_control.vision.tracker import CameraTracker, handoff
from src.voice_control.vision.types import Detection


def _det(box, source="frontleft_fisheye_image", bearing=0.0, dist=2.0, label="person"):
    return Detection(label=label, conf=0.9, source=source, box=box, mask=None,
                     keypoints=None, bearing_deg=bearing, distance_m=dist,
                     distance_method="depth" if dist is not None else None)


def test_tracker_keeps_id_across_five_frames_of_moving_box():
    tr = CameraTracker()
    ids = []
    for i in range(5):
        x = 100 + 15 * i
        out = tr.update([_det((x, 50, x + 60, 200))], t=0.1 * i)
        assert len(out) == 1
        ids.append(out[0].track_id)
    assert ids[0] is not None and len(set(ids)) == 1


def test_tracker_two_people_keep_separate_ids():
    tr = CameraTracker()
    a = tr.update([_det((0, 0, 50, 100)), _det((300, 0, 350, 100))], 0.0)
    b = tr.update([_det((305, 0, 355, 100)), _det((5, 0, 55, 100))], 0.1)
    by_x = {round(d.box[0]): d.track_id for d in b}
    assert by_x[5] == a[0].track_id and by_x[305] == a[1].track_id
    assert a[0].track_id != a[1].track_id


def test_tracker_id_dies_after_one_second_unseen():
    tr = CameraTracker()
    first = tr.update([_det((100, 50, 160, 200))], 0.0)[0].track_id
    assert tr.update([], 0.5) == []
    assert tr.update([], 1.2) == []          # aged out here (1.2 s unseen)
    again = tr.update([_det((100, 50, 160, 200))], 1.3)[0].track_id
    assert again != first


def test_tracker_id_survives_short_gap():
    tr = CameraTracker()
    first = tr.update([_det((100, 50, 160, 200))], 0.0)[0].track_id
    tr.update([], 0.4)
    assert tr.update([_det((104, 50, 164, 200))], 0.8)[0].track_id == first


def test_tracker_low_iou_gets_new_id():
    tr = CameraTracker()
    first = tr.update([_det((0, 0, 50, 100))], 0.0)[0].track_id
    assert tr.update([_det((400, 0, 450, 100))], 0.1)[0].track_id != first


def test_tracker_does_not_mutate_input():
    d = _det((0, 0, 50, 100))
    out = CameraTracker().update([d], 0.0)
    assert d.track_id is None and out[0].track_id is not None


def test_handoff_picks_nearest_within_limits():
    last = _det((0, 0, 1, 1), bearing=40.0, dist=3.0)
    near = _det((0, 0, 1, 1), source="left_fisheye_image", bearing=50.0, dist=3.1)
    nearer = _det((0, 0, 1, 1), source="left_fisheye_image", bearing=44.0, dist=3.0)
    far = _det((0, 0, 1, 1), source="left_fisheye_image", bearing=42.0, dist=4.5)   # > 1 m
    wide = _det((0, 0, 1, 1), source="left_fisheye_image", bearing=75.0, dist=3.0)   # > 30 deg
    assert handoff(last, [near, far, wide, nearer]) is nearer
    assert handoff(last, [far, wide]) is None
    assert handoff(last, []) is None


def test_handoff_bearing_limit_wraps_around_180():
    last = _det((0, 0, 1, 1), bearing=175.0, dist=1.0)
    other = _det((0, 0, 1, 1), bearing=-178.0, dist=1.0)
    assert handoff(last, [other]) is other


def test_handoff_candidate_without_distance_skipped():
    last = _det((0, 0, 1, 1), bearing=0.0, dist=2.0)
    assert handoff(last, [_det((0, 0, 1, 1), bearing=1.0, dist=None)]) is None


def test_handoff_last_without_distance_matches_on_bearing_only():
    last = _det((0, 0, 1, 1), bearing=10.0, dist=None)
    a = _det((0, 0, 1, 1), bearing=25.0, dist=5.0)
    b = _det((0, 0, 1, 1), bearing=14.0, dist=None)
    c = _det((0, 0, 1, 1), bearing=60.0, dist=2.0)
    assert handoff(last, [a, b, c]) is b
    assert handoff(last, [c]) is None
