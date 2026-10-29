"""Pure parts of the demo replay harness (tests/audio/replay_demo.py)."""
import importlib.util
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parents[1] / "audio" / "replay_demo.py"
_spec = importlib.util.spec_from_file_location("replay_demo", _P)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def test_chime_mask_masks_chime_plus_tail_and_keeps_talker_vad():
    m = R.ChimeMask(0.15)
    assert m.update(False, 0.0) == (False, False)
    assert m.update(True, 0.03) == (True, False)
    assert m.update(False, 0.12)[0] is True           # inside tail
    masked, ended = m.update(False, 0.21)
    assert (masked, ended) == (False, True)
    assert m.should_reset_vad(ended, in_utt=False)     # chime with no talker -> reset VAD
    m2 = R.ChimeMask(0.15)
    m2.update(False, 0.0, speech=True)
    m2.update(True, 0.03, speech=True)
    _, ended = m2.update(False, 0.3)
    assert ended and not m2.should_reset_vad(ended, in_utt=False)


def test_coverage_and_overlap():
    ref = [(1.0, 2.0), (3.0, 4.0)]
    assert R.overlap_s(ref, 1.5, 3.5) == 1.0
    assert R.coverage(ref, 0.0, 0.5, 4.5) == 1.0
    assert R.coverage(ref, 0.0, 0.5, 2.5) == 0.5      # ended in the pause -> cut
    assert R.coverage(ref, 3.5, 0.0, 9.0) == 1.0      # only speech after the wake counts
    assert R.coverage([], 0.0, 0.0, 1.0) is None


def test_lock_result_uses_calibration():
    # mic 179 deg is body front with offset 179, sign -1
    mic, body, ok = R.lock_result([178.0, 179.0, 180.0], 0, 179.0, -1, 35.0)
    assert mic == 179.0 and body == 0.0 and ok
    _, body, ok = R.lock_result([89.0, 90.0], 0, 179.0, -1, 35.0)
    assert body == 90.0 and not ok          # mic 89 = body LEFT
    assert R.lock_result([89.0, 90.0], 90, 179.0, -1, 35.0)[2]
    assert R.lock_result([], 0, 179.0, -1, 35.0) == (None, None, None)


def test_speech_degs_between_filters_flag_and_window():
    doa = [(0.1, 10.0, True), (0.5, 20.0, False), (1.0, 30.0, True), (2.0, 40.0, True)]
    assert R.speech_degs_between(doa, 0.0, 1.0) == [10.0, 30.0]


def test_norm_kw_and_rates():
    assert R.norm_kw("HEY_SPOT") == "HEY SPOT"
    assert R.norm_kw(" listen up spot ") == "LISTEN UP SPOT"
    assert R.per_min(3, 120.0) == 1.5
    assert R.last_word_heard("hey spot what is this poster about?", "What is this poster about")
    assert not R.last_word_heard("hey spot what is this poster about?", "what is this")


def test_split_frames_and_pad():
    pcm = np.arange(480 * 3 + 7, dtype=np.int16)
    fr = R.split_frames(pcm)
    assert len(fr) == 3 and len(fr[0]) == 960
    pad = R.pad_from_lead(np.ones(16000, dtype=np.int16), lead_s=0.1, pad_s=0.5)
    assert len(pad) == 8000 and pad.dtype == np.int16
    assert R.frames_of(1.2) == 40
