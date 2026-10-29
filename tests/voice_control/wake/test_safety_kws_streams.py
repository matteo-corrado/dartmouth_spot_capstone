"""Staggered-stream schedule of the always-on safety KWS (kwsfix 2026-10-29).

A long-lived sherpa stream loses quiet/far stop/freeze; the detector keeps
NUM_STREAMS streams reborn on a stagger so one is always young."""
import json
import pathlib
import wave

import numpy as np
import pytest

import src.voice_control.wake.safety_kws as sk

FRAME = b"\x00\x00" * 480


class FakeStream:
    def __init__(self, sid):
        self.sid, self.fed = sid, 0

    def accept_waveform(self, sr, samples):
        self.fed += len(samples)


class FakeSpotter:
    """Records births; reports `hit_on` from a stream once it has been fed
    `hit_after` frames' worth of audio."""

    def __init__(self, hit_after=None):
        self.births, self.n, self.hit_after = [], 0, hit_after
        self.frame = 0

    def create_stream(self):
        self.n += 1
        self.births.append(self.frame)
        return FakeStream(self.n)

    def is_ready(self, st):
        return False

    def decode_stream(self, st):
        pass

    def get_result(self, st):
        if self.hit_after is not None and st.fed >= self.hit_after * 480:
            return "stop"
        return ""


def _det(spotter):
    det = sk.SafetyKeywordDetector(model_dir="/nonexistent")
    det._available, det._spotter = True, spotter
    return det


def _feed(det, sp, n):
    out = []
    for _ in range(n):
        out.append(det.process_frame(FRAME))
        sp.frame += 1
    return out


def test_births_follow_stagger_schedule():
    sp = FakeSpotter()
    det = _det(sp)
    n = 3 * sk.NUM_STREAMS * sk.STAGGER_FRAMES
    _feed(det, sp, n)
    want = sorted(f for i in range(sk.NUM_STREAMS)
                  for f in range(i * sk.STAGGER_FRAMES, n, sk.NUM_STREAMS * sk.STAGGER_FRAMES))
    assert sp.births == want
    assert len(det._streams) == sk.NUM_STREAMS
    # Every frame after the warm-up, all streams are live and their ages are staggered.
    ages = sorted(s.fed // 480 for s in det._streams)
    assert len(set(ages)) == sk.NUM_STREAMS


def test_youngest_stream_age_bounded():
    # After warm-up there is always a stream no older than one stagger.
    sp = FakeSpotter()
    det = _det(sp)
    for _ in range(10 * sk.NUM_STREAMS * sk.STAGGER_FRAMES):
        det.process_frame(FRAME)
        sp.frame += 1
        if sp.frame >= sk.NUM_STREAMS * sk.STAGGER_FRAMES:
            assert min(s.fed for s in det._streams) <= sk.STAGGER_FRAMES * 480


def test_reset_restarts_schedule():
    sp = FakeSpotter()
    det = _det(sp)
    _feed(det, sp, 50)
    det.reset()
    assert det._streams == [None] * sk.NUM_STREAMS
    sp.births.clear()
    start = sp.frame
    _feed(det, sp, sk.STAGGER_FRAMES + 1)
    assert sp.births == [start, start + sk.STAGGER_FRAMES]


def test_hit_returns_label_and_resets():
    sp = FakeSpotter(hit_after=5)
    det = _det(sp)
    out = _feed(det, sp, 5)
    assert out == ["", "", "", "", "stop"]
    assert det._streams == [None] * sk.NUM_STREAMS and det._frame == 0


def test_stays_healthy_across_many_resets_and_hits():
    sp = FakeSpotter(hit_after=40)  # hits often: streams reborn constantly
    det = _det(sp)
    hits = 0
    for k in range(5000):
        if k % 97 == 0:
            det.reset()
        hits += det.process_frame(FRAME) == "stop"
        sp.frame += 1
        assert len(det._streams) == sk.NUM_STREAMS
        assert sum(s is not None for s in det._streams) <= sk.NUM_STREAMS
    assert hits > 50
    assert det.process_frame(FRAME) in ("", "stop")


def test_stream_error_propagates_and_reset_recovers():
    class Boom(FakeSpotter):
        bad = True

        def create_stream(self):
            st = super().create_stream()
            if self.bad:
                def bad(sr, x):
                    raise RuntimeError("onnx")
                st.accept_waveform = bad
            return st

    sp = Boom()
    det = _det(sp)
    with pytest.raises(RuntimeError):
        det.process_frame(FRAME)
    det.reset()
    sp.bad = False
    assert det.process_frame(FRAME) == ""


def test_hit_returned_before_a_later_stream_raises():
    sp = FakeSpotter()
    det = _det(sp)
    _feed(det, sp, sk.STAGGER_FRAMES + 1)  # streams 0 and 1 alive
    s0 = det._streams[0]
    sp.get_result = lambda st: "freeze" if st is s0 else ""

    def boom(sr, x):
        raise RuntimeError("onnx")
    det._streams[1].accept_waveform = boom
    assert det.process_frame(FRAME) == "freeze"   # hit kept, stream 1 never touched
    assert det._streams == [None] * sk.NUM_STREAMS


def test_create_stream_failure_in_rebirth_propagates():
    class NoCreate(FakeSpotter):
        def create_stream(self):
            raise RuntimeError("alloc")

    det = _det(NoCreate())
    with pytest.raises(RuntimeError):
        det.process_frame(FRAME)  # safety_kws_loop counts it toward health


def test_threshold_read_from_tuning(monkeypatch):
    from src.voice_control import tuning
    assert tuning.Tuning().safety_kws_threshold == sk.KEYWORDS_THRESHOLD == 0.25
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(safety_kws_threshold=0.2))
    assert sk._threshold() == 0.2

    def broken():
        raise OSError("bad yaml")
    monkeypatch.setattr(tuning, "get_tuning", broken)
    assert sk._threshold() == 0.25


@pytest.mark.parametrize("val,want,warn", [(0.25, 0.25, False), (0.05, 0.05, False), (0.40, 0.40, False),
                                           (0.01, 0.05, True), (0.9, 0.40, True), (-1.0, 0.05, True)])
def test_threshold_clamped(monkeypatch, capsys, val, want, warn):
    from src.voice_control import tuning
    monkeypatch.setattr(tuning, "_cached", tuning.Tuning(safety_kws_threshold=val))
    assert sk._threshold() == want
    assert ("out of range" in capsys.readouterr().out) is warn


# --- real model + operator recordings (skipped when absent) -----------------

_DEMO = pathlib.Path("/home/spotdog/spot/dartmouth_spot_capstone/tests/audio/demo")
_MODEL = pathlib.Path(sk._MODEL_DIR) / sk._SAFETY_KEYWORDS
_real = pytest.mark.skipif(not (_MODEL.exists() and (_DEMO / "safety/manifest.json").exists()
                                and (_DEMO / "fan/background.wav").exists()),
                           reason="safety KWS model or demo recordings absent")


def _load_ch1(p):
    with wave.open(str(p), "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, w.getnchannels())
    return x[:, 1 if x.shape[1] > 1 else 0].astype(np.float32) / 32768.0


def _run(det, x, refractory_s=1.5):
    """safety_kws_loop semantics: frames during the refractory are dropped."""
    hits, last = [], None
    for i in range(0, len(x) - 480 + 1, 480):
        t = i / 16000
        if last is not None and t - last < refractory_s:
            continue
        kw = det.process_frame((x[i:i + 480] * 32767).astype(np.int16).tobytes())
        if kw:
            hits.append((t, kw)); last = t; det.reset()
    return hits


@_real
def test_real_continuous_stream_recall_and_fires():
    det = sk.SafetyKeywordDetector()
    assert det.is_available()
    man = json.load(open(_DEMO / "safety/manifest.json"))
    fan = _load_ch1(_DEMO / "fan/background.wav")
    rng = np.random.default_rng(0)
    parts, spans, off = [], [], 0
    for m in man:
        pl = int(rng.uniform(1.0, 6.0) * 16000); s0 = int(rng.uniform(0, len(fan) - pl))
        parts.append(fan[s0:s0 + pl]); off += pl
        x = _load_ch1(_DEMO / "safety" / (m["clip"] + ".wav"))
        spans.append((m["expected_text"], off / 16000, (off + len(x)) / 16000))
        parts.append(x); off += len(x)
    parts.append(fan[:32000])
    hits = _run(det, np.concatenate(parts))
    ok = sum(any(kw == want and a <= t <= b + 0.8 for t, kw in hits) for want, a, b in spans)
    # Single long-lived stream (pre-fix) measured 7/14 on this seed; staggered 9/14.
    assert ok >= 8, (ok, hits)
    det.reset()
    fires = _run(det, fan)  # 2 min of fan; baseline measured 1-2 here
    assert len(fires) <= 2, fires
