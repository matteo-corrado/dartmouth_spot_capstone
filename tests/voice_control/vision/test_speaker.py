from types import SimpleNamespace

import pytest

from src.voice_control.vision import speaker
from src.voice_control.vision.types import Detection


def _p(bearing, conf=0.8):
    return Detection("person", conf, "frontleft_fisheye_image", (0, 0, 1, 1), None, None,
                     bearing, None, None)


@pytest.fixture
def gpu(monkeypatch):
    monkeypatch.setattr(speaker, "load_model", lambda name: SimpleNamespace(provider="trt"))
    monkeypatch.setattr(speaker, "cameras_covering", lambda b, frames=None: ["frontleft_fisheye_image"])


def _snap(monkeypatch, dets):
    monkeypatch.setattr(speaker, "snapshot", lambda *a, **k: list(dets))


def test_refine_picks_person_nearest_doa(gpu, monkeypatch):
    _snap(monkeypatch, [_p(12), _p(40)])
    assert speaker.refine_speaker_bearing(None, 25.0) == 12  # 13 vs 15 away


def test_refine_tie_prefers_higher_conf(gpu, monkeypatch):
    _snap(monkeypatch, [_p(10, 0.6), _p(40, 0.9)])
    assert speaker.refine_speaker_bearing(None, 25.0) == 40


def test_refine_no_person_returns_doa(gpu, monkeypatch):
    _snap(monkeypatch, [])
    assert speaker.refine_speaker_bearing(None, 25.0) == 25.0


def test_refine_outside_window_returns_doa(gpu, monkeypatch):
    _snap(monkeypatch, [_p(80)])
    assert speaker.refine_speaker_bearing(None, 25.0) == 25.0


def test_refine_wraparound(gpu, monkeypatch):
    _snap(monkeypatch, [_p(-178)])
    assert speaker.refine_speaker_bearing(None, 175.0) == -178


def test_refine_never_raises(gpu, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("x")
    monkeypatch.setattr(speaker, "snapshot", boom)
    assert speaker.refine_speaker_bearing(None, 25.0) == 25.0


def test_refine_none_in_none_out():
    assert speaker.refine_speaker_bearing(None, None) is None


def test_refine_no_gpu_returns_doa(monkeypatch):
    monkeypatch.setattr(speaker, "load_model", lambda name: SimpleNamespace(provider="cpu"))
    monkeypatch.setattr(speaker, "snapshot", lambda *a, **k: pytest.fail("no snapshot"))
    assert speaker.refine_speaker_bearing(None, 25.0) == 25.0
    def boom(name):
        raise RuntimeError("no model")
    monkeypatch.setattr(speaker, "load_model", boom)
    assert speaker.refine_speaker_bearing(None, 25.0) == 25.0


def test_getter(monkeypatch):
    monkeypatch.setattr(speaker, "_getter", None)
    assert speaker.speaker_bearing() is None
    speaker.set_speaker_bearing_getter(lambda: 33.0)
    assert speaker.speaker_bearing() == 33.0
    def boom():
        raise ValueError
    speaker.set_speaker_bearing_getter(boom)
    assert speaker.speaker_bearing() is None
    speaker.set_speaker_bearing_getter(lambda: None)
    assert speaker.speaker_bearing() is None
    monkeypatch.setattr(speaker, "_getter", None)
