"""I2 gaze + C5 remainder: the conversation loop only gazes when the gaze
feature is loaded; the one gaze worker has a stop Event set at shutdown."""
import queue
import threading
import time

import pytest

import src.spot.conversation.loop as _loop
import src.spot.features as _features
from src.spot.core.registry import Registry
from src.spot.features.enabled import ENABLED


@pytest.fixture
def gaze_env(monkeypatch):
    monkeypatch.delenv("SPOT_DRY_RUN", raising=False)
    monkeypatch.setattr(_loop, "_gaze_q", queue.Queue(maxsize=1))
    monkeypatch.setattr(_loop, "_gaze_worker", None)
    monkeypatch.setattr(_loop, "_gaze_stop", threading.Event())
    monkeypatch.setattr(_loop, "_safety_gen_now", lambda: 7)
    calls = []
    import src.spot.features.gaze.handlers as gaze_h
    monkeypatch.setattr(gaze_h, "gaze_at_bearing", lambda b, g: calls.append((b, g)) or True)
    yield calls
    _loop._gaze_stop.set()
    w = _loop._gaze_worker
    if w is not None:
        w.join(timeout=2)


def _wait(pred, t=2.0):
    end = time.monotonic() + t
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return pred()


def test_gaze_runs_when_feature_loaded(gaze_env):
    _loop._gaze_at_speaker(30.0)
    assert _wait(lambda: gaze_env == [(30.0, 7)])


def test_no_gaze_when_feature_not_in_enabled(gaze_env, monkeypatch):
    reg = Registry()
    reg.load([n for n in ENABLED if n != "gaze"])
    assert not reg.has("gaze")
    monkeypatch.setattr(_features, "REGISTRY", reg)
    _loop._gaze_at_speaker(30.0)
    assert _loop._gaze_worker is None and _loop._gaze_q.empty() and gaze_env == []


def test_stop_event_ends_worker_and_blocks_respawn(gaze_env):
    _loop._gaze_at_speaker(10.0)
    assert _wait(lambda: gaze_env == [(10.0, 7)])
    w = _loop._gaze_worker
    _loop.stop_gaze_worker()
    w.join(timeout=2)
    assert not w.is_alive()
    _loop._gaze_at_speaker(20.0)  # after shutdown: no new worker, no request
    assert _loop._gaze_worker is w and _loop._gaze_q.empty()
    assert gaze_env == [(10.0, 7)]


def test_only_one_worker_under_concurrent_requests(gaze_env):
    started = []
    real = threading.Thread

    class CountingThread(real):
        def start(self):
            if self.name == "gaze":
                started.append(self)
            super().start()
    import unittest.mock as um
    with um.patch.object(_loop.threading, "Thread", CountingThread):
        ts = [real(target=_loop._gaze_at_speaker, args=(float(i),)) for i in range(16)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    assert len(started) == 1
    assert sum(1 for t in threading.enumerate() if t.name == "gaze" and t.is_alive()) == 1


def test_missing_gaze_module_does_not_kill_worker(gaze_env, monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "src.spot.features.gaze.handlers", None)  # import fails
    _loop._gaze_at_speaker(5.0)
    assert _wait(lambda: _loop._gaze_worker is not None)
    time.sleep(0.1)
    assert _loop._gaze_worker.is_alive()


def test_none_gen_still_skipped(gaze_env, monkeypatch):
    monkeypatch.setattr(_loop, "_safety_gen_now", lambda: None)
    _loop._gaze_at_speaker(30.0)
    assert _loop._gaze_q.empty()


def test_shutdown_runtime_stops_gaze(monkeypatch):
    import src.spot.app.main as _app
    hit = []
    monkeypatch.setattr(_loop, "stop_gaze_worker", lambda: hit.append("gaze"))
    monkeypatch.setattr(_app._sl.LISTENER, "stop", lambda: hit.append("listener"))
    monkeypatch.setattr(_app._ring.RING, "_anim", None)
    monkeypatch.setattr(_app._cap.CAPTURE, "_mic", None)
    monkeypatch.setattr(_app, "cleanup_spot", lambda: hit.append("cleanup"))
    _app._shutdown_runtime()
    assert hit[:3] == ["listener", "gaze", "cleanup"]
