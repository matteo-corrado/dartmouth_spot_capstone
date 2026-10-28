"""Always-on safety KWS thread (every state, ignores mute/gate) + fail-closed."""
import queue
import threading
import time

import numpy as np

import src.voice_control.client_mic as cm


class FakeDet:
    def __init__(self):
        self.frames, self.resets = 0, 0

    def process_frame(self, pcm):
        self.frames += 1
        return "stop" if pcm == b"S" else ""

    def reset(self):
        self.resets += 1


def _run_loop(frames, refractory_s, clock):
    q, stop, hits, det = queue.Queue(), threading.Event(), [], FakeDet()
    for f in frames:
        q.put(f)
    t = threading.Thread(target=cm.safety_kws_loop,
                         args=(det, q, stop, hits.append),
                         kwargs=dict(refractory_s=refractory_s, clock=clock), daemon=True)
    t.start()
    deadline = time.monotonic() + 2
    while not q.empty() and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    t0 = time.monotonic()
    stop.set()
    t.join(1.0)
    return hits, det, t, time.monotonic() - t0


def test_hit_dispatches_once_and_refractory_honoured():
    now = [100.0]
    hits, det, t, _ = _run_loop([b"S", b"S", b"x"], 1.5, lambda: now[0])
    assert hits == ["stop"] and det.resets == 1
    assert det.frames == 1  # frames inside the refractory window are dropped


def test_hit_after_refractory_fires_again():
    times = iter([100.0, 102.0, 102.0])  # hit, refractory check, hit
    hits, *_ = _run_loop([b"S", b"S"], 1.5, lambda: next(times))
    assert hits == ["stop", "stop"]


def test_stop_event_ends_thread_fast():
    hits, det, t, dt = _run_loop([], 1.5, time.monotonic)
    assert not t.is_alive() and dt < 0.5


def test_handler_error_does_not_kill_thread():
    q, stop, det = queue.Queue(), threading.Event(), FakeDet()
    calls = []

    def boom(kw):
        calls.append(kw)
        raise RuntimeError("x")

    t = threading.Thread(target=cm.safety_kws_loop, args=(det, q, stop, boom),
                         kwargs=dict(refractory_s=0.0), daemon=True)
    t.start()
    q.put(b"S")
    q.put(b"S")
    time.sleep(0.3)
    stop.set()
    t.join(1.0)
    assert calls == ["stop", "stop"]


def _drain(q):
    while not q.empty():
        q.get_nowait()


def test_callback_feeds_safety_queue_even_when_muted_or_gated(monkeypatch):
    monkeypatch.setattr(cm, "_safety_active", True)
    monkeypatch.setattr(cm, "_operator_muted", True)
    monkeypatch.setattr(cm, "_mic_gated", True)
    _drain(cm._safety_q)
    _drain(cm.audio_queue)
    cm.audio_callback(np.zeros((480, 2), np.float32), 480, None, None)
    assert cm._safety_q.qsize() == 1 and cm.audio_queue.empty()
    _drain(cm._safety_q)


def test_on_safety_kw_hushes_aborts_and_dispatches(monkeypatch):
    sent, resets = [], []

    class P:
        def force_reset(self):
            resets.append(1)

    monkeypatch.setattr(cm, "_audio_player", P())
    monkeypatch.delenv("SPOT_DRY_RUN", raising=False)
    monkeypatch.setattr(cm, "execute_on_spot", lambda i, brain=None, speak=None: sent.append(i["intent"]) or True)
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)  # label map fallback
    monkeypatch.setattr(cm.beep, "command_ok", lambda: None)
    cm._abort_chain.clear()
    cm._on_safety_kw("estop")
    assert sent == ["estop"] and resets == [1] and cm._abort_chain.is_set()
    cm._abort_chain.clear()


def test_dry_run_safety_prints_and_skips_dispatch(monkeypatch, capsys):
    monkeypatch.setenv("SPOT_DRY_RUN", "1")
    monkeypatch.setattr(cm, "execute_on_spot", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    monkeypatch.setattr(cm, "_audio_player", None)
    cm._on_safety_kw("freeze")
    assert "[dry-run] SAFETY would dispatch: freeze" in capsys.readouterr().out
    cm._abort_chain.clear()


def test_fail_closed_on_robot_without_detector():
    assert cm.safety_startup_ok(True, False, False) is True
    assert cm.safety_startup_ok(False, True, False) is True   # dry-run warns only
    assert cm.safety_startup_ok(False, False, False) is False  # robot: refuse
    assert cm.safety_startup_ok(False, False, True) is True    # explicit override


def test_raw_transcript_checked_first(monkeypatch):
    seen = []
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: "Stop. Now.")
    monkeypatch.setattr(cm, "check_safety_command",
                        lambda t: seen.append(t) or ({"intent": "stop", "params": {}} if t == "Stop. Now." else None))
    monkeypatch.setattr(cm, "execute_on_spot", lambda i, brain=None, speak=None: True)
    monkeypatch.setenv("SPOT_DRY_RUN", "1")
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], has_wake_detector=True)
    assert seen[0] == "Stop. Now."
    cm._abort_chain.clear()
