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
        def is_busy(self):
            return True  # Spot is talking

        def force_reset(self):
            resets.append(1)

    monkeypatch.setattr(cm, "_audio_player", P())
    monkeypatch.delenv("SPOT_DRY_RUN", raising=False)
    monkeypatch.setattr(cm, "execute_on_spot", lambda i, brain=None, speak=None: sent.append(i["intent"]) or True)
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)  # label map fallback
    monkeypatch.setattr(cm.beep, "command_ok", lambda: None)
    cm._abort_chain.clear()
    cm._on_safety_kw("estop")
    deadline = time.monotonic() + 1
    while not resets and time.monotonic() < deadline:
        time.sleep(0.01)  # hush runs on its own daemon thread
    assert sent == ["estop"] and resets == [1] and cm._abort_chain.is_set()
    cm._abort_chain.clear()


def test_halt_dispatched_before_slow_hush(monkeypatch):
    # force_reset can take seconds: the halt must not wait for it.
    order = []
    release = threading.Event()

    class SlowPlayer:
        def is_busy(self):
            return True  # Spot is talking

        def force_reset(self):
            order.append("hush-start")
            release.wait(2)

    monkeypatch.setattr(cm, "_audio_player", SlowPlayer())
    monkeypatch.delenv("SPOT_DRY_RUN", raising=False)
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda i, brain=None, speak=None: order.append("dispatch") or True)
    monkeypatch.setattr(cm.beep, "command_ok", lambda: None)
    t0 = time.monotonic()
    cm._on_safety_kw("stop")
    assert time.monotonic() - t0 < 0.5      # returned without waiting on the hush
    assert "dispatch" in order              # halt sent while the hush is still running
    release.set()
    cm._abort_chain.clear()


def test_hush_starts_without_waiting_for_slow_dispatch(monkeypatch):
    # A slow halt RPC must not delay cutting Spot's speech.
    hushed = threading.Event()
    seen_during_dispatch = []

    class P:
        def is_busy(self):
            return True  # Spot is talking

        def force_reset(self):
            hushed.set()

    def slow_dispatch(i, brain=None, speak=None):
        seen_during_dispatch.append(hushed.wait(1.0))
        return True

    monkeypatch.setattr(cm, "_audio_player", P())
    monkeypatch.delenv("SPOT_DRY_RUN", raising=False)
    monkeypatch.setattr(cm, "execute_on_spot", slow_dispatch)
    monkeypatch.setattr(cm.beep, "command_ok", lambda: None)
    cm._on_safety_kw("stop")
    assert seen_during_dispatch == [True]
    cm._abort_chain.clear()


def test_safety_queue_drops_oldest():
    q = queue.Queue(maxsize=2)
    for x in (1, 2, 3):
        cm._put_drop_oldest(q, x)
    assert [q.get_nowait(), q.get_nowait()] == [2, 3]


def test_health_goes_false_after_consecutive_errors(monkeypatch):
    monkeypatch.setattr(cm, "SAFETY_ERROR_WARN_N", 3)

    class Bad:
        def process_frame(self, pcm):
            raise RuntimeError("onnx")

        def reset(self):
            pass

    q, stop, health = queue.Queue(), threading.Event(), {}
    for _ in range(3):
        q.put(b"x")
    t = threading.Thread(target=cm.safety_kws_loop, args=(Bad(), q, stop, lambda k: None),
                         kwargs=dict(health=health), daemon=True)
    t.start()
    deadline = time.monotonic() + 2
    while health.get("ok", True) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert health["ok"] is False
    stop.set()
    t.join(1)


def test_real_detector_errors_reach_health_counter(tmp_path, monkeypatch):
    # SafetyKeywordDetector must not swallow decoder errors, or health never trips.
    from src.voice_control.wake.safety_kws import SafetyKeywordDetector
    monkeypatch.setattr(cm, "SAFETY_ERROR_WARN_N", 3)

    class BrokenStream:
        def accept_waveform(self, sr, samples):
            raise RuntimeError("onnx")

    class Spotter:
        def create_stream(self):
            return BrokenStream()

    det = SafetyKeywordDetector(model_dir=tmp_path)
    det._available, det._spotter, det._stream = True, Spotter(), BrokenStream()
    q, stop, health = queue.Queue(), threading.Event(), {}
    for _ in range(3):
        q.put(b"\x00\x00" * 480)
    t = threading.Thread(target=cm.safety_kws_loop, args=(det, q, stop, lambda k: None),
                         kwargs=dict(health=health), daemon=True)
    t.start()
    deadline = time.monotonic() + 2
    while health.get("ok", True) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert health["ok"] is False
    assert t.is_alive()                      # loop stays up and keeps listening
    stop.set()
    t.join(1)


def test_status_reports_safety_kws_ok(monkeypatch):
    monkeypatch.setattr(cm, "_safety_active", False)
    assert cm._on_immediate("status", None)["safety_kws_ok"] is False


def test_safety_hit_during_utterance_blocks_motion(monkeypatch):
    # A KWS halt lands while the user is still speaking; that utterance's own
    # motion command (regex path) must not dispatch afterwards.
    sent = []
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: "walk forward")
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)
    monkeypatch.setattr(cm, "is_filler_only", lambda t, **k: False)
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda i, brain=None, speak=None: sent.append(i["intent"]) or True)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    monkeypatch.setattr(cm, "_last_safety_at", 50.0)
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], has_wake_detector=True, utt_t0=49.0)
    assert sent == []
    monkeypatch.setattr(cm, "_last_safety_at", 10.0)
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], has_wake_detector=True, utt_t0=49.0)
    assert sent == ["walk"]
    cm._abort_chain.clear()


def test_dry_run_print_hides_raw_words(monkeypatch, capsys):
    monkeypatch.setenv("SPOT_DRY_RUN", "1")
    monkeypatch.setattr(cm, "_store_transcript", False)
    cm.execute_on_spot({"intent": "sit", "params": {}, "raw": "please sit secretword"})
    assert "secretword" not in capsys.readouterr().out


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


def test_halt_survives_hush_thread_start_failure(monkeypatch):
    import src.voice_control.client_mic as cm

    class Boom:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(cm.threading, "Thread", Boom)
    cm._start_hush()  # must not raise


def test_safety_drop_counter_and_rate_limited_log(monkeypatch, capsys):
    monkeypatch.setattr(cm, "_safety_drops", {"total": 0, "pending": 0, "last_log": float("-inf")})
    q = queue.Queue(maxsize=1)
    assert cm._put_drop_oldest(q, 1) is False
    for k, now in enumerate((100.0, 100.1, 101.0, 104.9, 105.0, 105.2)):
        assert cm._put_drop_oldest(q, k) is True
        cm._note_safety_drop(now)
    lines = [ln for ln in capsys.readouterr().out.splitlines() if "queue overflow" in ln]
    # first drop logs at once; next log only >= 5 s later, carrying the 4 drops since
    assert lines == ["[SafetyKWS] !!! queue overflow: dropped 1 frames (safety listener behind)",
                     "[SafetyKWS] !!! queue overflow: dropped 4 frames (safety listener behind)"]
    assert cm._safety_drops["total"] == 6 and cm._safety_drops["pending"] == 1
    assert cm._status_snapshot()["safety_kws_dropped"] == 6


def test_callback_survives_raising_drop_report(monkeypatch):
    def boom(now):
        raise RuntimeError("report")
    monkeypatch.setattr(cm, "_safety_active", True)
    monkeypatch.setattr(cm, "_operator_muted", True)
    monkeypatch.setattr(cm, "_note_safety_drop", boom)
    monkeypatch.setattr(cm, "_safety_q", queue.Queue(maxsize=1))
    frame = np.zeros((480, 2), np.float32)
    cm.audio_callback(frame, 480, None, None)
    cm.audio_callback(frame, 480, None, None)  # overflow -> boom, swallowed
    assert cm._safety_q.qsize() == 1
