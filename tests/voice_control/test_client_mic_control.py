"""client_mic operator-control wiring (spotctl immediate commands)."""
import src.spot.app.main as _app
import src.spot.audio.capture as _cap
import src.spot.conversation.loop as _loop
import src.spot.operator.control as _ctl
import src.spot.safety.listener as _sl
import src.spot.audio.capture as _cap
import src.spot.conversation.loop as _loop
import src.spot.safety.listener as _sl


def test_motion_command_dispatches_and_aborts_chain(monkeypatch):
    sent = []
    monkeypatch.setattr(_loop, "execute_on_spot", lambda intent, brain=None: sent.append(intent) or True)
    _loop.CONVERSATION._abort_chain.clear()
    assert _ctl._on_immediate("stop", None) == {"ok": True}
    assert sent[0]["intent"] == "stop"
    assert _loop.CONVERSATION._abort_chain.is_set()
    _loop.CONVERSATION._abort_chain.clear()


def test_stow_and_freeze_use_dispatcher(monkeypatch):
    sent = []
    monkeypatch.setattr(_loop, "execute_on_spot", lambda intent, brain=None: sent.append(intent["intent"]) or False)
    assert _ctl._on_immediate("freeze", None) == {"ok": False}
    _ctl._on_immediate("stow", None)
    assert sent == ["freeze", "stow"]
    _loop.CONVERSATION._abort_chain.clear()


def test_spotctl_and_transcript_halts_hush_speech(monkeypatch):
    hushes = []
    monkeypatch.setattr(_sl, "_start_hush", lambda: hushes.append(1))
    monkeypatch.setattr(_loop, "execute_on_spot", lambda intent, brain=None: True)
    monkeypatch.setattr(_sl, "_dispatch_safety", lambda intent: True)
    _ctl._on_immediate("stop", None)
    _ctl._on_immediate("freeze", None)
    _ctl._on_immediate("stow", None)       # not a halt: speech keeps going
    assert hushes == [1, 1]
    assert _sl._run_safety("stop") is True
    assert hushes == [1, 1, 1]
    _loop.CONVERSATION._abort_chain.clear()


def test_probe_host_order(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "src.config", types.SimpleNamespace(BOSDYN_ROBOT_IP="10.0.0.9"))
    monkeypatch.setenv("SPOT_IP", "192.168.50.3")
    assert _app._probe_host() == "192.168.50.3"
    monkeypatch.delenv("SPOT_IP")
    assert _app._probe_host() == "10.0.0.9"
    monkeypatch.setitem(sys.modules, "src.config", types.SimpleNamespace(BOSDYN_ROBOT_IP=""))
    assert _app._probe_host() == "192.168.80.3"


def test_mute_unmute_and_status(monkeypatch):
    monkeypatch.setattr(_loop.CONVERSATION, "_operator_muted", False)
    assert _ctl._on_immediate("mute", None)["muted"] is True
    st = _ctl._on_immediate("status", None)
    assert st["muted"] is True
    for k in ("state", "locked_deg", "floor_dbfs", "last_transcript",
              "last_latency_s", "muted", "crowd", "lock", "voiceprint",
              "voiceprint_enrolled", "voiceprint_last_sim", "voiceprint_ready"):
        assert k in st
    assert _ctl._on_immediate("unmute", None)["muted"] is False


def test_mute_discards_callback_frames(monkeypatch):
    import numpy as np
    monkeypatch.setattr(_loop.CONVERSATION, "_operator_muted", True)
    while not _cap.CAPTURE.audio_queue.empty():
        _cap.CAPTURE.audio_queue.get_nowait()
    _cap.audio_callback(np.zeros((480, 2), np.float32), 480, None, None)
    assert _cap.CAPTURE.audio_queue.empty()


def test_hush_resets_player_and_aborts_chain(monkeypatch):
    calls = []

    class P:
        def is_busy(self):
            return True  # Spot is talking

        def force_reset(self):
            calls.append("reset")

    monkeypatch.setattr(_cap.CAPTURE, "_audio_player", P())
    monkeypatch.setenv("SPOT_DRY_RUN", "1")
    _loop.CONVERSATION._abort_chain.clear()
    assert _ctl._on_immediate("hush", None) == {"ok": True}
    assert calls == ["reset"] and _loop.CONVERSATION._abort_chain.is_set()
    _loop.CONVERSATION._abort_chain.clear()


def test_toggle():
    assert _ctl._toggle(False, None) is True
    assert _ctl._toggle(True, None) is False
    assert _ctl._toggle(True, "on") is True
    assert _ctl._toggle(False, "off") is False


def test_lock_toggle_off_drops_lock_and_skips_wake_lock(monkeypatch):
    C = _loop.CONVERSATION
    monkeypatch.setattr(C, "_lock_enabled", True)
    monkeypatch.setattr(_loop, "_gaze_at_speaker", lambda deg: None)
    monkeypatch.setattr(_loop, "_enter_state", lambda s: None)
    monkeypatch.setattr(_cap, "_mic_usable", lambda: False)
    monkeypatch.setattr(_cap, "_mic_speech_degs", lambda a, b: [90.0, 91.0])
    C._lock.lock([90.0], 0.0)
    C.on_queued("lock", "off")
    assert C._lock_enabled is False and not C._lock.locked
    assert _ctl._on_immediate("status", None)["lock"] is False
    C.on_wake(1.0, wake_pcm=b"")
    assert not C._lock.locked and C._status["locked_deg"] is None
    C.on_queued("lock", None)  # no arg flips back on
    assert C._lock_enabled is True
    C.on_wake(2.0, wake_pcm=b"")
    assert C._lock.locked
    C._lock.release()
