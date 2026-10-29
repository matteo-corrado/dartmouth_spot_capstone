"""client_mic operator-control wiring (spotctl immediate commands)."""
import src.voice_control.client_mic as cm


def test_motion_command_dispatches_and_aborts_chain(monkeypatch):
    sent = []
    monkeypatch.setattr(cm, "execute_on_spot", lambda intent, brain=None: sent.append(intent) or True)
    cm._abort_chain.clear()
    assert cm._on_immediate("stop", None) == {"ok": True}
    assert sent[0]["intent"] == "stop"
    assert cm._abort_chain.is_set()
    cm._abort_chain.clear()


def test_stow_and_freeze_use_dispatcher(monkeypatch):
    sent = []
    monkeypatch.setattr(cm, "execute_on_spot", lambda intent, brain=None: sent.append(intent["intent"]) or False)
    assert cm._on_immediate("freeze", None) == {"ok": False}
    cm._on_immediate("stow", None)
    assert sent == ["freeze", "stow"]
    cm._abort_chain.clear()


def test_spotctl_and_transcript_halts_hush_speech(monkeypatch):
    hushes = []
    monkeypatch.setattr(cm, "_start_hush", lambda: hushes.append(1))
    monkeypatch.setattr(cm, "execute_on_spot", lambda intent, brain=None: True)
    monkeypatch.setattr(cm, "_dispatch_safety", lambda intent: True)
    cm._on_immediate("stop", None)
    cm._on_immediate("freeze", None)
    cm._on_immediate("stow", None)       # not a halt: speech keeps going
    assert hushes == [1, 1]
    assert cm._run_safety("stop") is True
    assert hushes == [1, 1, 1]
    cm._abort_chain.clear()


def test_probe_host_order(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "src.config", types.SimpleNamespace(BOSDYN_ROBOT_IP="10.0.0.9"))
    monkeypatch.setenv("SPOT_IP", "192.168.50.3")
    assert cm._probe_host() == "192.168.50.3"
    monkeypatch.delenv("SPOT_IP")
    assert cm._probe_host() == "10.0.0.9"
    monkeypatch.setitem(sys.modules, "src.config", types.SimpleNamespace(BOSDYN_ROBOT_IP=""))
    assert cm._probe_host() == "192.168.80.3"


def test_mute_unmute_and_status(monkeypatch):
    monkeypatch.setattr(cm, "_operator_muted", False)
    assert cm._on_immediate("mute", None)["muted"] is True
    st = cm._on_immediate("status", None)
    assert st["muted"] is True
    for k in ("state", "locked_deg", "floor_dbfs", "last_transcript",
              "last_latency_s", "muted", "crowd", "voiceprint"):
        assert k in st
    assert cm._on_immediate("unmute", None)["muted"] is False


def test_mute_discards_callback_frames(monkeypatch):
    import numpy as np
    monkeypatch.setattr(cm, "_operator_muted", True)
    while not cm.audio_queue.empty():
        cm.audio_queue.get_nowait()
    cm.audio_callback(np.zeros((480, 2), np.float32), 480, None, None)
    assert cm.audio_queue.empty()


def test_hush_resets_player_and_aborts_chain(monkeypatch):
    calls = []

    class P:
        def is_busy(self):
            return True  # Spot is talking

        def force_reset(self):
            calls.append("reset")

    monkeypatch.setattr(cm, "_audio_player", P())
    monkeypatch.setenv("SPOT_DRY_RUN", "1")
    cm._abort_chain.clear()
    assert cm._on_immediate("hush", None) == {"ok": True}
    assert calls == ["reset"] and cm._abort_chain.is_set()
    cm._abort_chain.clear()


def test_toggle():
    assert cm._toggle(False, None) is True
    assert cm._toggle(True, None) is False
    assert cm._toggle(True, "on") is True
    assert cm._toggle(False, "off") is False
