"""Task 15: main builds the pipeline in the 9-step order; features load
loudly (C6); gaze is not put on the bus (C5); the auto-arm gate stays
set_auto_arm(auto_arm_wanted(...)) (C3); events publish after the change."""
import os
import threading
from types import SimpleNamespace

import pytest

import src.spot.app.main as _app
import src.spot.audio.capture as _cap
import src.spot.conversation.loop as _loop
import src.spot.conversation.utterance_log as _ul
import src.spot.core.events as _ev
import src.spot.mic.ring_control as _ring
import src.spot.operator.control as _ctl
import src.spot.safety.listener as _sl
import src.voice_control.wake.safety_kws as _skws
from src.spot.core.events import EventBus
from src.spot.features import REGISTRY
from src.spot.perception.vision import NullVision

STEPS = ["tuning", "bus", "robot", "features",
         "audio", "ring", "wake", "safety",
         "stt", "tts", "brain", "vision",
         "conversation", "control", "run"]


class _Avail:
    def is_available(self):
        return True

    def reset(self):
        pass


def _patch_pipeline(monkeypatch, seen, load_result=()):
    """Every constructor main reaches records its name; nothing real opens."""
    C = _loop.CONVERSATION
    for obj, attr in ((C, "_store_transcript"), (C, "_brain_ok"), (C, "_state_leds"),
                      (_cap, "MIC_CHANNEL"), (_cap.CAPTURE, "_audio_player"),
                      (_cap.CAPTURE, "_mic"), (_ring.RING, "_anim"), (_ev, "BUS")):
        monkeypatch.setattr(obj, attr, getattr(obj, attr))
    monkeypatch.setenv("SPOT_DRY_RUN", "0")
    monkeypatch.delenv("SPOT_NO_TRANSCRIPT_LOG", raising=False)
    import src.spot.features.mouth.handlers as mouth_h
    monkeypatch.setattr(mouth_h, "_auto_arm", False)  # the step-4 mouth start stays a no-op

    def rec(name, ret=None):
        return lambda *a, **k: seen.append(name) or ret

    real_tuning, real_owners = _app.get_tuning, _app._robot_owners
    monkeypatch.setattr(_app, "get_tuning", lambda: seen.append("tuning") or real_tuning())
    monkeypatch.setattr(_app, "EventBus", lambda: seen.append("bus") or EventBus())
    monkeypatch.setattr(_app, "_robot_owners", lambda: seen.append("robot") or real_owners())
    monkeypatch.setattr(REGISTRY, "load", rec("features", list(load_result)))
    monkeypatch.setattr(_cap.CAPTURE, "calibrate_floor", rec("audio", 1e-3))
    monkeypatch.setattr(_app, "AudioPlayer", lambda **k: object())
    monkeypatch.setattr(_app, "beep", SimpleNamespace(set_player=lambda p: None,
                                                      set_volume=lambda v: None,
                                                      ready=lambda: None))
    monkeypatch.setattr(_app, "VoiceActivityDetector", lambda *a, **k: object())
    monkeypatch.setattr(_cap, "_open_mic_array", lambda: None)
    monkeypatch.setattr(_ring, "_make_ring_animator", rec("ring"))
    monkeypatch.setattr(_app, "make_wake", rec("wake", _Avail()))
    monkeypatch.setattr(_skws, "make_safety_detector", lambda: _Avail())
    monkeypatch.setattr(_sl.LISTENER, "start", rec("safety"))
    monkeypatch.setattr(_app, "make_stt", rec("stt", SimpleNamespace(stub=None)))
    monkeypatch.setattr(_app, "make_tts", rec("tts"))
    monkeypatch.setattr(_app, "make_brain", rec("brain", SimpleNamespace(
        wait_available=lambda timeout_s: False)))
    monkeypatch.setattr(_app, "make_vision", rec("vision", NullVision()))
    monkeypatch.setattr(_ul, "UtteranceLog", rec("conversation"))
    monkeypatch.setattr(_app, "warm_up_async", lambda voice: None)
    monkeypatch.setattr(_cap.CAPTURE, "open", lambda dev: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(_ctl, "_start_control_server", rec("control"))
    monkeypatch.setattr(C, "run", rec("run"))


ARGS = ["--dry-run", "--device", "0", "--output-device", "0"]


def test_wiring_order(monkeypatch):
    seen = []
    _patch_pipeline(monkeypatch, seen)
    _app.main(ARGS)
    assert seen == STEPS


def test_feature_failure_fatal_on_robot(monkeypatch, capsys):
    seen = []
    tb = "Traceback (most recent call last):\nImportError: nope\n"
    _patch_pipeline(monkeypatch, seen, load_result=[("bogus", ImportError("nope"), tb)])
    _app.main(["--device", "0", "--output-device", "0"])
    out = capsys.readouterr().out
    assert "[Features] FAILED: bogus (ImportError: nope)" in out
    assert "ImportError: nope\n" in out and "Traceback" in out
    assert "refusing to start on the robot" in out
    assert seen == ["tuning", "bus", "robot", "features"]  # nothing opened, no robot probe


def test_feature_failure_warns_in_dry_run(monkeypatch, capsys):
    seen = []
    _patch_pipeline(monkeypatch, seen, load_result=[("bogus", ImportError("nope"), "tb\n")])
    _app.main(ARGS)
    out = capsys.readouterr().out
    assert "[Features] FAILED: bogus (ImportError: nope)" in out
    assert "WARNING: --dry-run continues" in out
    assert seen == STEPS


def test_feature_start_failure_is_loud(monkeypatch, capsys):
    from src.spot.core.registry import Feature, Registry
    import src.spot.features as _f

    def boom(ctx):
        raise RuntimeError("start broke")

    reg = Registry()
    reg.add(Feature("broken", start=boom))
    monkeypatch.setattr(_f, "REGISTRY", reg)
    monkeypatch.setattr(reg, "load", lambda names: [])
    args = SimpleNamespace(dry_run=False)
    assert _app._start_features(args, EventBus(), _app._robot_owners(), _app._tuning) is False
    out = capsys.readouterr().out
    assert "[Features] FAILED: broken (RuntimeError: start broke)" in out


def test_gaze_not_on_bus_and_mouth_start_does_not_fold(monkeypatch):
    """C5: gaze's speaker_locked handler is never subscribed (it would run the
    arm RPC on the publisher's thread). C3: the mouth start at step 4 is a
    no-op while auto-arm is off; only set_auto_arm (step 7) turns it on."""
    import src.spot.features.mouth.handlers as mouth_h
    monkeypatch.setattr(mouth_h, "_auto_arm", False)
    monkeypatch.setattr(mouth_h, "_auto_arm_pending", False)
    folds = []
    monkeypatch.setattr(mouth_h, "start_auto_arm", lambda *a, **k: folds.append(a))
    bus = EventBus()
    assert _app._start_features(SimpleNamespace(dry_run=True), bus,
                                _app._robot_owners(), _app._tuning) is True
    assert bus._subs.get("speaker_locked", []) == []
    assert folds == [] and mouth_h._auto_arm_pending is False


def test_auto_arm_gate_unchanged():
    tun = SimpleNamespace(auto_arm=True)
    assert _app.auto_arm_wanted(SimpleNamespace(no_arm=False, dry_run=False), tun) is True
    assert _app.auto_arm_wanted(SimpleNamespace(no_arm=True, dry_run=False), tun) is False
    assert _app.auto_arm_wanted(SimpleNamespace(no_arm=False, dry_run=True), tun) is False


def test_no_main_block_in_app_main():
    """Only the shim runs main as a script: one app.main module object."""
    import inspect
    assert '__name__ == "__main__"' not in inspect.getsource(_app)


def test_publish_is_noop_without_bus_and_never_raises(monkeypatch):
    monkeypatch.setattr(_ev, "BUS", None)
    _ev.publish("robot_stood")  # no bus: nothing
    monkeypatch.setattr(_ev, "BUS", EventBus())
    _ev.publish("not_an_event")  # unknown name is logged, not raised


def test_safety_halt_published_after_halt_returns(monkeypatch):
    order = []
    bus = EventBus()
    bus.subscribe("safety_halt", lambda **p: order.append(("event", p)))
    monkeypatch.setattr(_ev, "BUS", bus)
    monkeypatch.setattr(_loop.CONVERSATION, "_abort_chain", threading.Event())
    lst = _sl.SafetyListener(halt=lambda intent: order.append(("halt", intent["intent"])),
                             hush=lambda: None)
    lst.on_keyword("stop")
    assert order == [("halt", "stop"), ("event", {"intent": "stop"})]


def test_voice_state_changed_published(monkeypatch):
    from src.spot.core.voice_state import VoiceState
    got = []
    bus = EventBus()
    bus.subscribe("voice_state_changed", lambda **p: got.append(p["state"]))
    monkeypatch.setattr(_ev, "BUS", bus)
    monkeypatch.setattr(_loop, "beep", SimpleNamespace())  # chime lookup fails softly
    monkeypatch.setattr(_loop.CONVERSATION, "_state_leds", None)
    monkeypatch.setattr(_loop.CONVERSATION, "_status", {})
    monkeypatch.setattr(_ring.RING, "_ring_state", _ring.RING._ring_state)
    monkeypatch.setattr(_ring.RING, "update", lambda: None)
    _loop._enter_state(VoiceState.LISTENING)
    assert got == [VoiceState.LISTENING]


def test_null_vision():
    from src.spot.perception.factory import make_vision
    from src.spot.perception.vision import Vision
    v = make_vision()
    assert isinstance(v, NullVision) and isinstance(v, Vision)
    assert v.people() == [] and v.objects() == []


def test_non_gaze_subscription_is_on_the_bus(monkeypatch):
    from src.spot.core.registry import Feature, Registry
    import src.spot.features as _f

    def handler(**p):
        pass

    reg = Registry()
    reg.add(Feature("other", subscriptions=(("robot_stood", handler),)))
    monkeypatch.setattr(_f, "REGISTRY", reg)
    monkeypatch.setattr(reg, "load", lambda names: [])
    bus = EventBus()
    assert _app._start_features(SimpleNamespace(dry_run=True), bus,
                                _app._robot_owners(), _app._tuning) is True
    assert bus._subs["robot_stood"] == [handler]


# ---- R29: startup cleanup runs exactly once on any exit after step 5 ----

def _count_shutdown(monkeypatch, real=False):
    calls = []
    orig = _app._shutdown_runtime

    def counted():
        calls.append(1)
        if real:
            orig()
    monkeypatch.setattr(_app, "_shutdown_runtime", counted)
    return calls


def test_ctrl_c_in_brain_wait_shuts_down_once(monkeypatch):
    seen = []
    _patch_pipeline(monkeypatch, seen)

    def wait(timeout_s):
        raise KeyboardInterrupt
    monkeypatch.setattr(_app, "make_brain", lambda *a, **k: SimpleNamespace(wait_available=wait))
    calls = _count_shutdown(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _app.main(ARGS)
    assert calls == [1]
    assert "run" not in seen and "conversation" not in seen


def test_run_shutdown_is_not_repeated(monkeypatch):
    seen = []
    _patch_pipeline(monkeypatch, seen)

    def run(**k):
        k["shutdown"]()
        k["shutdown"]()
        raise KeyboardInterrupt
    monkeypatch.setattr(_loop.CONVERSATION, "run", run)
    calls = _count_shutdown(monkeypatch)
    with pytest.raises(KeyboardInterrupt):
        _app.main(ARGS)
    assert calls == [1]


def test_refusal_shuts_down_once(monkeypatch):
    seen = []
    _patch_pipeline(monkeypatch, seen)
    monkeypatch.setattr(_skws, "make_safety_detector", lambda: None)  # no safety KWS on the robot
    monkeypatch.setattr(_app, "_probe_host", lambda: "127.0.0.1")
    import socket

    class _Sock:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: _Sock())
    calls = _count_shutdown(monkeypatch)
    _app.main(["--device", "0", "--output-device", "0"])
    assert calls == [1] and "run" not in seen


class _FakeRobotSession:
    """A standing robot: records the close sequence (sit, power off, exit)."""

    def __init__(self, log):
        self.log = log
        self.robot = SimpleNamespace(power_off=lambda **k: log.append("power_off"))
        self.session = {"cmd": object(), "robot": self.robot}

    def __enter__(self):
        self.log.append("stood")
        return self.session

    def __exit__(self, *a):
        self.log.append("exit")
        return False


def _patch_robot_close(monkeypatch, log):
    import src.spot.robot as _r
    import src.spot.robot.connection as _conn
    import bosdyn.client.robot_command as _rc
    for attr in ("_session", "_context", "_closing"):
        monkeypatch.setattr(_r.CONNECTION, attr, getattr(_r.CONNECTION, attr))
    monkeypatch.setattr(_conn.time, "sleep", lambda s: None)
    monkeypatch.setattr(_conn._motion, "send_stop", lambda cmd: log.append("stop") or True)
    monkeypatch.setattr(_r.ARM, "close_mouth", lambda: None)
    monkeypatch.setattr(_r.ARM, "stow_now", lambda *a, **k: log.append("stow"))
    monkeypatch.setattr(_r.NAV, "cancel", lambda *a, **k: None)
    monkeypatch.setattr(_rc, "blocking_sit", lambda cmd, timeout_sec: log.append("sit"))
    return _r


def test_ctrl_c_after_map_stand_sits_and_powers_off(monkeypatch):
    """Ctrl-C after the --map bring-up stood the robot, before run(): the
    shutdown runs once and the robot is sat and powered off."""
    import src.voice_control.spot_dispatch as sd
    seen, log = [], []
    _patch_pipeline(monkeypatch, seen)
    _r = _patch_robot_close(monkeypatch, log)
    fake = _FakeRobotSession(log)

    def ensure(map_path=None):
        _r.CONNECTION._context = fake
        _r.CONNECTION._session = fake.__enter__()
        raise KeyboardInterrupt  # operator Ctrl-C right after the stand
    monkeypatch.setattr(sd, "ensure_spot_session", ensure)
    calls = _count_shutdown(monkeypatch, real=True)
    with pytest.raises(KeyboardInterrupt):
        _app.main(["--map", "m", "--device", "0", "--output-device", "0"])
    assert calls == [1]
    assert log == ["stood", "stop", "stow", "sit", "power_off", "exit"]
    assert _r.CONNECTION._session is None and "run" not in seen


def test_failed_bringup_after_stand_still_closes_robot(monkeypatch):
    """connection.ensure: a step after the stand raises (auto-arm start). The
    session stays closable; main closes it BEFORE degrading to dry-run."""
    import src.spot.robot.connection as _conn
    import src.voice_control.spot_dispatch as sd
    seen, log = [], []
    _patch_pipeline(monkeypatch, seen)
    _r = _patch_robot_close(monkeypatch, log)
    fake = _FakeRobotSession(log)
    monkeypatch.setattr(_conn, "spot_session", lambda **k: fake)

    def boom(*a, **k):
        raise RuntimeError("auto-arm failed")
    monkeypatch.setattr(sd, "start_auto_arm", boom)
    _app.main(["--map", "m", "--device", "0", "--output-device", "0"])
    assert log == ["stood", "stop", "stow", "sit", "power_off", "exit"]
    assert os.environ["SPOT_DRY_RUN"] == "1"
    assert seen[-1] == "run"  # degraded to dry-run and kept going


def test_ensure_keeps_live_session_closable(monkeypatch):
    import src.spot.robot.connection as _conn
    import src.voice_control.spot_dispatch as sd
    log = []
    _r = _patch_robot_close(monkeypatch, log)
    monkeypatch.setattr(_r.CONNECTION, "_closing", False)
    monkeypatch.setattr(_r.CONNECTION, "_session", None)
    monkeypatch.setattr(_r.CONNECTION, "_context", None)
    fake = _FakeRobotSession(log)
    monkeypatch.setattr(_conn, "spot_session", lambda **k: fake)
    monkeypatch.setattr(sd, "start_auto_arm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(RuntimeError):
        _r.CONNECTION.ensure()
    assert _r.CONNECTION._session is fake.session and _r.CONNECTION._context is fake
    _r.CONNECTION.close()
    assert log == ["stood", "stop", "stow", "sit", "power_off", "exit"]


def test_ensure_failed_enter_still_clears_context(monkeypatch):
    import src.spot.robot.connection as _conn
    _r = _patch_robot_close(monkeypatch, [])
    monkeypatch.setattr(_r.CONNECTION, "_closing", False)
    monkeypatch.setattr(_r.CONNECTION, "_session", None)
    monkeypatch.setattr(_r.CONNECTION, "_context", None)

    class _Bad:
        def __enter__(self):
            raise RuntimeError("unreachable")
    monkeypatch.setattr(_conn, "spot_session", lambda **k: _Bad())
    with pytest.raises(RuntimeError):
        _r.CONNECTION.ensure()
    assert _r.CONNECTION._session is None and _r.CONNECTION._context is None
