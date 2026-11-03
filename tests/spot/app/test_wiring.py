"""Task 15: main builds the pipeline in the 9-step order; features load
loudly (C6); gaze has no bus subscription (C5); the auto-arm gate stays
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
import src.spot.safety.kws as _skws
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
    monkeypatch.setattr(_app, "_robot_owners", lambda *a, **k: seen.append("robot") or real_owners(*a, **k))
    # Robot-mode runs: the step-3b TCP probe never touches the network (reachable)
    monkeypatch.setattr(_app, "_tcp_probe", lambda host: None)
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
    """C5: nothing subscribes speaker_locked (gaze declares no subscription;
    the conversation loop calls gaze directly, off the publisher's thread).
    C3: the mouth start at step 4 is a no-op while auto-arm is off; only
    set_auto_arm (step 7) turns it on."""
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
    import src.spot.robot.connection as sd
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
    import src.spot.features.mouth.handlers as mouth_h
    seen, log = [], []
    _patch_pipeline(monkeypatch, seen)
    _r = _patch_robot_close(monkeypatch, log)
    fake = _FakeRobotSession(log)
    monkeypatch.setattr(_conn, "spot_session", lambda **k: fake)

    def boom(*a, **k):
        raise RuntimeError("auto-arm failed")
    monkeypatch.setattr(mouth_h, "start_auto_arm", boom)
    _app.main(["--map", "m", "--device", "0", "--output-device", "0"])
    assert log == ["stood", "stop", "stow", "sit", "power_off", "exit"]
    assert os.environ["SPOT_DRY_RUN"] == "1"
    assert seen[-1] == "run"  # degraded to dry-run and kept going


def test_ensure_keeps_live_session_closable(monkeypatch):
    import src.spot.robot.connection as _conn
    import src.spot.features.mouth.handlers as mouth_h
    log = []
    _r = _patch_robot_close(monkeypatch, log)
    monkeypatch.setattr(_r.CONNECTION, "_closing", False)
    monkeypatch.setattr(_r.CONNECTION, "_session", None)
    monkeypatch.setattr(_r.CONNECTION, "_context", None)
    fake = _FakeRobotSession(log)
    monkeypatch.setattr(_conn, "spot_session", lambda **k: fake)
    monkeypatch.setattr(mouth_h, "start_auto_arm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
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


def test_failed_close_refuses_instead_of_dry_run(monkeypatch, capsys):
    """The half-up close raises partway (send_stop, once): the robot is still
    up, so main refuses (no dry-run degrade) and its shutdown retries the close."""
    import src.spot.robot.connection as _conn
    import src.spot.features.mouth.handlers as mouth_h
    seen, log = [], []
    _patch_pipeline(monkeypatch, seen)
    _r = _patch_robot_close(monkeypatch, log)
    fake = _FakeRobotSession(log)
    monkeypatch.setattr(_conn, "spot_session", lambda **k: fake)
    monkeypatch.setattr(mouth_h, "start_auto_arm", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    fails = [1]

    def send_stop(cmd):
        if fails:
            fails.pop()
            raise RuntimeError("stop rpc failed")
        log.append("stop")
        return True
    monkeypatch.setattr(_conn._motion, "send_stop", send_stop)
    calls = _count_shutdown(monkeypatch, real=True)
    _app.main(["--map", "m", "--device", "0", "--output-device", "0"])
    out = capsys.readouterr().out
    assert "ROBOT STILL POWERED" in out
    assert os.environ["SPOT_DRY_RUN"] == "0"  # no dry-run degrade
    assert calls == [1] and "run" not in seen
    assert log == ["stood", "stop", "stow", "sit", "power_off", "exit"]  # closed by the shutdown retry
    assert _r.CONNECTION._session is None


# ---- Wave 5 safety-gate warnings (main.py shutdown / bring-up / pre-try) ----

def test_ctrl_c_inside_first_shutdown_is_retried_and_powers_off(monkeypatch):
    """A Ctrl-C lands inside the first (pre-run) shutdown, in LISTENER.stop:
    the guard is not spent, so main's except path retries and the --map robot
    is sat and powered off."""
    import src.spot.robot.connection as _conn
    seen, log = [], []
    _patch_pipeline(monkeypatch, seen)
    _r = _patch_robot_close(monkeypatch, log)
    fake = _FakeRobotSession(log)
    monkeypatch.setattr(_conn, "spot_session", lambda **k: fake)

    def no_stream(dev):
        raise OSError("no input device")
    monkeypatch.setattr(_cap.CAPTURE, "open", no_stream)  # step 7 refuses -> shutdown()
    hits = [1]

    def stop():
        if hits:
            hits.pop()
            raise KeyboardInterrupt  # first Ctrl-C, during the join
    monkeypatch.setattr(_sl.LISTENER, "stop", stop)
    calls = _count_shutdown(monkeypatch, real=True)
    with pytest.raises(KeyboardInterrupt):
        _app.main(["--map", "m", "--device", "0", "--output-device", "0"])
    assert calls == [1, 1]  # interrupted once, retried once
    assert log == ["stood", "stop", "stow", "sit", "power_off", "exit"]
    assert _r.CONNECTION._session is None and "run" not in seen


def test_sdk_missing_after_stand_closes_robot_before_dry_run(monkeypatch):
    """ensure() stood the robot, then a ModuleNotFoundError: main closes the
    half-up session before degrading to dry-run (like any other failure)."""
    seen, log = [], []
    _patch_pipeline(monkeypatch, seen)
    _r = _patch_robot_close(monkeypatch, log)
    fake = _FakeRobotSession(log)

    def ensure(map_path=None):
        _r.CONNECTION._context = fake
        _r.CONNECTION._session = fake.__enter__()
        raise ModuleNotFoundError("No module named 'bosdyn.x'")
    monkeypatch.setattr(_r.CONNECTION, "ensure", ensure)
    _app.main(["--map", "m", "--device", "0", "--output-device", "0"])
    assert log == ["stood", "stop", "stow", "sit", "power_off", "exit"]
    assert os.environ["SPOT_DRY_RUN"] == "1"
    assert seen[-1] == "run"


def test_raise_before_run_try_closes_control_server_and_stream(monkeypatch):
    """C.run raises before its own try: main still stops the control server
    and the input stream, and shuts the runtime down once."""
    seen, closed = [], []
    _patch_pipeline(monkeypatch, seen)
    monkeypatch.setattr(_cap.CAPTURE, "open", lambda dev: SimpleNamespace(
        start=lambda: None, stop=lambda: closed.append("stream.stop"),
        close=lambda: closed.append("stream.close")))
    monkeypatch.setattr(_ctl, "_start_control_server", lambda q: SimpleNamespace(
        stop=lambda: closed.append("ctl.stop")))

    def run(**k):
        raise RuntimeError("before run()'s try")
    monkeypatch.setattr(_loop.CONVERSATION, "run", run)
    calls = _count_shutdown(monkeypatch)
    with pytest.raises(RuntimeError):
        _app.main(ARGS)
    assert closed == ["ctl.stop", "stream.stop", "stream.close"]
    assert calls == [1]


def test_startup_preloads_dispatcher_before_conversation():
    """Fix round 1: the robot-owners step (3) imports the dispatcher, so the
    first stop does not cold-import it before send_stop. Fresh interpreter,
    dry-run, nothing opened."""
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    code = ("import sys, src.spot.app.main as m; "
            "before = 'src.spot.robot.dispatcher' in sys.modules; "
            "m._robot_owners(); "
            "print('PRELOAD', before, 'src.spot.robot.dispatcher' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], cwd=str(root), capture_output=True,
                         text=True, timeout=120, env=dict(os.environ, SPOT_DRY_RUN="1"))
    assert out.returncode == 0, out.stderr[-2000:]
    assert "PRELOAD False True" in out.stdout


def test_robot_owners_step_runs_before_conversation(monkeypatch):
    import sys
    seen = []
    _patch_pipeline(monkeypatch, seen)
    loaded = []
    monkeypatch.setattr(_ul, "UtteranceLog",
                        lambda *a, **k: loaded.append("src.spot.robot.dispatcher" in sys.modules))
    _app.main(ARGS)
    assert loaded == [True]


def _fresh(code, **env):
    import subprocess
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[3]
    return subprocess.run([sys.executable, "-c", code], cwd=str(root), capture_output=True,
                          text=True, timeout=120, env=dict(os.environ, **env))


def test_cleanup_cancel_nav_in_fresh_process():
    """Fix round 1: cleanup_spot's cancel step (from src.spot.nav.runner import
    _cancel_nav) in a process where only src.spot.app.main was imported (a
    Ctrl-C during startup): no circular-import error, the cancel runs."""
    out = _fresh("import sys, src.spot.app.main as m; "
                 "assert 'src.spot.robot' not in sys.modules; "
                 "from src.spot.nav.runner import _cancel_nav, is_navigating; "
                 "_cancel_nav(); print('CANCEL_OK', is_navigating())", SPOT_DRY_RUN="1")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "CANCEL_OK False" in out.stdout
    assert "partially initialized" not in out.stdout + out.stderr


def test_cleanup_spot_cancel_step_in_fresh_process():
    out = _fresh("import src.spot.app.main as m; m.cleanup_spot(); print('CLEANUP_DONE')",
                 SPOT_DRY_RUN="0")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "CLEANUP_DONE" in out.stdout
    assert "cancel_nav error" not in out.stdout


def test_no_sdk_degrades_to_dry_run_in_fresh_process():
    """Fix round 1: with bosdyn missing, _bring_up_robot's --map branch hits
    ModuleNotFoundError and _close_session_if_up never imports the robot
    package (never raises), so the branch degrades to dry-run. This exercises
    that branch in isolation only: a full main() without bosdyn stops earlier,
    at step 3 (_robot_owners: robot mode fails loudly, --dry-run continues
    without the robot package; see the CR #1 tests below)."""
    code = ("import sys; sys.modules['bosdyn'] = None; "
            "import os, src.spot.app.main as m; from types import SimpleNamespace as N; "
            "print('CLOSE', m._close_session_if_up()); "
            "a = N(map='m', dry_run=False, allow_no_safety_kws=False); "
            "ok = m._bring_up_robot(a, N(have_safety_kws=True)); "
            "print('BRINGUP', ok, a.dry_run, os.environ['SPOT_DRY_RUN'])")
    out = _fresh(code, SPOT_DRY_RUN="0")
    assert out.returncode == 0, out.stderr[-2000:]
    assert "CLOSE True" in out.stdout
    assert "SDK not installed" in out.stdout
    assert "BRINGUP True True 1" in out.stdout


def test_unreachable_robot_with_failed_feature_degrades_and_only_warns(monkeypatch, capsys):
    """CR #1: the no-map TCP probe runs before the C6 decision, so an
    unreachable robot (auto dry-run) with a failed feature starts, warning."""
    seen = []
    _patch_pipeline(monkeypatch, seen, load_result=[("bogus", ImportError("nope"), "tb\n")])

    def unreachable(host):
        raise OSError("no route")
    monkeypatch.setattr(_app, "_tcp_probe", unreachable)
    monkeypatch.setattr(_app, "_probe_host", lambda: "192.0.2.1")
    assert _app.main(["--device", "0", "--output-device", "0"]) == 0
    out = capsys.readouterr().out
    assert out.index("Auto-enabling dry-run") < out.index("[Features] FAILED: bogus")
    assert "WARNING: --dry-run continues" in out
    assert "refusing to start on the robot" not in out
    assert os.environ["SPOT_DRY_RUN"] == "1"
    assert seen == STEPS


def test_map_run_with_failed_feature_refuses_before_any_session(monkeypatch, capsys):
    """CR #1: --map has no early probe; C6 refuses at step 4, before the
    eager session / stand (step 6b) and before anything opens."""
    import src.spot.robot as _r
    seen, ensured = [], []
    _patch_pipeline(monkeypatch, seen, load_result=[("bogus", ImportError("nope"), "tb\n")])
    monkeypatch.setattr(_r.CONNECTION, "ensure", lambda *a, **k: ensured.append(1))
    probed = []
    monkeypatch.setattr(_app, "_tcp_probe", lambda host: probed.append(host))
    assert _app.main(["--map", "m", "--device", "0", "--output-device", "0"]) == 1
    assert "refusing to start on the robot" in capsys.readouterr().out
    assert seen == ["tuning", "bus", "robot", "features"]
    assert ensured == [] and probed == []


_NO_SDK_MAIN = """
import sys
sys.modules['bosdyn'] = None
from _pytest.monkeypatch import MonkeyPatch
import tests.spot.app.test_wiring as w
from src.spot.features import REGISTRY
mp = MonkeyPatch()
seen = []
w._patch_pipeline(mp, seen)
mp.setattr(REGISTRY, "load", REGISTRY.__class__.load.__get__(REGISTRY))  # the real load
try:
    rc = w._app.main(ARGV)
    print('RC', rc)
except ModuleNotFoundError as e:
    print('RAISED', e)
print('SEEN', seen)
print('FEATURES', sorted(f.name for f in REGISTRY.features))
print('ROBOT', 'src.spot.robot' in sys.modules, 'src.spot.robot.dispatcher' in sys.modules)
"""


def test_dry_run_without_sdk_runs_with_all_features_in_fresh_process():
    """CR #1: with bosdyn blocked, --dry-run gets past step 3 with a clear
    line (base behaviour), all 8 features load, and the robot package is
    never imported."""
    out = _fresh(_NO_SDK_MAIN.replace("ARGV", repr(ARGS)), SPOT_DRY_RUN="0")
    assert out.returncode == 0, out.stderr[-2000:]
    o = out.stdout
    assert "[dry-run] Spot SDK not installed" in o and "continuing in dry-run" in o
    assert "RC 0" in o
    assert f"SEEN {['tuning', 'bus', 'robot', 'audio', 'ring', 'wake', 'safety', 'stt', 'tts', 'brain', 'vision', 'conversation', 'control', 'run']}" in o
    assert ("FEATURES ['dance', 'follow', 'gaze', 'mouth', 'navigation', 'settings', "
            "'visual_nav', 'walk']") in o
    assert "[Features] FAILED" not in o
    assert "ROBOT False False" in o


def test_robot_mode_without_sdk_fails_loudly_in_fresh_process():
    """CR #1: robot mode still needs the SDK: a loud line and the
    ModuleNotFoundError at step 3, before features, audio or any probe."""
    out = _fresh(_NO_SDK_MAIN.replace("ARGV", repr(["--device", "0", "--output-device", "0"])),
                 SPOT_DRY_RUN="0")
    assert out.returncode == 0, out.stderr[-2000:]
    o = out.stdout
    assert "[Spot] FATAL: Spot SDK not installed" in o
    assert "RAISED" in o and "SEEN ['tuning', 'bus', 'robot']" in o
    assert "TCP-probing" not in o


def test_mouth_stand_in_refuses_robot_start(monkeypatch, capsys):
    """Fix round 2 (W2): the core bound _NoMouth at import; even if the
    registry retry loads the real mouth, robot mode refuses (C6)."""
    import src.spot.robot as _r
    seen = []
    _patch_pipeline(monkeypatch, seen)
    monkeypatch.setattr(_r, "mouth_handlers", _r._NoMouth("ImportError: boom"))
    _app.main(["--device", "0", "--output-device", "0"])
    out = capsys.readouterr().out
    assert "[Features] FAILED: mouth (core binding)" in out
    assert "refusing to start on the robot" in out
    assert seen == ["tuning", "bus", "robot", "features"]


def test_mouth_stand_in_only_warns_in_dry_run(monkeypatch, capsys):
    import src.spot.robot as _r
    seen = []
    _patch_pipeline(monkeypatch, seen)
    monkeypatch.setattr(_r, "mouth_handlers", _r._NoMouth("ImportError: boom"))
    _app.main(ARGS)
    assert "WARNING: --dry-run continues" in capsys.readouterr().out
    assert seen == STEPS


def test_feature_stop_runs_after_session_close_and_never_raises(monkeypatch, capsys):
    """I3: Feature.stop(ctx) runs once per loaded feature, after cleanup_spot
    (the robot close order is untouched); a raising stop is printed only."""
    import src.spot.features as _f
    from src.spot.core.registry import Feature, Registry
    order = []
    reg = Registry()
    reg.add(Feature("a", stop=lambda ctx: order.append(("a", ctx))))
    reg.add(Feature("b", stop=lambda ctx: (_ for _ in ()).throw(RuntimeError("boom"))))
    reg.add(Feature("c", stop=lambda ctx: order.append(("c", ctx))))
    monkeypatch.setattr(_f, "REGISTRY", reg)
    monkeypatch.setattr(_app, "_FEATURE_CTX", "CTX")
    monkeypatch.setattr(_app._sl.LISTENER, "stop", lambda: None)
    monkeypatch.setattr(_app._ring.RING, "_anim", None)
    monkeypatch.setattr(_app._cap.CAPTURE, "_mic", None)
    monkeypatch.setattr(_app, "cleanup_spot", lambda: order.append("cleanup"))
    _app._shutdown_runtime()
    assert order == ["cleanup", ("a", "CTX"), ("c", "CTX")]
    assert "[Features] b stop failed: RuntimeError: boom" in capsys.readouterr().out


def test_feature_stops_skipped_if_session_still_open(monkeypatch, capsys):
    """h4: a failed close (session still set) skips every feature stop."""
    import src.spot.features as _f
    import src.spot.robot as _r
    from src.spot.core.registry import Feature, Registry
    order = []
    reg = Registry()
    reg.add(Feature("a", stop=lambda ctx: order.append("a")))
    monkeypatch.setattr(_f, "REGISTRY", reg)
    monkeypatch.setattr(_r.CONNECTION, "_session", {"cmd": object()})
    _app._stop_features()
    assert order == []
    assert "skipping feature stops" in capsys.readouterr().out


def test_feature_handlers_get_the_process_bus(monkeypatch):
    """M1: dispatcher builds the handler Ctx with the bus main set up."""
    import src.spot.features as _f
    import src.spot.robot.dispatcher as _d
    from src.spot.core.registry import ActionSpec, Feature, Registry
    got = []
    reg = Registry()
    reg.add(Feature("probe", actions=(ActionSpec("probe_x", lambda p, ctx: got.append(ctx.bus) or True,
                                                 None, None, needs_session=False),)))
    monkeypatch.setattr(_f, "REGISTRY", reg)
    bus = EventBus()
    monkeypatch.setattr(_ev, "BUS", bus)
    assert _d._dispatch_feature({"intent": "probe_x", "params": {}}) is True
    assert got == [bus]


def test_exit_codes(monkeypatch):
    """T15: main returns 0 on a normal exit, 1 on a refused start; the launch
    stub passes it to sys.exit."""
    seen = []
    _patch_pipeline(monkeypatch, seen)
    assert _app.main(ARGS) == 0
    seen.clear()
    _patch_pipeline(monkeypatch, seen, load_result=[("bogus", ImportError("nope"), "tb\n")])
    assert _app.main(["--device", "0", "--output-device", "0"]) == 1  # C6 refusal
    seen.clear()
    _patch_pipeline(monkeypatch, seen)
    monkeypatch.setattr(_app, "_bring_up_robot", lambda args, audio: False)  # safety KWS / still powered
    monkeypatch.setattr(_app, "_shutdown_runtime", lambda: None)
    assert _app.main(ARGS) == 1


def test_client_mic_stub_exits_with_main_rc():
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[3] / "src/voice_control/client_mic.py").read_text()
    assert "sys.exit(main())" in src


def test_exit_code_list_devices_0_no_mic_1(monkeypatch):
    """p6: --list-devices exits 0 (even with no mic attached); no mic found exits 1."""
    seen = []
    _patch_pipeline(monkeypatch, seen)
    monkeypatch.setattr(_app, "sd", SimpleNamespace(query_devices=lambda: "devs"))
    monkeypatch.setattr(_cap, "_find_device_by_name", lambda *a: None)
    assert _app.main(["--list-devices"]) == 0
    assert _app.main([]) == 1
    assert "features" not in seen  # returned before any startup step


def test_spotctl_persona_gated_on_settings_feature(monkeypatch, capsys):
    """p5: spotctl persona reports instead of importing settings when off."""
    import src.spot.features as _f
    from src.spot.core.registry import Registry
    monkeypatch.setattr(_f, "REGISTRY", Registry())
    C = _loop.CONVERSATION
    monkeypatch.setattr(C, "brain", object(), raising=False)
    C.on_queued("persona", "pirate")
    out = capsys.readouterr().out
    assert "settings feature not enabled" in out


def test_mouth_not_installed_only_when_disabled(monkeypatch):
    """p4: a missing mouth folder is calm only while mouth is out of ENABLED."""
    import src.spot.features.enabled as _en
    import src.spot.robot as robot_pkg
    gone = ModuleNotFoundError("No module named 'src.spot.features.mouth'",
                               name="src.spot.features.mouth")
    monkeypatch.setattr(_en, "ENABLED", [n for n in _en.ENABLED if n != "mouth"])
    assert robot_pkg._mouth_not_installed(gone) is True
    assert robot_pkg._mouth_not_installed(ImportError("broken")) is False
    monkeypatch.setattr(_en, "ENABLED", list(_en.ENABLED) + ["mouth"])
    assert robot_pkg._mouth_not_installed(gone) is False
