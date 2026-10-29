"""Task 9: the stop/freeze/estop path keeps the base (2d851a4) call order.

EXPECTED was recorded by running the base commit's spot_dispatch once with
these same fakes (state set on its module globals instead of the owners).
The recording is the truth: never edit EXPECTED to match new code.
Fakes only: no robot, no mic, no network.
"""
import threading
import time

import pytest
from bosdyn.api import robot_state_pb2

import src.spot.robot.arm as arm_mod
import src.spot.robot.dispatcher as dispatcher
import src.spot.features.mouth.handlers as d
import src.spot.features.mouth.handlers as mouth_h
import src.spot.tts.speaker as tts_mod
from src.spot.core.events import EventBus
from src.spot.core.registry import ActionSpec
from src.spot.features import REGISTRY
from src.spot.robot import ARM, CONNECTION, NAV, SAFETY_GEN
from tests.voice_control.test_arm_policy import _kind

EXPECTED = {'talk_walking_standing/stop': {'calls': [['brain_state', 'arm_pose', 'talk'],
                                          ['rpc', 'stop', 2.0],
                                          ['mouth', 'disable'],
                                          ['nav_event_set'],
                                          ['nav_join', 2.0],
                                          ['rpc', 'stop', 2.0],
                                          ['rpc', 'gripper', 2.0],
                                          ['rpc', 'arm_stow', 2.0],
                                          ['brain_state', 'arm_pose', 'stowed']],
                                'final': {'ok': True,
                                          'arm_pose': 'stowed',
                                          'talk_resume': True,
                                          'auto_arm_pending': True,
                                          'gen_delta': 1,
                                          'nav_cleared': True,
                                          'gaze_held': True,
                                          'dispatch_gen': None}},
 'talk_walking_standing/freeze': {'calls': [['brain_state', 'arm_pose', 'talk'],
                                            ['rpc', 'stop', 2.0],
                                            ['mouth', 'disable'],
                                            ['nav_event_set'],
                                            ['nav_join', 2.0],
                                            ['rpc', 'stop', 2.0],
                                            ['rpc', 'mobility', 2.0],
                                            ['rpc', 'gripper', 2.0],
                                            ['rpc', 'arm_stow', 2.0],
                                            ['brain_state', 'arm_pose', 'stowed']],
                                  'final': {'ok': True,
                                            'arm_pose': 'stowed',
                                            'talk_resume': True,
                                            'auto_arm_pending': True,
                                            'gen_delta': 1,
                                            'nav_cleared': True,
                                            'gaze_held': True,
                                            'dispatch_gen': None}},
 'talk_walking_standing/estop': {'calls': [['brain_state', 'arm_pose', 'talk'],
                                           ['rpc', 'stop', 2.0],
                                           ['mouth', 'disable'],
                                           ['nav_event_set'],
                                           ['nav_join', 2.0],
                                           ['rpc', 'stop', 2.0],
                                           ['rpc', 'gripper', 2.0],
                                           ['rpc', 'arm_stow', 2.0],
                                           ['brain_state', 'arm_pose', 'stowed']],
                                 'final': {'ok': True,
                                           'arm_pose': 'stowed',
                                           'talk_resume': False,
                                           'auto_arm_pending': False,
                                           'gen_delta': 1,
                                           'nav_cleared': True,
                                           'gaze_held': True,
                                           'dispatch_gen': None}},
 'talk_emote_standing/stop': {'calls': [['brain_state', 'arm_pose', 'talk'],
                                        ['rpc', 'stop', 2.0],
                                        ['mouth', 'disable'],
                                        ['nav_event_set'],
                                        ['nav_join', 2.0],
                                        ['rpc', 'stop', 2.0],
                                        ['rpc', 'stop', 2.0],
                                        ['rpc', 'gripper', 2.0],
                                        ['rpc', 'arm_stow', 2.0],
                                        ['brain_state', 'arm_pose', 'stowed']],
                              'final': {'ok': True,
                                        'arm_pose': 'stowed',
                                        'talk_resume': True,
                                        'auto_arm_pending': True,
                                        'gen_delta': 1,
                                        'nav_cleared': True,
                                        'gaze_held': True,
                                        'dispatch_gen': None}},
 'talk_emote_standing/freeze': {'calls': [['brain_state', 'arm_pose', 'talk'],
                                          ['rpc', 'stop', 2.0],
                                          ['mouth', 'disable'],
                                          ['nav_event_set'],
                                          ['nav_join', 2.0],
                                          ['rpc', 'stop', 2.0],
                                          ['rpc', 'mobility', 2.0],
                                          ['rpc', 'stop', 2.0],
                                          ['rpc', 'mobility', 2.0],
                                          ['rpc', 'gripper', 2.0],
                                          ['rpc', 'arm_stow', 2.0],
                                          ['brain_state', 'arm_pose', 'stowed']],
                                'final': {'ok': True,
                                          'arm_pose': 'stowed',
                                          'talk_resume': True,
                                          'auto_arm_pending': True,
                                          'gen_delta': 1,
                                          'nav_cleared': True,
                                          'gaze_held': True,
                                          'dispatch_gen': None}},
 'talk_emote_standing/estop': {'calls': [['brain_state', 'arm_pose', 'talk'],
                                         ['rpc', 'stop', 2.0],
                                         ['mouth', 'disable'],
                                         ['nav_event_set'],
                                         ['nav_join', 2.0],
                                         ['rpc', 'stop', 2.0],
                                         ['rpc', 'stop', 2.0],
                                         ['rpc', 'gripper', 2.0],
                                         ['rpc', 'arm_stow', 2.0],
                                         ['brain_state', 'arm_pose', 'stowed']],
                               'final': {'ok': True,
                                         'arm_pose': 'stowed',
                                         'talk_resume': False,
                                         'auto_arm_pending': False,
                                         'gen_delta': 1,
                                         'nav_cleared': True,
                                         'gaze_held': True,
                                         'dispatch_gen': None}},
 'stowed_idle_sitting/stop': {'calls': [['brain_state', 'arm_pose', 'stowed'],
                                        ['rpc', 'stop', 2.0],
                                        ['mouth', 'disable'],
                                        ['rpc', 'stop', 2.0]],
                              'final': {'ok': True,
                                        'arm_pose': 'stowed',
                                        'talk_resume': False,
                                        'auto_arm_pending': True,
                                        'gen_delta': 1,
                                        'nav_cleared': True,
                                        'gaze_held': True,
                                        'dispatch_gen': None}},
 'stowed_idle_sitting/freeze': {'calls': [['brain_state', 'arm_pose', 'stowed'],
                                          ['rpc', 'stop', 2.0],
                                          ['mouth', 'disable'],
                                          ['rpc', 'stop', 2.0]],
                                'final': {'ok': True,
                                          'arm_pose': 'stowed',
                                          'talk_resume': False,
                                          'auto_arm_pending': True,
                                          'gen_delta': 1,
                                          'nav_cleared': True,
                                          'gaze_held': True,
                                          'dispatch_gen': None}},
 'stowed_idle_sitting/estop': {'calls': [['brain_state', 'arm_pose', 'stowed'],
                                         ['rpc', 'stop', 2.0],
                                         ['mouth', 'disable'],
                                         ['rpc', 'stop', 2.0]],
                               'final': {'ok': True,
                                         'arm_pose': 'stowed',
                                         'talk_resume': False,
                                         'auto_arm_pending': False,
                                         'gen_delta': 1,
                                         'nav_cleared': True,
                                         'gaze_held': True,
                                         'dispatch_gen': None}},
 'no_session/stop': {'calls': [['brain_state', 'arm_pose', 'talk'],
                               ['mouth', 'disable'],
                               ['nav_event_set'],
                               ['nav_join', 2.0]],
                     'final': {'ok': False,
                               'arm_pose': 'talk',
                               'talk_resume': False,
                               'auto_arm_pending': True,
                               'gen_delta': 1,
                               'nav_cleared': True,
                               'gaze_held': True,
                               'dispatch_gen': None}},
 'no_session/freeze': {'calls': [['brain_state', 'arm_pose', 'talk'],
                                 ['mouth', 'disable'],
                                 ['nav_event_set'],
                                 ['nav_join', 2.0]],
                       'final': {'ok': False,
                                 'arm_pose': 'talk',
                                 'talk_resume': False,
                                 'auto_arm_pending': True,
                                 'gen_delta': 1,
                                 'nav_cleared': True,
                                 'gaze_held': True,
                                 'dispatch_gen': None}},
 'no_session/estop': {'calls': [['brain_state', 'arm_pose', 'talk'],
                                ['mouth', 'disable'],
                                ['nav_event_set'],
                                ['nav_join', 2.0]],
                      'final': {'ok': False,
                                'arm_pose': 'talk',
                                'talk_resume': False,
                                'auto_arm_pending': True,
                                'gen_delta': 1,
                                'nav_cleared': True,
                                'gaze_held': True,
                                'dispatch_gen': None}}}


class _Cmd:
    def __init__(self, log):
        self.log = log

    def robot_command(self, cmd, end_time_secs=None, **kw):
        self.log.append(["rpc", _kind(cmd), kw.get("timeout")])
        return 1


class _State:
    def __init__(self, standing):
        self.standing = standing

    def get_robot_state(self):
        rs = robot_state_pb2.RobotState()
        rs.power_state.motor_power_state = rs.power_state.STATE_ON
        rs.behavior_state.state = 3 if self.standing else 1
        return rs


class _Mouth:
    enabled = True

    def __init__(self, log):
        self.log = log

    def disable(self):
        self.log.append(["mouth", "disable"])

    def enable(self):
        self.log.append(["mouth", "enable"])


class _Ev(threading.Event):
    def __init__(self, log):
        super().__init__()
        self.log = log

    def set(self):
        self.log.append(["nav_event_set"])
        super().set()


class _Thr:
    def __init__(self, log):
        self.log, self.alive = log, True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.log.append(["nav_join", timeout])
        self.alive = False


class _SS:
    def __init__(self, log):
        object.__setattr__(self, "_log", log)

    def __setattr__(self, k, v):
        self._log.append(["brain_state", k, v])
        object.__setattr__(self, k, v)


class _Brain:
    def __init__(self, log):
        self.session_state = _SS(log)


SCEN = {
    "talk_walking_standing": dict(session=True, pose="talk", nav="walk", standing=True),
    "talk_emote_standing": dict(session=True, pose="talk", nav="emote", standing=True),
    "stowed_idle_sitting": dict(session=True, pose="stowed", nav=None, standing=False),
    "no_session": dict(session=False, pose="talk", nav="walk", standing=True),
}


def _run(monkeypatch, sname, name):
    s = SCEN[sname]
    log = []
    mouth = _Mouth(log)
    monkeypatch.setattr(tts_mod, "get_tts", lambda: type("T", (), {"mouth": mouth})())
    monkeypatch.setattr(arm_mod, "block_until_arm_arrives",
                        lambda *a: pytest.fail("the safety path must not wait for the arm"))
    sess = ({"cmd": _Cmd(log), "state": _State(s["standing"]), "robot": object()}
            if s["session"] else None)
    monkeypatch.setattr(CONNECTION, "_session", sess)
    monkeypatch.setattr(CONNECTION, "_context", object() if s["session"] else None)
    monkeypatch.setattr(CONNECTION, "_closing", False)
    monkeypatch.setattr(ARM, "_pose", s["pose"])
    monkeypatch.setattr(ARM, "_pose_before_walk", None)
    monkeypatch.setattr(ARM, "_pose_during_walk", None)
    monkeypatch.setattr(ARM, "_mouth_resume", False)
    monkeypatch.setattr(ARM, "_talk_resume", False)
    monkeypatch.setattr(ARM, "_last_brain", None)
    monkeypatch.setattr(ARM, "_settle_until", 0.0)
    monkeypatch.setattr(mouth_h, "_auto_arm", True)
    monkeypatch.setattr(mouth_h, "_auto_arm_pending", True)
    monkeypatch.setattr(mouth_h, "_auto_arm_thread", None)
    if s["nav"]:
        monkeypatch.setattr(NAV, "_thread", _Thr(log))
        monkeypatch.setattr(NAV, "_stop_event", _Ev(log))
        monkeypatch.setattr(NAV, "_mode", s["nav"])
    else:
        monkeypatch.setattr(NAV, "_thread", None)
        monkeypatch.setattr(NAV, "_stop_event", None)
        monkeypatch.setattr(NAV, "_mode", None)
    g0 = SAFETY_GEN.current()
    t_before = time.monotonic()
    ok = dispatcher.dispatch_intent({"intent": name, "params": {}}, brain=_Brain(log))
    final = {"ok": ok, "arm_pose": ARM._pose, "talk_resume": ARM._talk_resume,
             "auto_arm_pending": d._auto_arm_pending, "gen_delta": SAFETY_GEN.current() - g0,
             "nav_cleared": NAV._thread is None and NAV._mode is None,
             "gaze_held": ARM._settle_until > t_before,
             "dispatch_gen": SAFETY_GEN.dispatch_current()}
    return {"calls": log, "final": final}


@pytest.mark.parametrize("key", list(EXPECTED))
def test_stop_path_order_unchanged(monkeypatch, key):
    sname, name = key.split("/")
    assert _run(monkeypatch, sname, name) == EXPECTED[key]


@pytest.mark.parametrize("name", ["stop", "freeze", "estop"])
def test_halt_does_not_touch_bus(monkeypatch, name):
    """Review Focus 4: the halt never goes through the event bus (or the
    feature registry), so a failing handler can't delay or block it."""
    def no_bus(self, event, **payload):
        raise AssertionError(f"halt published {event}")

    def no_registry(n):
        raise AssertionError(f"halt consulted the registry for {n}")
    monkeypatch.setattr(EventBus, "publish", no_bus)
    monkeypatch.setattr(REGISTRY, "action", no_registry)
    out = _run(monkeypatch, "talk_walking_standing", name)
    assert out == EXPECTED[f"talk_walking_standing/{name}"]


def test_registry_dispatch_none_when_unregistered():
    assert REGISTRY.action("__never_registered__") is None
    assert dispatcher.registry_dispatch("__never_registered__", {}, None) is None


def test_registered_false_is_not_redispatched_to_legacy(monkeypatch):
    seen = []

    def handler(params, ctx):
        seen.append((params, ctx.arm is ARM, ctx.connection is CONNECTION,
                     ctx.nav is NAV, ctx.safety is SAFETY_GEN))
        return False
    spec = ActionSpec(name="t9_probe", handler=handler, grammar_params=None, prompt_line=None)
    monkeypatch.setattr(REGISTRY, "_actions", {"t9_probe": spec})
    monkeypatch.setattr(dispatcher, "_unknown_intent",
                        lambda *a, **k: pytest.fail("a registered False must not fall through"))
    assert dispatcher.dispatch_intent({"intent": "t9_probe", "params": {"x": 1}}) is False
    assert seen == [({"x": 1}, True, True, True, True)]


@pytest.mark.parametrize("ret", [False, None, True])
def test_registered_result_is_final(monkeypatch, ret):
    spec = ActionSpec(name="t9_probe", handler=lambda p, c: ret, grammar_params=None, prompt_line=None)
    monkeypatch.setattr(REGISTRY, "_actions", {"t9_probe": spec})
    monkeypatch.setattr(dispatcher, "_unknown_intent",
                        lambda *a, **k: pytest.fail("a registered action must not fall through"))
    assert dispatcher.dispatch_intent({"intent": "t9_probe", "params": {}}) is ret


def test_unregistered_falls_through_to_legacy(monkeypatch):
    calls = []
    monkeypatch.setattr(REGISTRY, "_actions", {})  # Task 13 registers set_speed; probe an empty registry
    monkeypatch.setattr(dispatcher, "_unknown_intent",  # the old legacy fall-through
                        lambda name, session=None:
                        calls.append(name) or True)
    assert dispatcher.dispatch_intent({"intent": "set_speed", "params": {}}) is True
    assert calls == ["set_speed"]
