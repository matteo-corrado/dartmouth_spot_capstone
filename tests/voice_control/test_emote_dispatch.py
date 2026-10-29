"""Task 15 dispatcher side: emote branch. Fakes only, no robot."""
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest
from bosdyn.api import robot_state_pb2

import src.voice_control.spot_dispatch as d
import src.voice_control.perception.obstacle_query as oq
from src.voice_control.emotes import DANCES
from tests.voice_control.test_arm_policy import FakeCmd, _Mouth


class FakeChoreo:
    def __init__(self, polls_dancing=2, block=None, final=2):
        self.executed = []
        self.polls = polls_dancing
        self.block = block  # Event: keep "dancing" until set
        self.final = final  # status after the dance (2 = COMPLETED_SEQUENCE)

    def execute_choreography(self, name, client_start_time, choreography_starting_slice, **kw):
        self.executed.append((name, client_start_time, choreography_starting_slice))

    def get_choreography_status(self):
        if self.block is not None and not self.block.is_set():
            return SimpleNamespace(status=1), 0.0
        self.polls -= 1
        return SimpleNamespace(status=1 if self.polls > 0 else self.final), 0.0


class FakeRobot:
    def __init__(self, choreo):
        self.choreo = choreo

    def ensure_client(self, name):
        assert name == "choreography"
        return self.choreo


class FakeState:
    def __init__(self, standing=True, powered=True):
        self.standing, self.powered = standing, powered

    def get_robot_state(self):
        rs = robot_state_pb2.RobotState()
        rs.power_state.motor_power_state = (rs.power_state.STATE_ON if self.powered
                                            else rs.power_state.STATE_OFF)
        rs.behavior_state.state = 3 if self.standing else 1
        return rs


CLEAR = lambda r, max_radius_m: (True, None)  # noqa: E731


@pytest.fixture
def env(monkeypatch):
    cmd = FakeCmd()
    choreo = FakeChoreo()
    sess = {"cmd": cmd, "state": FakeState(), "robot": FakeRobot(choreo)}
    monkeypatch.setattr(d, "_spot_session", sess)
    monkeypatch.setattr(d, "_session_context", object())
    monkeypatch.setattr(d, "block_until_arm_arrives", lambda c, i, t: True)
    monkeypatch.setattr(d, "_close_mouth", lambda: None)
    monkeypatch.setattr(d, "_arm_pose", "talk")
    monkeypatch.setattr(d, "_pose_before_walk", None)
    monkeypatch.setattr(d, "_last_routine", None)
    monkeypatch.setattr(d, "_mouth_resume", False)
    monkeypatch.setattr(oq, "clearance_scan", CLEAR)
    yield SimpleNamespace(cmd=cmd, choreo=choreo, sess=sess)
    d._cancel_nav()


def _join():
    if d._nav_thread is not None:
        d._nav_thread.join(timeout=10)


def _wait_executed(env):
    for _ in range(250):
        if env.choreo.executed:
            return
        time.sleep(0.02)


def test_emote_runs_routine_with_arm_stowed_then_restores(env):
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}}) is True
    _join()
    assert [e[0] for e in env.choreo.executed] == ["tail_wag"]
    name, t0, slice0 = env.choreo.executed[0]
    assert slice0 == 0 and t0 > time.time() - 5
    assert env.cmd.kinds() == ["gripper", "arm_stow", "arm_joints"]  # stow, restore
    assert d.get_arm_pose() == "talk"


def test_dance_again_never_repeats(env):
    d.dispatch_intent({"intent": "emote", "params": {"name": "dance"}})
    _join()
    first = env.choreo.executed[-1][0]
    env.choreo.polls = 2
    d.dispatch_intent({"intent": "emote", "params": {"name": "dance"}})
    _join()
    second = env.choreo.executed[-1][0]
    assert first in DANCES and second in DANCES and first != second


def test_refuses_when_crowded(env, monkeypatch):
    monkeypatch.setattr(oq, "clearance_scan",
                        lambda r, max_radius_m: (True, SimpleNamespace(distance_m=0.8)))
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}},
                             speak=said.append) is False
    assert said == ["I need a little more room to dance."]
    assert env.choreo.executed == [] and env.cmd.kinds() == []


def test_refuses_without_usable_grid(env, monkeypatch):
    monkeypatch.setattr(oq, "clearance_scan", lambda r, max_radius_m: (False, None))
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {}}, speak=said.append) is False
    assert said == ["I can't check my surroundings right now, so I'll skip the dance."]
    assert env.choreo.executed == []


def test_refuses_when_obstacle_check_fails(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("grid rpc down")
    monkeypatch.setattr(oq, "clearance_scan", boom)
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {}}, speak=said.append) is False
    assert said and env.choreo.executed == []


def test_rechecks_clearance_right_before_dancing(env, monkeypatch):
    results = iter([(True, None), (True, SimpleNamespace(distance_m=0.5))])
    monkeypatch.setattr(oq, "clearance_scan", lambda r, max_radius_m: next(results))
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}},
                             speak=said.append) is True
    _join()
    assert env.choreo.executed == []           # someone stepped in: no dance
    assert said == ["I need a little more room to dance."]
    assert env.cmd.kinds() == ["gripper", "arm_stow", "arm_joints"]  # arm put back


def test_refuses_when_sitting(env):
    env.sess["state"] = FakeState(standing=False)
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}},
                             speak=said.append) is False
    assert said == ["Ask me to stand up first."]


def test_refuses_while_navigating(env, monkeypatch):
    monkeypatch.setattr(d, "is_navigating", lambda: True)
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "dance"}},
                             speak=said.append) is False
    assert said and env.choreo.executed == []


def test_unknown_routine_refused(env):
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "backflip"}},
                             speak=said.append) is False
    assert said


def test_stop_mid_dance_stops_and_never_restores(env):
    env.choreo.block = threading.Event()  # dances until released
    d.dispatch_intent({"intent": "emote", "params": {"name": "performance_bow"}})
    _wait_executed(env)
    assert env.choreo.executed
    assert d.dispatch_intent({"intent": "stop", "params": {}}) is True
    _join()
    k = env.cmd.kinds()
    i = k.index("stop")
    assert k[i:] == ["stop", "stop", "stop"]  # first, branch, re-sent for the routine
    assert d.get_arm_pose() == "stowed"
    assert not d.is_navigating()


def test_interrupted_routine_stops_and_keeps_arm_stowed(env):
    env.choreo.final = 6  # STATUS_INTERRUPTED
    d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}})
    _join()
    k = env.cmd.kinds()
    assert k == ["gripper", "arm_stow", "stop"]
    assert d.get_arm_pose() == "stowed"


def test_failed_stow_restores_previous_pose(env, monkeypatch):
    monkeypatch.setattr(d, "block_until_arm_arrives", lambda c, i, t: False)
    d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}})
    _join()
    assert env.choreo.executed == []
    assert env.cmd.kinds()[-1] == "arm_joints"
    assert d.get_arm_pose() == "talk"


def test_mouth_off_during_dance(env, monkeypatch):
    m = _Mouth()
    import src.voice_control.spot_tts as tts_mod
    monkeypatch.setattr(tts_mod, "get_tts", lambda: type("T", (), {"mouth": m})())
    env.choreo.block = threading.Event()
    d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}})
    _wait_executed(env)
    assert m.log == ["off"]
    env.choreo.block.set()
    _join()
    assert m.log == ["off", "on"]
    # restored to the folded joints pose (sh0 centred) before the mouth came back
    assert env.cmd.kinds() == ["gripper", "arm_stow", "arm_joints"]
    pt = env.cmd.sent[-1][1].synchronized_command.arm_command.arm_joint_move_command.trajectory.points[0]
    assert pt.position.sh0.value == pytest.approx(0.0)
    assert d.get_arm_pose() == "talk"


def test_grammar_has_emote_on_one_line():
    g = pathlib.Path(d.project_root, "src/voice_control/grammar/spot_action.gbnf").read_text()
    lines = [l for l in g.splitlines() if l.startswith("action-name ::=")]
    assert len(lines) == 1 and '"\\"emote\\""' in lines[0]


def test_stop_not_delayed_by_emote_arm_wait(env, monkeypatch):
    def slow_never(c, i, t):
        time.sleep(t)
        return False
    monkeypatch.setattr(d, "block_until_arm_arrives", slow_never)
    d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}})
    time.sleep(0.5)  # parked in the arm-stow wait
    t = time.time()
    d.dispatch_intent({"intent": "stop", "params": {}})
    assert time.time() - t < 1.0
    _join()
    assert env.choreo.executed == []


def test_stale_completed_status_not_trusted(env):
    env.choreo.polls = 0  # first polls report COMPLETED (left over)
    t = time.time()
    d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}})
    _join()
    assert time.time() - t > 4.0  # waited past the grace window
    assert "stop" in env.cmd.kinds() and d.get_arm_pose() == "stowed"  # untrusted: no restore


def test_stow_mid_dance_re_stops(env):
    env.choreo.block = threading.Event()
    d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}})
    _wait_executed(env)
    d.dispatch_intent({"intent": "stow", "params": {}})
    _join()
    k = env.cmd.kinds()
    i = k.index("stop")
    assert k[i:i + 2] == ["stop", "stop"]  # first stop, then re-stop after the join
    assert not d.is_navigating() and d.get_arm_pose() == "stowed"


def test_closing_blocks_live_session_use(env, monkeypatch):
    monkeypatch.setattr(d, "_closing", True)
    assert d.dispatch_intent({"intent": "stand", "params": {}}) is False
    assert env.cmd.kinds() == []
    with pytest.raises(RuntimeError):
        d.ensure_spot_session()


def test_orphan_emote_thread_never_starts_routine(env):
    stale = d._safety_gen - 1  # a stop landed after the dispatch was checked
    d._emote_routine(env.sess, "tail_wag", "talk", threading.Event(), None, stale)
    assert env.choreo.executed == []


def test_emote_name_normalised_before_resolve(env):
    assert d.dispatch_intent({"intent": "emote", "params": {"name": " Tail Wag "}}) is True
    _join()
    assert [e[0] for e in env.choreo.executed] == ["tail_wag"]
    assert d._last_routine == "tail_wag"


def test_last_routine_only_set_when_routine_started(env, monkeypatch):
    calls = []

    def scan(r, max_radius_m):
        calls.append(1)  # clear at dispatch, crowded at the thread's re-check
        return (True, None) if len(calls) == 1 else (True, SimpleNamespace(distance_m=0.5))

    monkeypatch.setattr(oq, "clearance_scan", scan)
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}}) is True
    _join()
    assert env.choreo.executed == [] and d._last_routine is None
