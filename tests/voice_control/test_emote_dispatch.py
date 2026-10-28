"""Task 15 dispatcher side: emote branch. Fakes only, no robot."""
import pathlib
import threading
import time
from types import SimpleNamespace

import pytest

import src.voice_control.spot_dispatch as d
import src.voice_control.perception.obstacle_query as oq
from src.voice_control.emotes import DANCES
from tests.voice_control.test_arm_policy import FakeCmd, FakeState


class FakeChoreo:
    def __init__(self, polls_dancing=2, block=None):
        self.executed = []
        self.polls = polls_dancing
        self.block = block  # Event: keep "dancing" until set

    def execute_choreography(self, name, client_start_time, choreography_starting_slice):
        self.executed.append((name, client_start_time, choreography_starting_slice))

    def get_choreography_status(self):
        if self.block is not None and not self.block.is_set():
            return SimpleNamespace(status=1), 0.0
        self.polls -= 1
        return SimpleNamespace(status=1 if self.polls > 0 else 2), 0.0


class FakeRobot:
    def __init__(self, choreo):
        self.choreo = choreo

    def ensure_client(self, name):
        assert name == "choreography"
        return self.choreo


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
    monkeypatch.setattr(oq, "nearest_obstacle_in_body_frame", lambda r, max_radius_m=3.0: None)
    yield SimpleNamespace(cmd=cmd, choreo=choreo, sess=sess)
    d._cancel_nav()


def _join():
    if d._nav_thread is not None:
        d._nav_thread.join(timeout=10)


def test_emote_runs_routine_with_arm_stowed_then_restores(env):
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}}) is True
    _join()
    assert [e[0] for e in env.choreo.executed] == ["tail_wag"]
    name, t0, slice0 = env.choreo.executed[0]
    assert slice0 == 0 and t0 > time.time() - 5
    k = env.cmd.kinds()
    assert k == ["mobility", "gripper", "arm_stow", "arm_ready"]  # stand, stow, restore
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
    monkeypatch.setattr(oq, "nearest_obstacle_in_body_frame",
                        lambda r, max_radius_m=3.0: SimpleNamespace(distance_m=0.8))
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {"name": "tail_wag"}},
                             speak=said.append) is False
    assert said == ["I need a little more room to dance."]
    assert env.choreo.executed == [] and env.cmd.kinds() == []


def test_refuses_when_obstacle_check_fails(env, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("grid rpc down")
    monkeypatch.setattr(oq, "nearest_obstacle_in_body_frame", boom)
    said = []
    assert d.dispatch_intent({"intent": "emote", "params": {}}, speak=said.append) is False
    assert said and env.choreo.executed == []


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
    for _ in range(250):
        if env.choreo.executed:
            break
        time.sleep(0.02)
    assert env.choreo.executed
    assert d.dispatch_intent({"intent": "stop", "params": {}}) is True
    _join()
    k = env.cmd.kinds()
    i = k.index("stop")
    assert "arm_ready" not in k[i:]
    assert k[i:] == ["stop", "stop"]  # arm already stowed for the dance; stop re-sent
    assert d.get_arm_pose() == "stowed"
    assert not d.is_navigating()


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
    time.sleep(1.8)  # past the stand wait, parked in the arm-stow wait
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
