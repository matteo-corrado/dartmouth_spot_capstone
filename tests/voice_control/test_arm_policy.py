"""Task 14: arm policy, walking pose, speed cap. Fakes only, no robot."""
import threading
import time

import pytest
from bosdyn.api import robot_command_pb2

import src.session as session_mod
import src.voice_control.spot_dispatch as d
from src.graph_nav_utils import capped_mobility_params, capped_travel_params
from src.voice_control.spot_dispatch import arm_policy_for


# ---------------------------------------------------------------- pure policy
def test_locomotion_tucks_then_restores():
    assert arm_policy_for("walk", "talk") == ("walk", "talk")
    assert arm_policy_for("go_to", "talk") == ("walk", "talk")


def test_safety_stows():
    for n in ("stop", "freeze", "estop"):
        assert arm_policy_for(n, "talk") == ("stowed", "stowed")
    assert arm_policy_for("power_off", "talk") == ("stowed", "stowed")


def test_chat_untouched():
    assert arm_policy_for("set_persona", "talk") == ("talk", "talk")


def test_emote_stows_then_restores():
    assert arm_policy_for("emote", "talk") == ("stowed", "talk")


def test_all_locomotion_intents_tuck():
    for n in ("walk_to", "turn", "walk", "strafe", "go_to", "tour", "patrol",
              "come_back", "go_to_object", "follow_me"):
        assert arm_policy_for(n, "talk")[0] == "walk"


# ---------------------------------------------------------------- fakes
def _kind(cmd):
    if cmd.HasField("full_body_command"):
        return "stop" if cmd.full_body_command.HasField("stop_request") else "full_body"
    sc = cmd.synchronized_command
    if sc.HasField("mobility_command"):
        return "mobility"
    if sc.HasField("arm_command"):
        ac = sc.arm_command
        if ac.HasField("named_arm_position_command"):
            return {1: "arm_carry", 2: "arm_ready", 3: "arm_stow"}[
                ac.named_arm_position_command.position]
        if ac.HasField("arm_joint_move_command"):
            return "arm_joints"
        if ac.HasField("arm_gaze_command"):
            return "arm_gaze"
    if sc.HasField("gripper_command"):
        return "gripper"
    return "other"


class FakeCmd:
    def __init__(self, at_goal=True):
        self.sent = []
        self.at_goal = at_goal
        self._lock = threading.Lock()

    def robot_command(self, cmd, end_time_secs=None, **kw):
        with self._lock:
            self.sent.append((_kind(cmd), cmd))
            return len(self.sent)

    def robot_command_feedback(self, cmd_id):
        fb = robot_command_pb2.RobotCommandFeedbackResponse()
        t = fb.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback
        t.status = 1 if self.at_goal else 2
        t.body_movement_status = 2 if self.at_goal else 1
        return fb

    def kinds(self):
        return [k for k, _ in self.sent]


class FakeState:
    def get_robot_state(self):
        from bosdyn.api import robot_state_pb2
        rs = robot_state_pb2.RobotState()
        snap = rs.kinematic_state.transforms_snapshot
        snap.child_to_parent_edge_map["odom"].parent_frame_name = ""
        e = snap.child_to_parent_edge_map["body"]
        e.parent_frame_name = "odom"
        e.parent_tform_child.rotation.w = 1.0
        return rs


@pytest.fixture
def fake(monkeypatch):
    cmd = FakeCmd()
    sess = {"cmd": cmd, "state": FakeState(), "robot": object()}
    monkeypatch.setattr(d, "_spot_session", sess)
    monkeypatch.setattr(d, "_session_context", object())
    monkeypatch.setattr(d, "block_until_arm_arrives", lambda c, i, t: True)
    monkeypatch.setattr(d, "_close_mouth", lambda: None)
    monkeypatch.setattr(d, "_arm_pose", "talk")
    monkeypatch.setattr(d, "_pose_before_walk", None)
    monkeypatch.setattr(d, "_last_brain", None)
    yield cmd
    d._cancel_nav()


def _join_nav():
    t = d._nav_thread
    if t is not None:
        t.join(timeout=5)


# ---------------------------------------------------------------- dispatcher
def test_stop_sends_motion_stop_first_then_stows(fake):
    assert d.dispatch_intent({"intent": "stop", "params": {}}) is True
    # stop FIRST (before any join), original branch's stop, then no-wait stow
    assert fake.kinds() == ["stop", "stop", "gripper", "arm_stow"]
    assert d.get_arm_pose() == "stowed"


@pytest.mark.parametrize("name", ["freeze", "estop"])
def test_freeze_estop_stop_first(fake, name):
    d.dispatch_intent({"intent": name, "params": {}})
    k = fake.kinds()
    assert k[0] == "stop"
    assert k[-2:] == ["gripper", "arm_stow"]
    assert "arm_ready" not in k


def test_stow_after_stop_never_waits(fake, monkeypatch):
    def boom(*a):
        raise AssertionError("safety stow must not wait for arrival")
    monkeypatch.setattr(d, "block_until_arm_arrives", boom)
    assert d.dispatch_intent({"intent": "stop", "params": {}}) is True


def test_walk_tucks_moves_capped_then_restores(fake):
    assert d.dispatch_intent({"intent": "walk", "params": {"direction": "forward", "distance": 1}})
    _join_nav()
    k = fake.kinds()
    # walk pose "stow" default: claw closed, stow, THEN the mobility command
    assert k[:3] == ["gripper", "arm_stow", "mobility"]
    assert k[3:] == ["arm_ready"]          # restored after arrival
    mob = fake.sent[2][1].synchronized_command.mobility_command
    from bosdyn.api.spot import robot_command_pb2 as spot_cmd
    params = spot_cmd.MobilityParams()
    mob.params.Unpack(params)
    assert params.vel_limit.max_vel.linear.x == pytest.approx(0.5)
    assert d.get_arm_pose() == "talk"


def test_stop_mid_walk_stows_and_never_restores(fake):
    fake.at_goal = False
    d.dispatch_intent({"intent": "walk", "params": {"direction": "forward", "distance": 1}})
    assert d.is_navigating()
    d.dispatch_intent({"intent": "stop", "params": {}})
    time.sleep(0.6)
    k = fake.kinds()
    i = k.index("stop")
    assert k[i:] == ["stop", "stop", "gripper", "arm_stow"]
    assert "arm_ready" not in k
    assert d.get_arm_pose() == "stowed"


def test_tuck_failure_refuses_to_move(fake, monkeypatch):
    monkeypatch.setattr(d, "block_until_arm_arrives", lambda c, i, t: False)
    said = []
    assert d.dispatch_intent({"intent": "walk", "params": {}}, speak=said.append) is False
    assert "mobility" not in fake.kinds()
    assert said


def test_walk_with_stowed_arm_moves_arm_never(fake, monkeypatch):
    monkeypatch.setattr(d, "_arm_pose", "stowed")
    d.dispatch_intent({"intent": "strafe", "params": {"direction": "left", "distance": 0.5}})
    _join_nav()
    assert fake.kinds() == ["mobility"]


def test_enable_mouth_refused_while_walking(fake, monkeypatch):
    fake.at_goal = False
    monkeypatch.setattr(d, "_arm_pose", "stowed")
    d.dispatch_intent({"intent": "walk", "params": {}})
    assert d.is_navigating()
    import src.voice_control.spot_tts as tts_mod

    class T:
        mouth = None
    monkeypatch.setattr(tts_mod, "get_tts", lambda: T())

    class R:
        def has_arm(self): return True
        def is_powered_on(self): return True
    d._spot_session["robot"] = R()
    assert d.dispatch_intent({"intent": "enable_mouth"}) is False
    assert "arm_ready" not in fake.kinds()


def test_brain_session_state_mirrors_pose(fake):
    from src.voice_control.chatbot.session_state import SessionState

    class B:
        session_state = SessionState()
    b = B()
    d.dispatch_intent({"intent": "stop", "params": {}}, brain=b)
    assert b.session_state.arm_pose == "stowed"
    assert SessionState().as_dict()["arm_pose"] == "stowed"


# ---------------------------------------------------------------- session.py
def test_walk_pose_variants():
    k = lambda c: _kind(c)
    assert k(session_mod.walk_pose_command("stow")) == "arm_stow"
    assert k(session_mod.walk_pose_command("carry")) == "arm_carry"
    j = session_mod.walk_pose_command("joints")
    assert k(j) == "arm_joints"
    pt = j.synchronized_command.arm_command.arm_joint_move_command.trajectory.points[0]
    assert pt.position.sh1.value == pytest.approx(-2.6)
    assert pt.position.el0.value == pytest.approx(2.6)
    assert pt.time_since_reference.seconds >= 2
    with pytest.raises(ValueError):
        session_mod.walk_pose_command("nope")


def test_move_arm_to_poses(monkeypatch):
    import bosdyn.client.robot_command as rc
    monkeypatch.setattr(rc, "block_until_arm_arrives", lambda c, i, t: True)
    c = FakeCmd()
    assert session_mod.move_arm_to(c, "stowed") is True
    assert c.kinds() == ["gripper", "arm_stow"]
    c = FakeCmd()
    session_mod.move_arm_to(c, "talk")
    assert c.kinds() == ["arm_ready"]
    c = FakeCmd()
    session_mod.move_arm_to(c, "walk")
    assert c.kinds()[-1] == {"stow": "arm_stow", "carry": "arm_carry",
                             "joints": "arm_joints"}[session_mod.WALK_POSE]


# ---------------------------------------------------------------- speed cap
def test_speed_cap_params():
    p = capped_mobility_params(0.5)
    assert p.vel_limit.max_vel.linear.x == pytest.approx(0.5)
    assert p.vel_limit.min_vel.linear.x == pytest.approx(-0.5)
    t = capped_travel_params(0.5)
    assert t.velocity_limit.max_vel.linear.y == pytest.approx(0.5)


def test_visual_nav_velocity_clamped(monkeypatch):
    from src.voice_control import visual_nav
    c = FakeCmd()
    visual_nav._send_velocity({"cmd": c}, 2.0, -2.0, 0.0)
    mob = c.sent[0][1].synchronized_command.mobility_command
    v = mob.se2_velocity_request.velocity.linear
    assert v.x == pytest.approx(0.5) and v.y == pytest.approx(-0.5)


def test_stop_when_stowed_sends_only_stop(fake, monkeypatch):
    monkeypatch.setattr(d, "_arm_pose", "stowed")
    d.dispatch_intent({"intent": "stop", "params": {}})
    assert fake.kinds() == ["stop", "stop"]


def test_sit_stows_before_sitting(fake):
    assert arm_policy_for("sit", "talk") == ("stowed", "stowed")
    d.dispatch_intent({"intent": "sit", "params": {}})
    assert fake.kinds() == ["stop", "gripper", "arm_stow", "mobility"]
    assert d.get_arm_pose() == "stowed"


def test_stop_without_session_never_connects(monkeypatch):
    monkeypatch.setattr(d, "_spot_session", None)
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("no bring-up"))
    monkeypatch.setattr(d, "_close_mouth", lambda: None)
    for n in ("stop", "freeze", "estop"):
        assert d.dispatch_intent({"intent": n, "params": {}}) is False


def test_unfinished_trajectory_stops_and_keeps_tuck(fake, monkeypatch):
    fake.at_goal = False
    monkeypatch.setattr(d, "_arm_pose", "walk")
    monkeypatch.setattr(d, "_pose_before_walk", "talk")
    d._start_restore_watch(fake, 1, time.time() + 0.4)
    _join_nav()
    assert fake.kinds() == ["stop"]       # stopped, no restore
    assert d.get_arm_pose() == "walk"


def test_unknown_body_status_not_trusted():
    class Unknown(FakeCmd):
        def robot_command_feedback(self, cmd_id):
            fb = super().robot_command_feedback(cmd_id)
            fb.feedback.synchronized_feedback.mobility_command_feedback \
                .se2_trajectory_feedback.body_movement_status = 0
            return fb
    ev = threading.Event()
    assert d._wait_trajectory_done(Unknown(), 1, time.time() + 0.3, ev) is False


def test_stop_during_tuck_aborts_walk(fake, monkeypatch):
    calls = {"n": 0}

    def arrive_after_stop(c, i, t):
        if calls["n"] == 0:  # another thread says "stop" while we tuck
            calls["n"] += 1
            d.dispatch_intent({"intent": "stop", "params": {}})
        return True
    monkeypatch.setattr(d, "block_until_arm_arrives", arrive_after_stop)
    assert d.dispatch_intent({"intent": "walk", "params": {}}) is False
    k = fake.kinds()
    assert "mobility" not in k and "arm_ready" not in k
    assert k[-2:] == ["gripper", "arm_stow"]   # re-stowed
    assert d.get_arm_pose() == "stowed"


class _Mouth:
    def __init__(self):
        self.enabled = True
        self.log = []

    def disable(self):
        self.enabled = False
        self.log.append("off")

    def enable(self):
        self.enabled = True
        self.log.append("on")


def test_mouth_latched_off_while_walking_and_restored(fake, monkeypatch):
    m = _Mouth()
    import src.voice_control.spot_tts as tts_mod
    monkeypatch.setattr(tts_mod, "get_tts", lambda: type("T", (), {"mouth": m})())
    monkeypatch.setattr(d, "_mouth_resume", False)
    fake.at_goal = False
    d.dispatch_intent({"intent": "walk", "params": {}})
    assert m.log == ["off"] and not m.enabled   # claw still while walking
    fake.at_goal = True
    _join_nav()
    assert m.log == ["off", "on"] and d.get_arm_pose() == "talk"


def test_sit_bumps_safety_gen(fake):
    g = d._safety_gen
    d.dispatch_intent({"intent": "sit", "params": {}})
    assert d._safety_gen == g + 1


def test_travel_params_keep_arrival_thresholds():
    t = capped_travel_params(0.5)
    assert t.max_distance == pytest.approx(0.5) and t.max_yaw == pytest.approx(0.5)


def test_freeze_never_stands_a_sitting_robot(fake):
    # FakeState reports behavior_state UNKNOWN (not standing)
    d.dispatch_intent({"intent": "freeze", "params": {}})
    assert "mobility" not in fake.kinds()


def test_freeze_holds_stand_when_standing(fake, monkeypatch):
    monkeypatch.setattr(d, "_is_standing", lambda s: True)
    d.dispatch_intent({"intent": "freeze", "params": {}})
    assert fake.kinds()[:3] == ["stop", "stop", "mobility"]


@pytest.mark.parametrize("name", ["sit", "power_off", "stow"])
def test_no_session_shutdown_intents_never_connect(monkeypatch, name):
    monkeypatch.setattr(d, "_spot_session", None)
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("no bring-up"))
    assert d.dispatch_intent({"intent": name, "params": {}}) is False


def test_stand_fault_recovery_never_self_rights(fake):
    calls = []

    class Faulty(FakeCmd):
        def robot_command(self, cmd, end_time_secs=None, **kw):
            calls.append(_kind(cmd))
            if len(calls) == 1:
                raise RuntimeError("BehaviorFaultError")
            return 1

        def clear_behavior_fault(self, behavior_fault_id):
            pass
    d._spot_session["cmd"] = Faulty()
    assert d.dispatch_intent({"intent": "stand", "params": {}}) is True
    # selfright is a full-body command; only synchro stands are allowed here
    assert calls == ["mobility", "mobility"]
