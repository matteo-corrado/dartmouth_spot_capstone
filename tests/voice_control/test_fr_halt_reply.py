"""Final-review I1: a safety halt during the user's utterance blocks only that
turn's motion; the reply still plays, and the talk pose comes back on the
next motion-allowed turn (the halt itself is unchanged: stop first)."""
import numpy as np
import pytest

import src.voice_control.client_mic as cm
import src.voice_control.spot_dispatch as d
from tests.voice_control.test_arm_policy import FakeCmd, FakeState


class _Brain:
    on_token_callback = None

    def __init__(self, result, on_process=None):
        self.result, self.on_process = result, on_process

    def process(self, text, state):
        if self.on_process is not None:
            self.on_process()
        return self.result


class _TTS:
    def __init__(self):
        self.said = []

    def is_available(self):
        return False  # no streaming chunker: replies go through _say

    def speak(self, text, voice=None):
        self.said.append(text)

    def wait(self):
        pass


def _turn(monkeypatch, brain, utt_t0, last_safety, transcript="what's your next stop after this"):
    sent, resume = [], []
    monkeypatch.setattr(cm, "_brain_ok", True)
    monkeypatch.setattr(cm, "get_spot_state", lambda: {})
    monkeypatch.setattr(cm, "_resolve_voice", lambda b: None)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: transcript)
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)
    monkeypatch.setattr(cm, "is_filler_only", lambda t, **k: False)
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda i, brain=None, speak=None: sent.append(i["intent"]) or True)
    monkeypatch.setattr(cm, "_last_safety_at", last_safety)
    monkeypatch.setattr(cm, "_resume_talk_after_halt", lambda: resume.append("resume"))
    tts = _TTS()
    cm._abort_chain.clear()
    try:
        cm.process_utterance(None, None, [np.zeros(16000, np.float32)], brain=brain,
                             tts=tts, info={}, has_wake_detector=True, utt_t0=utt_t0)
    finally:
        aborted = cm._abort_chain.is_set()
        cm._abort_chain.clear()
    return sent, tts.said, resume, aborted


def test_halt_during_utterance_keeps_reply_blocks_motion(monkeypatch):
    res = {"response": "My next stop is the robotics poster.",
           "actions": [{"intent": "walk", "params": {}}, {"intent": "status", "params": {}}]}
    sent, said, resume, aborted = _turn(monkeypatch, _Brain(res), utt_t0=49.0, last_safety=50.0)
    assert said == ["My next stop is the robotics poster."]
    assert sent == ["status"]          # motion skipped, info action kept
    assert resume == []                # no arm motion in a blocked turn
    assert not aborted


def test_conversation_only_reply_survives_earlier_halt(monkeypatch):
    sent, said, _, _ = _turn(monkeypatch, _Brain({"response": "Hello there!", "actions": []}),
                             utt_t0=49.0, last_safety=50.0)
    assert said == ["Hello there!"]


def test_motion_allowed_turn_resumes_talk_pose(monkeypatch):
    res = {"response": "Okay.", "actions": [{"intent": "walk", "params": {}}]}
    sent, said, resume, _ = _turn(monkeypatch, _Brain(res), utt_t0=49.0, last_safety=10.0)
    assert sent == ["walk"] and resume == ["resume"]


def test_halt_during_processing_still_hushes(monkeypatch):
    # A stop that lands after the turn started (e.g. during Spot's reply)
    # still aborts the turn: nothing is said, nothing moves.
    res = {"response": "Walking!", "actions": [{"intent": "walk", "params": {}}]}
    sent, said, _, aborted = _turn(monkeypatch, _Brain(res, on_process=cm._mark_safety),
                                   utt_t0=1.0, last_safety=0.0, transcript="walk forward")
    assert said == [] and sent == [] and aborted


# ------------------------------------------------------------ dispatcher side
class _Mouth:
    def __init__(self):
        self.enabled, self.log = True, []

    def disable(self):
        self.enabled = False
        self.log.append("off")

    def enable(self):
        self.enabled = True
        self.log.append("on")


@pytest.fixture
def robot(monkeypatch):
    cmd = FakeCmd()
    sess = {"cmd": cmd, "state": FakeState(), "robot": object()}
    m = _Mouth()
    import src.voice_control.spot_tts as tts_mod
    monkeypatch.setattr(tts_mod, "get_tts", lambda: type("T", (), {"mouth": m})())
    monkeypatch.setattr(d, "_spot_session", sess)
    monkeypatch.setattr(d, "_session_context", object())
    monkeypatch.setattr(d, "block_until_arm_arrives", lambda c, i, t: True)
    monkeypatch.setattr(d, "_arm_pose", "talk")
    monkeypatch.setattr(d, "_pose_before_walk", None)
    monkeypatch.setattr(d, "_mouth_resume", False)
    monkeypatch.setattr(d, "_talk_resume", False)
    monkeypatch.setattr(d, "_last_brain", None)
    monkeypatch.setattr(d, "_standing_and_powered", lambda s: True)
    yield cmd, m
    d._cancel_nav()


def test_stop_from_talk_still_stops_first_then_resume_restores(robot):
    cmd, m = robot
    assert d.dispatch_intent({"intent": "stop", "params": {}}) is True
    assert cmd.kinds()[0] == "stop"                # halt first, unchanged
    assert d.get_arm_pose() == "stowed" and m.log == ["off"]
    assert d._talk_resume is True
    t = d.resume_talk_after_halt()
    assert t
    t.join(2)                                       # mouth only after the arm arrives
    assert d.get_arm_pose() == "talk" and m.log == ["off", "on"]
    assert cmd.kinds()[-1] == "arm_ready"
    assert d.resume_talk_after_halt() is False     # one-shot


def test_no_resume_when_not_in_talk_or_after_estop(robot, monkeypatch):
    monkeypatch.setattr(d, "_arm_pose", "stowed")
    d.dispatch_intent({"intent": "stop", "params": {}})
    assert d._talk_resume is False
    monkeypatch.setattr(d, "_arm_pose", "talk")
    d.dispatch_intent({"intent": "estop", "params": {}})
    assert d._talk_resume is False


def test_resume_cleared_by_sit(robot):
    d.dispatch_intent({"intent": "stop", "params": {}})
    assert d._talk_resume is True
    d.dispatch_intent({"intent": "sit", "params": {}})
    assert d._talk_resume is False and d.resume_talk_after_halt() is False


def test_resume_loses_to_a_racing_stop(robot, monkeypatch):
    cmd, m = robot
    d.dispatch_intent({"intent": "stop", "params": {}})
    monkeypatch.setattr(d, "_aborted", lambda gen: True)  # a stop bumped the generation
    assert d.resume_talk_after_halt() is False
    assert d.get_arm_pose() == "stowed" and m.log == ["off"]


def test_resume_keeps_mouth_off_if_arm_never_arrives(robot, monkeypatch):
    cmd, m = robot
    d.dispatch_intent({"intent": "stop", "params": {}})
    monkeypatch.setattr(d, "block_until_arm_arrives", lambda c, i, t: False)
    d.resume_talk_after_halt().join(2)
    assert m.log == ["off"] and not m.enabled


def test_stop_during_arm_raise_keeps_mouth_off(robot, monkeypatch):
    cmd, m = robot
    d.dispatch_intent({"intent": "stop", "params": {}})

    def arrives_after_a_stop(c, i, t):
        d.dispatch_intent({"intent": "stop", "params": {}})  # stop lands mid-raise
        return True
    monkeypatch.setattr(d, "block_until_arm_arrives", arrives_after_a_stop)
    d.resume_talk_after_halt().join(2)
    assert m.log == ["off", "off"] and d.get_arm_pose() == "stowed"


def test_hush_skips_stream_rebuild_when_nothing_plays(monkeypatch):
    calls = []

    class P:
        def is_busy(self):
            return False

        def force_reset(self):
            calls.append("reset")
    monkeypatch.setattr(cm, "_audio_player", P())
    cm._hush_player()
    assert calls == []


def test_resume_waits_while_moving(robot, monkeypatch):
    d.dispatch_intent({"intent": "stop", "params": {}})
    monkeypatch.setattr(d, "is_navigating", lambda: True)
    assert d.resume_talk_after_halt() is False and d._talk_resume is True


# ------------------------------------------------- M2: every stop has a timeout
class _TimedCmd(FakeCmd):
    def __init__(self):
        super().__init__()
        self.timeouts = []

    def robot_command(self, cmd, end_time_secs=None, **kw):
        self.timeouts.append(kw.get("timeout"))
        return super().robot_command(cmd, end_time_secs, **kw)


@pytest.mark.parametrize("name", ["stop", "freeze", "estop"])
def test_safety_path_rpcs_all_have_timeouts(robot, monkeypatch, name):
    cmd = _TimedCmd()
    monkeypatch.setattr(d, "_spot_session", {"cmd": cmd, "state": FakeState(), "robot": object()})
    monkeypatch.setattr(d, "_is_standing", lambda s: True)
    d.dispatch_intent({"intent": name, "params": {}})
    stops = [t for (k, _), t in zip(cmd.sent, cmd.timeouts) if k in ("stop", "mobility")]
    assert len(stops) >= 2 and all(t == d.RPC_TIMEOUT_S for t in stops)


# ------------------------------------------- M3: stow during nav re-stops
@pytest.mark.parametrize("mode", ["graphnav", "visual_nav"])
def test_stow_during_nav_sends_second_stop_after_join(robot, monkeypatch, mode):
    import threading
    cmd, _ = robot
    ev = threading.Event()
    t = threading.Thread(target=ev.wait, args=(5,), daemon=True)
    t.start()
    monkeypatch.setattr(d, "_nav_thread", t)
    monkeypatch.setattr(d, "_nav_stop_event", ev)
    monkeypatch.setattr(d, "_nav_mode", mode)
    d.dispatch_intent({"intent": "stow", "params": {}})
    k = cmd.kinds()
    assert k[:2] == ["stop", "stop"]  # first stop, nav join, then the re-stop
    assert not t.is_alive()
