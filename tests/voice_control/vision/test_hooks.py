"""Task 13: the vision package wired into the grammar, prompt, dispatcher and voice loop.

Fakes only: no robot, no GPU, no LLM.
"""
import importlib
import pathlib
import re
import threading
import time

import numpy as np
import pytest

import src.voice_control.client_mic as cm
import src.voice_control.spot_dispatch as d
from src.voice_control import llm_brain
from src.voice_control.vision.types import ActionResult, Detection
from tests.voice_control.test_arm_policy import FakeCmd, FakeState

# importlib, not ``from vision import goto``: the package re-exports the functions.
G = importlib.import_module("src.voice_control.vision.goto")
FO = importlib.import_module("src.voice_control.vision.follow")
P = importlib.import_module("src.voice_control.vision.people")
W = importlib.import_module("src.voice_control.vision.world")
DE = importlib.import_module("src.voice_control.vision.describe")
FI = importlib.import_module("src.voice_control.vision.find")
SP = importlib.import_module("src.voice_control.vision.speaker")
RT = importlib.import_module("src.voice_control.vision.runtime")

ROOT = pathlib.Path(__file__).resolve().parents[3]
BAD = re.compile(r"\b(stop|halt|freez\w*|e-?stop|estop)\b", re.IGNORECASE)


def _det(label="chair", bearing=0.0, dist=2.0, conf=0.9, source="frontleft_fisheye_image"):
    return Detection(label=label, conf=conf, source=source, box=(0, 0, 10, 10), mask=None,
                     keypoints=None, bearing_deg=bearing, distance_m=dist,
                     distance_method="depth" if dist is not None else None)


# ----------------------------------------------------------------- grammar + prompt
def test_grammar_has_find_and_look_around():
    g = (ROOT / "src/voice_control/grammar/spot_action.gbnf").read_text()
    lines = [l for l in g.splitlines() if l.startswith("action-name ::=")]
    assert len(lines) == 1
    names = set(re.findall(r'"\\"(\w+)\\""', lines[0]))
    assert {"find", "look_around", "describe", "go_to_object", "follow_me"} <= names


def test_prompt_lists_vision_actions_and_routes_do_you_see_to_find():
    p = llm_brain.SYSTEM_PROMPT
    assert "\n- find:" in p and "\n- look_around:" in p
    desc = next(l for l in p.splitlines() if l.startswith("- describe:"))
    assert '"side"' in desc and '"back"' in desc
    rule = next(l for l in p.splitlines() if '"Do you see X?"' in l)
    assert "use find" in rule
    assert '{"action": "find", "params": {"what":' in p
    assert '"description": "person in red shirt"' in p
    assert '"description": "person on my left"' in p
    assert '"description": "me"' in p


def test_history_sanitised_for_all_vision_answers():
    src = (ROOT / "src/voice_control/llm_brain.py").read_text()
    assert "_VISION_ANSWER_ACTIONS" in src
    assert set(llm_brain._VISION_ANSWER_ACTIONS) == {"describe", "find", "look_around"}


# ----------------------------------------------------------------- dispatcher fakes
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
    monkeypatch.setattr(d, "_pose_during_walk", None)
    monkeypatch.setattr(d, "_gaze_deg", 0.0)
    monkeypatch.setattr(d, "_auto_arm", False)
    monkeypatch.setattr(d, "_auto_arm_pending", False)
    monkeypatch.setattr(d, "_last_brain", None)
    monkeypatch.setattr(d, "_last_nav_ok", None)
    monkeypatch.setattr(d, "_standing_and_powered", lambda s: True)
    monkeypatch.setattr(d, "_list_saved_locations", lambda: {})
    wo = importlib.import_module("src.voice_control.world_objects")
    monkeypatch.setattr(wo, "find_navigation_target", lambda robot, desc: None)
    yield sess
    d._cancel_nav()


def _join_nav():
    t = d._nav_thread
    if t is not None:
        t.join(timeout=5)


class Rec:
    """Records the goto/follow call; returns ``result``."""

    def __init__(self, result):
        self.result, self.calls, self.dispatch_gen_seen = result, [], "unset"
        self.go = threading.Event()
        self.go.set()

    def goto(self, io, target, label, is_person, stop_event, **kw):
        self.go.wait(5)
        self.calls.append(("goto", io, target, label, is_person, stop_event))
        self.dispatch_gen_seen = d._dispatch_gen
        return self.result

    def follow(self, io, target, stop_event, **kw):
        self.calls.append(("follow", io, target, stop_event))
        self.dispatch_gen_seen = d._dispatch_gen
        return self.result


def test_find_and_look_around_are_answers_without_a_session(monkeypatch):
    monkeypatch.setattr(d, "_spot_session", None)

    def no_session(*a, **k):
        raise AssertionError("vision answers must not bring up a session in the dispatcher")
    monkeypatch.setattr(d, "ensure_spot_session", no_session)
    for name in ("describe", "find", "look_around"):
        assert d.dispatch_intent({"intent": name, "params": {}}) is True
        assert name in d._UNLOCKED_INTENTS


def test_go_to_object_walks_to_nearest_on_nav_thread_with_request_gen(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    far, near = _det(dist=4.0, bearing=30.0), _det(dist=1.8, bearing=-20.0)
    seen = {}

    def snap(robot, sources=None, model="yolo26l-seg", labels=None, with_depth=True):
        seen.update(model=model, labels=labels)
        return [far, _det(label="person", dist=1.0), near]
    monkeypatch.setattr(W, "snapshot", snap)
    said = []
    gen = d._safety_gen
    rec.go.clear()  # the walk body runs only after the dispatch returned
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "the red chair"}},
                             speak=said.append) is True
    assert d._nav_mode == "visual_nav"
    rec.go.set()
    _join_nav()
    (_, io, target, label, is_person, ev), = rec.calls
    assert target is near and label == "chair" and is_person is False
    assert seen == {"model": "yolo26l-seg", "labels": {"chair"}}
    assert io.gen == gen                  # captured when the request was dispatched ...
    assert rec.dispatch_gen_seen is None  # ... not read at pickup (dispatch_gen is gone by then)
    assert said == ["I'm here."]
    assert d.last_nav_ok() is True


def test_go_to_object_not_seen_speaks_and_does_not_move(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [_det(label="person")])
    said = []
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "chair"}},
                             speak=said.append) is False
    assert rec.calls == [] and not d.is_navigating()
    assert said == [d.CANT_SEE_THAT_LINE]
    assert "mobility" not in fake["cmd"].kinds()


def test_go_to_object_unknown_word_speaks_and_does_not_move(fake, monkeypatch):
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: pytest.fail("no label, no snapshot"))
    said = []
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "flibbertigibbet"}},
                             speak=said.append) is False
    assert said == [d.CANT_SEE_THAT_LINE] and not d.is_navigating()


def test_go_to_object_yoloe_label_uses_yoloe_model(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    voc = importlib.import_module("src.voice_control.vision.vocab")
    monkeypatch.setattr(voc, "resolve_label", lambda p: ("yoloe", "door") if p == "door" else None)
    seen = {}

    def snap(robot, sources=None, model="yolo26l-seg", labels=None, with_depth=True):
        seen["model"] = model
        return [_det(label="door", dist=3.0)]
    monkeypatch.setattr(W, "snapshot", snap)
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "the door"}})
    _join_nav()
    assert seen["model"] == "yoloe-26l-seg" and rec.calls[0][3] == "door"


@pytest.mark.parametrize("desc,label", [
    ("red chair", "chair"), ("the cup on the floor", "cup"), ("dining table", "dining table"),
    ("backpacks", "backpack"), ("flibbertigibbet", None)])
def test_object_label_resolution(desc, label):
    hit = d._vision_object_label(desc)
    assert (hit[1] if hit else None) == label


@pytest.mark.parametrize("desc,person", [
    ("me", True), ("Me.", True), ("person in red shirt", True), ("person on my left", True),
    ("the one on the left", True), ("the man in the blue jacket", True),
    ("someone wearing a hat", True), ("the guy with the red shirt", True),
    ("red chair", False), ("the chair on my left", False), ("the cup on the floor", False),
    ("door", False), ("the bag with the stripes", False)])
def test_is_person_phrase(desc, person):
    assert d._is_person_phrase(desc) is person


def test_go_to_object_person_uses_select_person(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: pytest.fail("people go through select_person"))
    target = _det(label="person", dist=3.0)
    got = {}

    def select(robot, phrase, speak, gaze):
        got.update(phrase=phrase, robot=robot)
        speak("Can you raise your hand?")
        return target, None
    monkeypatch.setattr(P, "select_person", select)
    said = []
    assert d.dispatch_intent({"intent": "go_to_object",
                              "params": {"description": "person in red shirt"}}, speak=said.append)
    _join_nav()
    assert got == {"phrase": "person in red shirt", "robot": fake["robot"]}
    (_, io, t, label, is_person, _ev), = rec.calls
    assert t is target and label == "person" and is_person is True
    assert said == ["Can you raise your hand?", "I'm here."]


def test_select_person_message_spoken_and_no_motion(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    monkeypatch.setattr(P, "select_person", lambda robot, phrase, speak, gaze: (None, P.MSG_CANT_SEE_YOU))
    said = []
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "me"}},
                             speak=said.append) is False
    assert said == [P.MSG_CANT_SEE_YOU] and rec.calls == [] and not d.is_navigating()


def test_follow_me_selects_speaker_and_follows(fake, monkeypatch):
    rec = Rec(ActionResult(FO.TEXTS["lost"], False))
    monkeypatch.setattr(FO, "follow", rec.follow)
    target = _det(label="person", dist=2.5)
    phrases = []
    monkeypatch.setattr(P, "select_person",
                        lambda robot, phrase, speak, gaze: phrases.append(phrase) or (target, None))
    said = []
    gen = d._safety_gen
    assert d.dispatch_intent({"intent": "follow_me", "params": {}}, speak=said.append) is True
    assert d._nav_mode == "follow"
    _join_nav()
    (_, io, t, _ev), = rec.calls
    assert phrases == ["me"] and t is target and io.gen == gen
    assert said == [FO.TEXTS["lost"]]
    assert d.last_nav_ok() is False


def test_paused_result_is_not_spoken(fake, monkeypatch):
    monkeypatch.setattr(G, "goto", Rec(ActionResult(G.TEXTS["paused"], False)).goto)
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [_det()])
    said = []
    d.dispatch_intent({"intent": "go_to_object", "params": {"description": "chair"}},
                      speak=said.append)
    _join_nav()
    assert said == []
    assert d.last_nav_ok() is False


def test_not_standing_refuses_before_looking(fake, monkeypatch):
    monkeypatch.setattr(d, "_standing_and_powered", lambda s: False)
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: pytest.fail("must not look"))
    monkeypatch.setattr(P, "select_person", lambda *a, **k: pytest.fail("must not look"))
    for intent in ({"intent": "go_to_object", "params": {"description": "chair"}},
                   {"intent": "follow_me", "params": {}}):
        said = []
        assert d.dispatch_intent(intent, speak=said.append) is False
        assert said == [d.NOT_STANDING_LINE] and not d.is_navigating()


def test_stop_during_selection_prevents_motion(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)

    def select(robot, phrase, speak, gaze):
        d.dispatch_intent({"intent": "stop", "params": {}})  # a stop lands mid-selection
        return _det(label="person"), None
    monkeypatch.setattr(P, "select_person", select)
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "me"}}) is False
    _join_nav()
    assert rec.calls == [] and not d.is_navigating()


def test_selection_gaze_recentred_before_walking(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)

    def select(robot, phrase, speak, gaze):
        d._gaze_deg = 40.0  # the selection aimed the gripper at a candidate
        return _det(label="person"), None
    monkeypatch.setattr(P, "select_person", select)
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "person in red"}})
    _join_nav()
    assert "arm_joints" in fake["cmd"].kinds()  # folded pose re-sent: sh0 back to centre
    assert d._gaze_deg == 0.0


def test_saved_location_resets_nav_outcome(fake, monkeypatch):
    monkeypatch.setattr(d, "_last_nav_ok", False)
    monkeypatch.setattr(d, "_list_saved_locations", lambda: {"kitchen": "wp1"})
    d.dispatch_intent({"intent": "go_to_object", "params": {"description": "kitchen"}})
    assert d.last_nav_ok() is None


def test_superseded_run_does_not_overwrite_outcome(fake, monkeypatch):
    release = threading.Event()

    def slow_goto(io, target, label, is_person, stop_event, **kw):
        release.wait(5)
        return ActionResult("I'm here.", True)
    monkeypatch.setattr(G, "goto", slow_goto)
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [_det()])
    d.dispatch_intent({"intent": "go_to_object", "params": {"description": "chair"}})
    old = d._nav_thread
    monkeypatch.setattr(d, "_list_saved_locations", lambda: {"kitchen": "wp1"})
    d.dispatch_intent({"intent": "go_to_object", "params": {"description": "kitchen"}})
    release.set()
    old.join(5)
    assert d.last_nav_ok() is None


def test_selection_gaze_waits_for_settle_and_counts_already_aimed(fake, monkeypatch):
    calls = []
    monkeypatch.setattr(d, "gaze_at_bearing", lambda b, gen=None: calls.append(time.monotonic()) or True)
    t0 = time.monotonic()
    monkeypatch.setattr(d, "_arm_settle_until", t0 + 0.3)
    monkeypatch.setattr(d, "_gaze_t", 0.0)
    assert d._selection_gaze(20.0, d._safety_gen) is True
    assert calls and calls[0] >= t0 + 0.3
    # already looking within the min-change band: aimed, nothing sent
    monkeypatch.setattr(d, "gaze_at_bearing", lambda b, gen=None: False)
    monkeypatch.setattr(d, "_gaze_deg", 25.0)
    assert d._selection_gaze(20.0, d._safety_gen) is True
    assert d._selection_gaze(-30.0, d._safety_gen) is False
    # a stop since the request: refused without waiting
    monkeypatch.setattr(d, "_arm_settle_until", time.monotonic() + 30)
    t = time.monotonic()
    assert d._selection_gaze(20.0, d._safety_gen - 1) is False
    assert time.monotonic() - t < 0.5


def test_new_spoken_lines_have_no_stop_words():
    for s in (d.CANT_SEE_THAT_LINE, d.NOT_STANDING_LINE, d.VISION_DOWN_LINE):
        assert s and not BAD.search(s)
    for s in cm._DESCRIBE_ACKS + (cm._FIND_ACK,):
        assert not BAD.search(s)


# ----------------------------------------------------------------- client_mic
def test_vision_answer_action_matching():
    assert cm._vision_answer_action([{"intent": "sit"}]) is None
    a = {"intent": "find", "params": {"what": "backpack"}}
    assert cm._vision_answer_action([{"intent": "sit"}, a]) is a
    for n in ("describe", "look_around"):
        assert cm._vision_answer_action([{"intent": n}])["intent"] == n
    assert {"find", "look_around", "describe"} <= cm._MOTION_FREE_INTENTS


def test_vision_ack_never_repeats_gemma():
    assert cm._vision_ack({"intent": "find", "params": {"what": "backpack"}}) == \
        cm._FIND_ACK.format(what="backpack")
    assert cm._vision_ack({"intent": "look_around", "params": {}}) in cm._DESCRIBE_ACKS
    assert cm._vision_ack({"intent": "describe", "params": {}}) in cm._DESCRIBE_ACKS


class FakeTTS:
    def __init__(self):
        self.said = []

    def is_available(self):
        return False  # no streaming chunker: the fallback speak path

    def speak(self, text, voice=None):
        self.said.append(text)

    def wait(self):
        pass


def _turn(monkeypatch, actions, response="", transcript="do you see a backpack", tts=None):
    events = []

    class Brain:
        on_token_callback = None

        def process(self, text, state):
            return {"response": response, "actions": actions}

    monkeypatch.setattr(cm, "_brain_ok", True)
    monkeypatch.setattr(cm, "get_spot_state", lambda: {})
    monkeypatch.setattr(cm, "_resolve_voice", lambda b: None)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: transcript)
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)
    monkeypatch.setattr(cm, "_wait_for_nav_complete", lambda *a, **k: events.append("wait"))
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda intent, brain=None, speak=None: events.append(intent["intent"]) or True)
    cm._abort_chain.clear()
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], brain=Brain(),
                         info={}, has_wake_detector=True, tts=tts)
    return events


@pytest.fixture
def robot(monkeypatch):
    r = object()
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: {"robot": r})
    return r


def test_find_reply_not_spoken(monkeypatch, robot):
    calls = []
    monkeypatch.setattr(FI, "find", lambda rb, what: calls.append((rb, what)) or
                        ActionResult("Yes, there's a backpack behind me.", True))
    tts = FakeTTS()
    events = _turn(monkeypatch, [{"intent": "find", "params": {"what": "backpack"}}],
                   response="Yes! I can see a lovely backpack.", tts=tts)
    assert calls == [(robot, "backpack")] and events == []
    assert tts.said == [cm._FIND_ACK.format(what="backpack"), "Yes, there's a backpack behind me."]


def test_describe_side_and_look_around_call_vision(monkeypatch, robot):
    calls = []
    monkeypatch.setattr(DE, "describe", lambda rb, side="front", question="":
                        calls.append((rb, side, question)) or ActionResult(f"{side} view", True))
    tts = FakeTTS()
    _turn(monkeypatch, [{"intent": "describe", "params": {"side": "back"}}],
          response="I see a wall.", transcript="what's behind you", tts=tts)
    _turn(monkeypatch, [{"intent": "look_around", "params": {}}], transcript="look around", tts=tts)
    _turn(monkeypatch, [{"intent": "describe", "params": {"camera": "left"}}],
          transcript="what's on your left", tts=tts)
    assert calls == [(robot, "back", "what's behind you"), (robot, "around", ""),
                     (robot, "left", "what's on your left")]
    assert "I see a wall." not in tts.said
    assert "back view" in tts.said and "around view" in tts.said


def test_vision_answer_failure_speaks_honest_fallback(monkeypatch, robot):
    def boom(rb, what):
        raise RuntimeError("no camera")
    monkeypatch.setattr(FI, "find", boom)
    tts = FakeTTS()
    _turn(monkeypatch, [{"intent": "find", "params": {"what": "cup"}}], tts=tts)
    assert tts.said[-1] == "Sorry, my vision system isn't working right now."


def test_token_watch_suppresses_find_reply():
    class Chunker:
        suppressed = False

        def __init__(self):
            self.got = []

        def accept(self, delta):
            self.got.append(delta)

    for name in ("find", "look_around"):
        c = Chunker()
        w = cm._TokenWatch(c)
        w('{"actions": [{"action": "%s", "params": {}}], ' % name)
        w('"response": "Yes I see it')
        assert c.suppressed is True


def test_chain_breaks_on_failed_goto(monkeypatch):
    monkeypatch.setattr(cm, "_last_nav_ok", lambda: False)
    events = _turn(monkeypatch, [{"intent": "go_to_object", "params": {"description": "chair"}},
                                 {"intent": "sit", "params": {}}], transcript="go to the chair and sit")
    assert events == ["go_to_object", "wait"]


@pytest.mark.parametrize("outcome", [True, None])
def test_chain_continues_after_good_or_unknown_nav(monkeypatch, outcome):
    monkeypatch.setattr(cm, "_last_nav_ok", lambda: outcome)
    events = _turn(monkeypatch, [{"intent": "follow_me", "params": {}},
                                 {"intent": "sit", "params": {}}], transcript="follow me then sit")
    assert events == ["follow_me", "wait", "sit"]


def test_last_nav_ok_wrapper_dry_run(monkeypatch):
    monkeypatch.setenv("SPOT_DRY_RUN", "1")
    assert cm._last_nav_ok() is None
    monkeypatch.delenv("SPOT_DRY_RUN")
    monkeypatch.setattr(d, "_last_nav_ok", False)
    assert cm._last_nav_ok() is False


# ----------------------------------------------------------------- gaze worker
@pytest.fixture
def gaze_env(monkeypatch):
    sent = []
    monkeypatch.delenv("SPOT_DRY_RUN", raising=False)
    monkeypatch.setattr(d, "gaze_at_bearing", lambda b, gen=None: sent.append((b, gen)) or True)
    monkeypatch.setattr(d, "_spot_session", {"robot": "R"})
    monkeypatch.setattr(d, "_arm_pose", "talk")
    monkeypatch.setattr(d, "is_navigating", lambda: False)
    monkeypatch.setattr(cm, "_vision_warm", threading.Event())
    cm._vision_warm.set()
    return sent


def test_gaze_loop_uses_refined_bearing(monkeypatch, gaze_env):
    seen = []
    monkeypatch.setattr(SP, "refine_speaker_bearing", lambda robot, b: seen.append((robot, b)) or 12.0)
    cm._gaze_once(30.0, 7)
    assert seen == [("R", 30.0)] and gaze_env == [(12.0, 7)]


def test_gaze_loop_refine_raising_uses_raw(monkeypatch, gaze_env):
    def boom(robot, b):
        raise RuntimeError("cuda busy")
    monkeypatch.setattr(SP, "refine_speaker_bearing", boom)
    cm._gaze_once(30.0, 7)
    assert gaze_env == [(30.0, 7)]


@pytest.mark.parametrize("case", ["no_session", "dry_run", "centre", "cold", "walking"])
def test_gaze_loop_skips_refinement(monkeypatch, gaze_env, case):
    monkeypatch.setattr(SP, "refine_speaker_bearing", lambda *a: pytest.fail("no refinement"))
    body = 30.0
    if case == "no_session":
        monkeypatch.setattr(d, "_spot_session", None)
        monkeypatch.setattr(d, "ensure_spot_session", lambda *a: pytest.fail("no bring-up"))
    elif case == "dry_run":
        monkeypatch.setenv("SPOT_DRY_RUN", "1")
    elif case == "centre":
        body = None
    elif case == "cold":
        cm._vision_warm.clear()
    elif case == "walking":
        monkeypatch.setattr(d, "is_navigating", lambda: True)
    cm._gaze_once(body, 7)
    assert gaze_env == [(body, 7)]


# ----------------------------------------------------------------- startup
def test_setup_vision_registers_getter_and_warms(monkeypatch):
    loaded = []
    started = []

    class T:
        def __init__(self, target=None, name=None, daemon=None):
            self.target, self.daemon = target, daemon

        def start(self):
            started.append(self.daemon)
            self.target()

    def load(name):
        loaded.append(name)
        if name == "yolo26l-pose":
            raise RuntimeError("engine build failed")
        return type("M", (), {"provider": "trt"})()
    monkeypatch.setattr(RT, "load_model", load)
    monkeypatch.setattr(cm, "_vision_warm", threading.Event())
    monkeypatch.setattr(cm.threading, "Thread", T)
    monkeypatch.setitem(cm._status, "locked_deg", 42.0)
    monkeypatch.setattr(SP, "_getter", None)
    cm._setup_vision(dry_run=False)
    assert SP.speaker_bearing() == 42.0
    assert loaded == ["yolo26l-seg", "yolo26l-pose", "yoloe-26l-seg"]  # errors don't end warm-up
    assert started == [True] and cm._vision_warm.is_set()


def test_setup_vision_dry_run_skips_warmup(monkeypatch):
    monkeypatch.setattr(RT, "load_model", lambda name: pytest.fail("no warm-up in dry-run"))
    monkeypatch.setattr(SP, "_getter", None)
    monkeypatch.setitem(cm._status, "locked_deg", None)
    cm._setup_vision(dry_run=True)
    assert SP._getter is not None and SP.speaker_bearing() is None


def test_package_reexports():
    import src.voice_control.vision as V
    for name in ("describe", "find", "refine_speaker_bearing", "set_speaker_bearing_getter",
                 "goto", "follow", "select_person"):
        assert callable(getattr(V, name)), name
