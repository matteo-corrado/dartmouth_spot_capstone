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
    monkeypatch.setattr(d, "_vision_warm", threading.Event())
    d.mark_vision_warm()
    monkeypatch.setattr(d, "_selecting", False)
    # the fresh look after person selection (I1) sees the selected person
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [_det(label="person", dist=2.5)])
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
    assert said == ["I'll go to the chair in front of me.", "I'm here."]  # several: say which
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
    fresh = _det(label="person", dist=2.8, bearing=4.0)
    looks = []

    def snap(robot, sources=None, model="yolo26l-seg", labels=None, with_depth=True):
        looks.append((labels, with_depth))
        return [fresh]
    monkeypatch.setattr(W, "snapshot", snap)
    target = _det(label="person", dist=3.0)
    got = {}

    def select(robot, phrase, speak, gaze, **kw):
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
    assert t is fresh and label == "person" and is_person is True  # the fresh sighting
    assert looks == [({"person"}, True)]
    assert said == ["Can you raise your hand?", "I'm here."]


def test_select_person_message_spoken_and_no_motion(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    monkeypatch.setattr(P, "select_person", lambda robot, phrase, speak, gaze, **kw: (None, P.MSG_CANT_SEE_YOU))
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
                        lambda robot, phrase, speak, gaze, **kw: phrases.append(phrase) or (target, None))
    said = []
    gen = d._safety_gen
    assert d.dispatch_intent({"intent": "follow_me", "params": {}}, speak=said.append) is True
    assert d._nav_mode == "follow"
    _join_nav()
    (_, io, t, _ev), = rec.calls
    assert phrases == ["me"] and t.label == "person" and io.gen == gen
    assert io.stop_event is _ev  # M1: navio refuses sends once the nav thread is cancelled
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

    def select(robot, phrase, speak, gaze, **kw):
        d.dispatch_intent({"intent": "stop", "params": {}})  # a stop lands mid-selection
        return _det(label="person"), None
    monkeypatch.setattr(P, "select_person", select)
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "me"}}) is False
    _join_nav()
    assert rec.calls == [] and not d.is_navigating()


def test_selection_gaze_recentred_before_walking(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)

    def select(robot, phrase, speak, gaze, **kw):
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
    monkeypatch.setattr(d, "gaze_at_bearing",
                        lambda b, gen=None, selection=False: calls.append((time.monotonic(), selection)) or True)
    t0 = time.monotonic()
    monkeypatch.setattr(d, "_arm_settle_until", t0 + 0.3)
    monkeypatch.setattr(d, "_gaze_t", 0.0)
    assert d._selection_gaze(20.0, d._safety_gen) is True
    assert calls and calls[0][0] >= t0 + 0.3 and calls[0][1] is True
    # the aim was refused: aimed only if already within 4 deg
    monkeypatch.setattr(d, "gaze_at_bearing", lambda b, gen=None, selection=False: False)
    monkeypatch.setattr(d, "_gaze_deg", 23.0)
    assert d._selection_gaze(20.0, d._safety_gen) is True
    assert d._selection_gaze(12.0, d._safety_gen) is False   # 11 deg off: not the same person
    # a stop since the request: refused without waiting
    monkeypatch.setattr(d, "_arm_settle_until", time.monotonic() + 30)
    t = time.monotonic()
    assert d._selection_gaze(20.0, d._safety_gen - 1) is False
    assert time.monotonic() - t < 0.5


def test_selection_gaze_refuses_past_the_clamp(fake, monkeypatch):
    monkeypatch.setattr(d, "gaze_at_bearing", lambda *a, **k: pytest.fail("no aim past the clamp"))
    from src.voice_control.tuning import get_tuning
    limit = min(float(get_tuning().gaze_max_deg), d.GAZE_HARD_MAX_DEG)
    monkeypatch.setattr(d, "_gaze_deg", limit)  # arm sits at the clamp
    assert d._selection_gaze(limit + 5.0, d._safety_gen) is False
    assert d._selection_gaze(-(limit + 20.0), d._safety_gen) is False
    assert d._selection_gaze(400.0 + limit, d._safety_gen) is False  # wrapped: still past it


def _gaze_ready(monkeypatch):
    monkeypatch.setattr(d, "_arm_settle_until", 0.0)
    monkeypatch.setattr(d, "_gaze_t", 0.0)
    monkeypatch.setattr(d, "_standing_and_powered", lambda s: True)


def test_gaze_requests_blocked_while_selecting(fake, monkeypatch):
    _gaze_ready(monkeypatch)
    monkeypatch.setattr(d, "_selecting", True)
    assert d.gaze_at_bearing(10.0, d._safety_gen) is False          # e.g. the client_mic gaze worker
    assert "arm_joints" not in fake["cmd"].kinds()
    assert d.gaze_at_bearing(10.0, d._safety_gen, selection=True) is True  # selection's own aim
    assert fake["cmd"].kinds() == ["arm_joints"]
    # selection re-aims at 4+ deg (not the 15 deg band of speaker gaze)
    monkeypatch.setattr(d, "_gaze_t", 0.0)
    assert d.gaze_at_bearing(16.0, d._safety_gen, selection=True) is True


def test_selecting_flag_held_during_selection_only(fake, monkeypatch):
    seen = []

    def select(robot, phrase, speak, gaze, **kw):
        seen.append(d._selecting)
        return None, P.MSG_NOT_SURE
    monkeypatch.setattr(P, "select_person", select)
    d.dispatch_intent({"intent": "go_to_object", "params": {"description": "me"}})
    assert seen == [True] and d._selecting is False

    def boom(*a, **k):
        raise RuntimeError("cuda")
    monkeypatch.setattr(P, "select_person", boom)
    d.dispatch_intent({"intent": "go_to_object", "params": {"description": "me"}})
    assert d._selecting is False


def test_stop_during_selection_ends_waits_and_speech(fake, monkeypatch):
    said, timing = [], {}

    def select(robot, phrase, speak, gaze, clock, sleep, abort):
        assert abort() is False
        d.dispatch_intent({"intent": "stop", "params": {}})
        timing["abort"] = abort()
        t = time.monotonic()
        sleep(5.0)                       # e.g. the raised-hand wait / gaze settle
        timing["slept"] = time.monotonic() - t
        timing["clock"] = clock()
        speak(P.ASK_HAND)
        return None, P.MSG_FEW_PEOPLE
    monkeypatch.setattr(P, "select_person", select)
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "me"}},
                             speak=said.append) is False
    assert timing["slept"] < 0.5 and timing["clock"] == float("inf") and timing["abort"] is True
    assert said == [] and not d.is_navigating()


def test_go_to_and_follow_refuse_until_warm(fake, monkeypatch):
    monkeypatch.setattr(d, "_vision_warm", threading.Event())
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: pytest.fail("must not look"))
    monkeypatch.setattr(P, "select_person", lambda *a, **k: pytest.fail("must not look"))
    for intent in ({"intent": "go_to_object", "params": {"description": "chair"}},
                   {"intent": "follow_me", "params": {}}):
        said = []
        assert d.dispatch_intent(intent, speak=said.append) is False
        assert said == [d.WARMING_LINE] and not d.is_navigating()
    assert "mobility" not in fake["cmd"].kinds()


def test_follow_paused_text_compared_with_follow_texts(fake, monkeypatch):
    monkeypatch.setattr(FO, "TEXTS", dict(FO.TEXTS, paused="Holding still now."))
    monkeypatch.setattr(FO, "follow", Rec(ActionResult("Holding still now.", False)).follow)
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (_det(label="person"), None))
    said = []
    d.dispatch_intent({"intent": "follow_me", "params": {}}, speak=said.append)
    _join_nav()
    assert said == []


def test_spoken_vision_results_are_scrubbed(fake, monkeypatch):
    monkeypatch.setattr(G, "goto", Rec(ActionResult("I reached the stop sign. Halting.", True)).goto)
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [_det()])
    said = []
    d.dispatch_intent({"intent": "go_to_object", "params": {"description": "chair"}}, speak=said.append)
    _join_nav()
    assert said == ["I reached the road sign."]


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
    monkeypatch.setattr(d, "_spot_session", {"robot": r})
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("answers never stand the robot"))
    _warm_answers(monkeypatch)
    return r


def _warm_answers(monkeypatch):
    ev = threading.Event()
    ev.set()
    monkeypatch.setattr(cm, "_vision_warm", ev)


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
    # conversation-only and motion replies still stream
    for head in ('{"actions": [], ', '{"actions": [{"action": "go_to_object", "params": {}}], '):
        c = Chunker()
        w = cm._TokenWatch(c)
        w(head)
        w('"response": "Sure thing')
        assert c.suppressed is False and any("Sure" in g for g in c.got)


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
        assert not cm._vision_warm.is_set() and not d._vision_warm.is_set()  # not warm mid-build
        if name == "yolo26l-pose":
            raise RuntimeError("engine build failed")
        return type("M", (), {"provider": "trt",
                              "detect": lambda self, img: detected.append((name, img.shape))})()
    detected = []
    monkeypatch.setattr(RT, "load_model", load)
    monkeypatch.setattr(cm, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_ok", None)
    monkeypatch.setattr(cm.threading, "Thread", T)
    monkeypatch.setitem(cm._status, "locked_deg", 42.0)
    monkeypatch.setattr(SP, "_getter", None)
    cm._setup_vision(dry_run=False)
    assert SP.speaker_bearing() == 42.0
    assert loaded == ["yolo26l-seg", "yolo26l-pose", "yoloe-26l-seg"]  # errors don't end warm-up
    assert started == [True] and cm._vision_warm.is_set() and d._vision_warm.is_set()
    # one dummy detect per loaded model on a blank 640x480 image (I5c)
    assert detected == [("yolo26l-seg", (480, 640, 3)), ("yoloe-26l-seg", (480, 640, 3))]
    assert d._vision_ok is True


def test_setup_vision_dry_run_skips_warmup(monkeypatch):
    monkeypatch.setattr(RT, "load_model", lambda name: pytest.fail("no warm-up in dry-run"))
    monkeypatch.setattr(cm, "_vision_warm", threading.Event())
    monkeypatch.setattr(SP, "_getter", None)
    monkeypatch.setitem(cm._status, "locked_deg", None)
    cm._setup_vision(dry_run=True)
    assert SP._getter is not None and SP.speaker_bearing() is None
    assert cm._vision_warm.is_set()  # camera answers are not gated forever in dry-run


def test_package_reexports():
    import src.voice_control.vision as V
    for name in ("describe", "find", "refine_speaker_bearing", "set_speaker_bearing_getter",
                 "goto", "follow", "select_person"):
        assert callable(getattr(V, name)), name


# ----------------------------------------------------------------- fix round 1
SPW = importlib.import_module("src.voice_control.vision.speech")


@pytest.mark.parametrize("text,out", [
    ("stop sign", "road sign"), ("two Stop-Signs here", "two road signs here"),
    ("Let me look for the stop...", "Let me look for the..."),
    ("I halted near the e-stop, then froze.", "I near the, then."),
    ("a backpack", "a backpack"), ("", "")])
def test_safe_words(text, out):
    assert SPW.safe_words(text) == out
    assert not BAD.search(SPW.safe_words(text))


def test_stop_sign_question_never_says_stop(monkeypatch, robot):
    """"Do you see a stop sign?": ack, detector answer and VLM fallback stay clean."""
    sign = _det(label="stop sign", dist=3.0, bearing=170.0, source="back_fisheye_image")
    monkeypatch.setattr(FI, "_detect", lambda *a, **k: [sign])
    tts = FakeTTS()
    _turn(monkeypatch, [{"intent": "find", "params": {"what": "stop sign"}}],
          response="Yes, a stop sign!", transcript="do you see a stop sign", tts=tts)
    assert len(tts.said) == 2 and "road sign" in tts.said[1]
    assert not any(BAD.search(t) for t in tts.said), tts.said
    # not detected -> VLM "yes" -> MAYBE line with the phrase
    monkeypatch.setattr(FI, "_detect", lambda *a, **k: [])
    monkeypatch.setattr(FI, "capture", lambda *a, **k: {})
    monkeypatch.setattr(FI, "front_images", lambda f: [b"jpg"])
    monkeypatch.setattr(FI, "ask_vlm", lambda *a, **k: "Yes.")
    res = FI.find(robot, "big stop sign")
    assert res.ok and "road sign" in res.text and not BAD.search(res.text)
    # describe's also-line names YOLO labels
    line = DE.also_line([sign])
    assert "road sign" in line and not BAD.search(line)


def test_vision_answer_text_scrubbed(monkeypatch, robot):
    monkeypatch.setattr(DE, "describe", lambda rb, side="front", question="":
                        ActionResult("There's a stop sign, so I'd halt here.", True))
    tts = FakeTTS()
    _turn(monkeypatch, [{"intent": "describe", "params": {}}], transcript="what do you see", tts=tts)
    assert not any(BAD.search(t) for t in tts.said), tts.said


@pytest.fixture
def no_session(monkeypatch):
    monkeypatch.setattr(d, "_spot_session", None)
    monkeypatch.setattr(d, "ensure_spot_session", lambda *a, **k: pytest.fail("answers never stand the robot"))
    monkeypatch.setattr(cm, "_eyes_robot", None)
    monkeypatch.setattr(cm, "_eyes_fail_until", 0.0)
    _warm_answers(monkeypatch)
    sess_mod = importlib.import_module("src.session")
    made = []

    def quick(name, **kw):
        made.append((name, kw))
        return "EYES"
    monkeypatch.setattr(sess_mod, "quick_robot", quick)
    return made


def test_vision_answers_use_read_only_robot_without_session(monkeypatch, no_session):
    got = []
    monkeypatch.setattr(FI, "find", lambda rb, what: got.append(rb) or ActionResult("No.", False))
    monkeypatch.setattr(DE, "describe", lambda rb, side="front", question="":
                        got.append(rb) or ActionResult("A wall.", True))
    _turn(monkeypatch, [{"intent": "find", "params": {"what": "cup"}}], tts=FakeTTS())
    _turn(monkeypatch, [{"intent": "describe", "params": {}}], tts=FakeTTS())
    _turn(monkeypatch, [{"intent": "look_around", "params": {}}], tts=FakeTTS())
    assert got == ["EYES"] * 3
    assert no_session == [("spot-voice-eyes", {})]  # one cached connection, default robot IP


def test_vision_answer_without_any_robot_says_so(monkeypatch, no_session):
    sess_mod = importlib.import_module("src.session")

    def down(name, **kw):
        raise ConnectionError("no route")
    monkeypatch.setattr(sess_mod, "quick_robot", down)
    monkeypatch.setattr(FI, "find", lambda *a: pytest.fail("no robot, no look"))
    tts = FakeTTS()
    _turn(monkeypatch, [{"intent": "find", "params": {"what": "cup"}}], tts=tts)
    assert tts.said[-1] == cm.EYES_OFF_LINE
    assert not BAD.search(cm.EYES_OFF_LINE) and not BAD.search(d.WARMING_LINE)


# ----------------------------------------------------------------- final fix: C1, I1, M4, I5a
PERSON_WORDS = ["persons", "the kids", "kids", "the children", "people", "some people",
                "the guys", "the ladies", "the boys", "girls", "someone", "a man", "the woman",
                "the child", "humans", "folks"]


@pytest.mark.parametrize("desc", PERSON_WORDS)
def test_person_words_take_the_person_path(fake, monkeypatch, desc):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (_det(label="person", dist=3.0), None))
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": desc}})
    _join_nav()
    (_, _io, _t, label, is_person, _ev), = rec.calls
    assert label == "person" and is_person is True


def test_no_person_label_reaches_the_object_path(fake, monkeypatch):
    """Every description that resolves to the "person" label goes to select_person."""
    voc = importlib.import_module("src.voice_control.vision.vocab")
    phrases = [k for k, v in voc.SYNONYMS.items() if v == "person"] + ["person"]
    phrases += [w + "s" for w in phrases] + [f"the {w}" for w in phrases] + PERSON_WORDS
    reached = []
    real = d._vision_object_target
    monkeypatch.setattr(d, "_vision_object_target",
                        lambda s, desc, sp: reached.append(desc) or real(s, desc, sp))
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (None, P.MSG_NOT_SURE))
    for desc in phrases:
        hit = d._vision_object_label(desc)
        if hit and hit[1] == "person":
            d.dispatch_intent({"intent": "go_to_object", "params": {"description": desc}})
            assert desc not in reached, desc
    # the object path itself refuses a person label (belt and braces)
    said = []
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: pytest.fail("no object look for a person"))
    assert real(fake, "persons", said.append) is None and said == [d.CANT_SEE_THAT_LINE]


def test_fresh_look_none_says_cant_see_and_does_not_move(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    monkeypatch.setattr(FO, "follow", rec.follow)
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (_det(label="person", bearing=0.0), None))
    # a bystander 30 deg away and a chair do not count as the selected person
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [_det(label="person", bearing=30.0),
                                                         _det(label="chair", bearing=0.0)])
    said = []
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "person in red"}},
                             speak=said.append) is False
    assert said == [d.CANT_SEE_THAT_LINE]
    said.clear()
    assert d.dispatch_intent({"intent": "follow_me", "params": {}}, speak=said.append) is False
    assert said == [P.MSG_CANT_SEE_YOU]
    _join_nav()
    assert rec.calls == [] and not d.is_navigating() and "mobility" not in fake["cmd"].kinds()


def test_fresh_look_failure_says_vision_down(fake, monkeypatch):
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (_det(label="person"), None))

    def boom(*a, **k):
        raise RuntimeError("camera rpc")
    monkeypatch.setattr(W, "snapshot", boom)
    said = []
    assert d.dispatch_intent({"intent": "follow_me", "params": {}}, speak=said.append) is False
    assert said == [d.VISION_DOWN_LINE] and not d.is_navigating()


def test_fresh_look_picks_the_selected_person_on_covering_cameras(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(FO, "follow", rec.follow)
    sel = _det(label="person", bearing=-90.0, dist=3.0, source="right_fisheye_image")
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (sel, None))
    near_b = _det(label="person", bearing=-88.0, dist=3.1, source="right_fisheye_image")
    other = _det(label="person", bearing=-100.0, dist=1.5, source="right_fisheye_image")
    seen = {}

    def snap(robot, sources=None, model="yolo26l-seg", labels=None, with_depth=True):
        seen.update(sources=list(sources), labels=labels, with_depth=with_depth)
        return [other, near_b]
    monkeypatch.setattr(W, "snapshot", snap)
    assert d.dispatch_intent({"intent": "follow_me", "params": {}})
    _join_nav()
    (_, _io, t, _ev), = rec.calls
    assert t is near_b
    assert seen == {"sources": W.cameras_covering(-90.0), "labels": {"person"}, "with_depth": True}


def test_fresh_look_after_stop_is_silent(fake, monkeypatch):
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (_det(label="person"), None))

    def snap(*a, **k):
        d.dispatch_intent({"intent": "stop", "params": {}})
        return [_det(label="person")]
    monkeypatch.setattr(W, "snapshot", snap)
    said = []
    assert d.dispatch_intent({"intent": "follow_me", "params": {}}, speak=said.append) is False
    assert said == [] and not d.is_navigating()


def test_single_object_match_says_nothing_extra(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(G, "goto", rec.goto)
    # one chair seen by both front cameras is one chair: no "which" line
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [
        _det(bearing=40.0, dist=2.0), _det(bearing=41.0, dist=2.1, source="frontright_fisheye_image")])
    said = []
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "chair"}},
                             speak=said.append)
    _join_nav()
    assert said == ["I'm here."]


def test_several_objects_say_which_before_walking(fake, monkeypatch):
    order = []
    monkeypatch.setattr(G, "goto", lambda *a, **k: order.append("walk") or ActionResult("", True))
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [
        _det(bearing=90.0, dist=2.0), _det(bearing=-90.0, dist=4.0, source="right_fisheye_image")])
    assert d.dispatch_intent({"intent": "go_to_object", "params": {"description": "chair"}},
                             speak=lambda t: order.append(t))
    _join_nav()
    assert order == ["I'll go to the chair on my left.", "walk"]


def test_vision_unavailable_says_eyes_down(fake, monkeypatch):
    monkeypatch.setattr(d, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_ok", True)
    d.mark_vision_warm(ok=False)
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: pytest.fail("must not look"))
    monkeypatch.setattr(P, "select_person", lambda *a, **k: pytest.fail("must not look"))
    for intent in ({"intent": "go_to_object", "params": {"description": "chair"}},
                   {"intent": "follow_me", "params": {}}):
        said = []
        assert d.dispatch_intent(intent, speak=said.append) is False
        assert said == [d.EYES_DOWN_LINE] and not d.is_navigating()
    for s in (d.EYES_DOWN_LINE, d.SAY_WHICH_LINE):
        assert not BAD.search(s)


# ----------------------------------------------------------------- final fix: I5, M2, D2
def test_warmup_runtime_import_failure_still_marks_warm(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "src.voice_control.vision.runtime", None)  # import fails
    monkeypatch.setattr(cm, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_ok", True)
    cm._warm_vision_models()
    assert cm._vision_warm.is_set() and d._vision_warm.is_set()
    assert d._vision_ok is False  # go-to/follow say the eyes are down, not "warming up"


def test_warmup_main_detector_failure_marks_unavailable(monkeypatch):
    def load(name):
        if name == "yolo26l-seg":
            raise RuntimeError("engine build failed")
        return type("M", (), {"provider": "trt", "detect": lambda self, img: []})()
    monkeypatch.setattr(RT, "load_model", load)
    monkeypatch.setattr(cm, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_ok", True)
    cm._warm_vision_models()
    assert d._vision_warm.is_set() and d._vision_ok is False


def test_warmup_dummy_detect_failure_is_not_a_load_failure(monkeypatch):
    def bad_detect(self, img):
        raise RuntimeError("cuda oom")
    monkeypatch.setattr(RT, "load_model",
                        lambda name: type("M", (), {"provider": "trt", "detect": bad_detect})())
    monkeypatch.setattr(cm, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_warm", threading.Event())
    monkeypatch.setattr(d, "_vision_ok", None)
    cm._warm_vision_models()
    assert d._vision_ok is True


def test_vision_answers_wait_for_warmup(monkeypatch, robot):
    monkeypatch.setattr(cm, "_vision_warm", threading.Event())  # still warming
    monkeypatch.setattr(FI, "find", lambda *a: pytest.fail("no detector while warming"))
    monkeypatch.setattr(DE, "describe", lambda *a, **k: pytest.fail("no detector while warming"))
    for cmd, params in (("find", {"what": "cup"}), ("describe", {}), ("look_around", {})):
        tts = FakeTTS()
        _turn(monkeypatch, [{"intent": cmd, "params": params}], tts=tts)
        assert tts.said[-1] == cm.VISION_WARMING_LINE
    assert not BAD.search(cm.VISION_WARMING_LINE)


def _legacy_turn(monkeypatch, transcript, tts=None):
    events = []
    monkeypatch.setattr(cm, "_brain_ok", False)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: transcript)
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda intent, brain=None, speak=None: events.append(intent["intent"]) or True)
    cm._abort_chain.clear()
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], brain=None,
                         info={}, has_wake_detector=True, tts=tts)
    return events


@pytest.mark.parametrize("transcript,what", [
    ("find a chair", "chair"), ("where is the backpack", "backpack"),
    ("do you see a cup", "cup"), ("can you see the dog", "dog"), ("look for my keys", "keys")])
def test_regex_fallback_find_answers_and_never_walks(monkeypatch, robot, transcript, what):
    from src.voice_control.intent import parse_intent
    assert parse_intent(transcript) == {"intent": "find", "params": {"what": what},
                                        "raw": transcript}
    calls = []
    monkeypatch.setattr(FI, "find", lambda rb, w: calls.append(w) or
                        ActionResult(f"Yes, there's a {w} in front.", True))
    tts = FakeTTS()
    events = _legacy_turn(monkeypatch, transcript, tts=tts)
    assert calls == [what] and events == []          # answered, never dispatched (no walk)
    assert tts.said == [f"Yes, there's a {what} in front."]


def test_regex_fallback_describe_and_look_around(monkeypatch, robot):
    from src.voice_control.intent import parse_intent
    assert parse_intent("what do you see")["intent"] == "describe"
    assert parse_intent("look around")["intent"] == "look_around"
    calls = []
    monkeypatch.setattr(DE, "describe", lambda rb, side="front", question="":
                        calls.append(side) or ActionResult("A hallway.", True))
    tts = FakeTTS()
    assert _legacy_turn(monkeypatch, "what do you see", tts=tts) == []
    assert _legacy_turn(monkeypatch, "look around", tts=tts) == []
    assert calls == ["front", "around"] and tts.said == ["A hallway.", "A hallway."]


def test_eyes_connect_failure_cached_for_a_while(monkeypatch, no_session):
    sess_mod = importlib.import_module("src.session")
    tries = []

    def down(name, **kw):
        tries.append(name)
        raise ConnectionError("no route")
    monkeypatch.setattr(sess_mod, "quick_robot", down)
    now = {"t": 1000.0}
    monkeypatch.setattr(cm.time, "monotonic", lambda: now["t"])
    assert cm._vision_robot() is None and cm._vision_robot() is None
    assert tries == ["spot-voice-eyes"]          # no reconnect inside the window
    now["t"] += cm.EYES_RETRY_S + 1
    assert cm._vision_robot() is None and len(tries) == 2


def test_eyes_robot_dropped_after_connection_error(monkeypatch, no_session):
    def boom(rb, what):
        raise ConnectionError("robot rebooted")
    monkeypatch.setattr(FI, "find", boom)
    tts = FakeTTS()
    _turn(monkeypatch, [{"intent": "find", "params": {"what": "cup"}}], tts=tts)
    assert tts.said[-1] == cm.VISION_ERROR_LINE and cm._eyes_robot is None
    monkeypatch.setattr(FI, "find", lambda rb, what: ActionResult("No.", False))
    _turn(monkeypatch, [{"intent": "find", "params": {"what": "cup"}}], tts=FakeTTS())
    assert no_session == [("spot-voice-eyes", {}), ("spot-voice-eyes", {})]  # reconnected


def test_session_robot_never_dropped(monkeypatch, robot):
    def boom(rb, what):
        raise ConnectionError("blip")
    monkeypatch.setattr(FI, "find", boom)
    _turn(monkeypatch, [{"intent": "find", "params": {"what": "cup"}}], tts=FakeTTS())
    assert d._spot_session["robot"] is robot


def test_selecting_flag_held_over_the_fresh_look(fake, monkeypatch):
    seen = []
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (_det(label="person"), None))
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: seen.append(d._selecting) or [])
    d.dispatch_intent({"intent": "follow_me", "params": {}})
    assert seen == [True] and d._selecting is False


def test_fresh_look_prefers_depth_range_at_same_bearing(fake, monkeypatch):
    rec = Rec(ActionResult("I'm here.", True))
    monkeypatch.setattr(FO, "follow", rec.follow)
    monkeypatch.setattr(P, "select_person", lambda *a, **k: (_det(label="person"), None))
    ground = _det(label="person", dist=2.0, bearing=1.0)
    ground.distance_method = "ground"
    depth = _det(label="person", dist=2.2, bearing=1.0, source="frontright_fisheye_image")
    monkeypatch.setattr(W, "snapshot", lambda *a, **k: [ground, depth])
    assert d.dispatch_intent({"intent": "follow_me", "params": {}})
    _join_nav()
    assert rec.calls[0][2] is depth
