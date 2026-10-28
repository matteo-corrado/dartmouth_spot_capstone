"""client_mic turn handling (Task 12 items + safety re-check after strip)."""
import src.voice_control.client_mic as cm


def _first_word_safety(text):
    # Stand-in rule so these tests exercise client_mic's ordering, not the
    # intent.py rule: fires only when the FIRST word is stop/freeze.
    w = text.strip().lower().split()
    if w and w[0].strip(",.!") in ("stop", "freeze"):
        return {"intent": w[0].strip(",.!"), "params": {}, "raw": text}
    return None


def _run(monkeypatch, transcript, **kw):
    sent = []
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: transcript)
    monkeypatch.setattr(cm, "check_safety_command", _first_word_safety)
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda intent, brain=None, speak=None: sent.append(intent["intent"]) or True)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    import numpy as np
    audio = [np.zeros(16000, np.float32)]
    info = {}
    r = cm.process_utterance(None, None, audio, brain=None, info=info, **kw)
    return r, sent, info


def test_safety_rechecked_after_wake_strip(monkeypatch):
    # Raw text starts with "hey" (no hit); after the strip "stop ..." must
    # halt and the trailing "come back" must never dispatch.
    r, sent, info = _run(monkeypatch, "hey spot stop now and come back", has_wake_detector=True)
    assert sent == ["stop"]
    assert not info.get("reached_llm")


def test_safety_rechecked_after_asr_fallback_wake(monkeypatch):
    r, sent, _ = _run(monkeypatch, "hey spot freeze then come back",
                      safety_only=True, has_wake_detector=False)
    assert sent == ["freeze"]


def test_raw_safety_first(monkeypatch):
    r, sent, _ = _run(monkeypatch, "stop come back", has_wake_detector=True)
    assert sent == ["stop"]


def test_filler_only_skipped(monkeypatch):
    monkeypatch.setattr(cm, "is_filler_only", lambda t: True)
    r, sent, info = _run(monkeypatch, "um uh hmm", has_wake_detector=True)
    assert sent == [] and not info.get("reached_llm")


class _Timer:
    def __init__(self, delay, fn):
        self.delay, self.fn, self.cancelled, self.daemon = delay, fn, False, False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


def test_thinking_filler_speaks_once_when_silent():
    said = []
    f = cm.ThinkingFiller(lambda: said.append(1), 4.0, lambda: False, timer_factory=_Timer)
    f._timer.fn()
    f._timer.fn()
    assert said == [1] and f._timer.delay == 4.0


def test_thinking_filler_cancelled_or_busy_is_silent():
    said = []
    f = cm.ThinkingFiller(lambda: said.append(1), 4.0, lambda: False, timer_factory=_Timer)
    f.cancel()
    f._timer.fn()
    g = cm.ThinkingFiller(lambda: said.append(1), 4.0, lambda: True, timer_factory=_Timer)
    g._timer.fn()
    assert said == [] and f._timer.cancelled


class _Chunker:
    def __init__(self):
        self.suppressed = False
        self.got = []

    def accept(self, d):
        self.got.append(d)


class _Filler:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True


def test_token_watch_suppresses_persona_reply_and_cancels_filler():
    c, f = _Chunker(), _Filler()
    w = cm._TokenWatch(c, f)
    for d in ['{"actions": [{"action": "set_', 'persona", "params": {"name": "pirate"}}], ',
              '"response": "', 'Arr matey."}']:
        w(d)
    assert c.suppressed is True and f.cancelled is True
    assert "".join(c.got).endswith('Arr matey."}')


def test_token_watch_leaves_normal_reply_streaming():
    c, f = _Chunker(), _Filler()
    w = cm._TokenWatch(c, f)
    w('{"actions": [{"action": "sit"}], "response": "Sitting."}')
    assert c.suppressed is False and f.cancelled is True


def test_brain_ready_rechecks_until_up(monkeypatch):
    class B:
        n = 0

        def is_available(self):
            self.n += 1
            return self.n >= 2

    b = B()
    monkeypatch.setattr(cm, "_brain_ok", False)
    assert cm._brain_ready(b) is False
    assert cm._brain_ready(b) is True
    assert cm._brain_ready(b) is True and b.n == 2  # positive cached


def test_real_rule_hey_spot_stop_come_back(monkeypatch):
    # Production rule from intent.py: "Hey Spot, stop. Come back." must stop
    # and never dispatch come_back.
    from src.voice_control.intent import check_safety_command
    sent = []
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: "Hey Spot, stop. Come back.")
    monkeypatch.setattr(cm, "check_safety_command", check_safety_command)
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda intent, brain=None, speak=None: sent.append(intent["intent"]) or True)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    import numpy as np
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], brain=None,
                         has_wake_detector=True)
    assert sent == ["stop"]


def test_fuzzy_wake_strip_reaches_regex(monkeypatch):
    seen = []
    monkeypatch.setattr(cm, "WAKE_PHRASES", ["HELLO SPOTTY"])
    monkeypatch.setattr(cm, "parse_intent", lambda t: seen.append(t) or None)
    monkeypatch.setattr(cm, "is_filler_only", lambda t: False)
    r, sent, info = _run(monkeypatch, "hello spotted sit down", has_wake_detector=True)
    assert seen == ["sit down"] and info["reached_llm"]


def test_asr_fallback_bare_wake_phrase(monkeypatch):
    r, sent, _ = _run(monkeypatch, "hey spot", safety_only=True, has_wake_detector=False)
    assert r == "wake_detected" and sent == []


def test_no_strip_mid_sentence(monkeypatch):
    seen = []
    monkeypatch.setattr(cm, "parse_intent", lambda t: seen.append(t) or None)
    monkeypatch.setattr(cm, "is_filler_only", lambda t: False)
    _run(monkeypatch, "is the spot free", has_wake_detector=True)
    assert seen == ["is the spot free"]


def test_wake_detector_gets_peak_target(monkeypatch):
    import src.voice_control.wake.sherpa_onnx as so
    got = {}

    class D:
        def __init__(self, peak_target=None):
            got["pt"] = peak_target

    monkeypatch.setattr(so, "WakeWordDetector", D)
    monkeypatch.delenv("SPOT_WAKE_BACKEND", raising=False)
    cm._make_wake_detector()
    assert got["pt"] == cm._tuning.kws_peak_target == 0.9


def test_chain_waits_for_dance_before_next_action(monkeypatch):
    # "dance then sit": the emote runs on the nav thread, so the chain must
    # wait for it (like go_to) before dispatching the next action.
    import numpy as np
    events = []

    class Brain:
        on_token_callback = None

        def process(self, text, state):
            return {"response": "", "actions": [
                {"intent": "emote", "params": {"name": "dance"}},
                {"intent": "sit", "params": {}}]}

    monkeypatch.setattr(cm, "_brain_ok", True)
    monkeypatch.setattr(cm, "get_spot_state", lambda: {})
    monkeypatch.setattr(cm, "_resolve_voice", lambda b: None)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: "dance then sit")
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)
    monkeypatch.setattr(cm, "_wait_for_nav_complete", lambda *a, **k: events.append("wait"))
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda intent, brain=None, speak=None:
                        events.append(intent["intent"]) or True)
    cm._abort_chain.clear()
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], brain=Brain(),
                         info={}, has_wake_detector=True)
    assert events == ["emote", "wait", "sit"]
