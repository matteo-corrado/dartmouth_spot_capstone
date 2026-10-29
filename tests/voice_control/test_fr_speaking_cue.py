"""Final-review I2: the RESPONDING signal (ring + speak cue) fires once, just
before the first reply line is queued, not after the action chain."""
import numpy as np

import src.voice_control.client_mic as cm
from src.voice_control.spot_tts import TTSChunker


class _TTS:
    def __init__(self, events, stream=True):
        self.events, self.stream = events, stream

    def is_available(self):
        return self.stream

    def enqueue_streaming(self, voice=None):
        return TTSChunker(self, voice=voice)

    def speak(self, text, voice=None):
        self.events.append(("speak", text))

    def wait(self):
        pass


class _StreamBrain:
    on_token_callback = None

    def __init__(self, raw, result):
        self.raw, self.result = raw, result

    def process(self, text, state):
        for ch in self.raw:
            self.on_token_callback(ch)
        return self.result


def _setup(monkeypatch, events):
    monkeypatch.setattr(cm, "_brain_ok", True)
    monkeypatch.setattr(cm, "get_spot_state", lambda: {})
    monkeypatch.setattr(cm, "_resolve_voice", lambda b: None)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: "walk forward and tell me a joke")
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)
    monkeypatch.setattr(cm, "is_filler_only", lambda t: False)
    monkeypatch.setattr(cm, "_resume_talk_after_halt", lambda: None)
    monkeypatch.setattr(cm, "_last_safety_at", float("-inf"))
    monkeypatch.setattr(cm, "_tuning", cm._tuning.__class__(thinking_filler_s=60.0))
    monkeypatch.setattr(cm, "execute_on_spot",
                        lambda i, brain=None, speak=None: events.append(("action", i["intent"])) or True)
    cm._abort_chain.clear()


def _run(brain, tts, events):
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], brain=brain, tts=tts,
                         info={}, has_wake_detector=True, utt_t0=0.0,
                         on_speaking=lambda: events.append(("cue",)))


def test_streamed_reply_cues_before_first_sentence_and_before_actions(monkeypatch):
    events = []
    _setup(monkeypatch, events)
    raw = '{"actions": [{"action": "walk"}], "response": "Here I go. Watch me walk!"}'
    res = {"response": "Here I go. Watch me walk!", "actions": [{"intent": "walk", "params": {}}]}
    _run(_StreamBrain(raw, res), _TTS(events), events)
    assert events[0] == ("cue",)
    assert events[1][0] == "speak"
    assert events.count(("cue",)) == 1
    assert events.index(("cue",)) < events.index(("action", "walk"))


def test_non_streamed_reply_cues_before_say(monkeypatch):
    events = []
    _setup(monkeypatch, events)

    class B:
        on_token_callback = None

        def process(self, text, state):
            return {"response": "Hello!", "actions": []}
    _run(B(), _TTS(events, stream=False), events)
    assert events == [("cue",), ("speak", "Hello!")]


def test_silent_turn_never_cues(monkeypatch):
    events = []
    _setup(monkeypatch, events)

    class B:
        on_token_callback = None

        def process(self, text, state):
            return {"response": "", "actions": [{"intent": "sit", "params": {}}]}
    _run(B(), _TTS(events, stream=False), events)
    assert ("cue",) not in events  # main loop falls back to RESPONDING after the turn


def test_thinking_filler_does_not_cue(monkeypatch):
    events = []
    monkeypatch.setattr(cm, "_speaking_hook", lambda: events.append("cue"))
    cm._abort_chain.clear()
    cm._say(_TTS(events), "One moment.", cue=False)
    assert events == [("speak", "One moment.")]
    cm._speaking_hook = None


def test_hook_cleared_after_turn(monkeypatch):
    events = []
    _setup(monkeypatch, events)

    class B:
        on_token_callback = None

        def process(self, text, state):
            return {"response": "Hi.", "actions": []}
    _run(B(), _TTS(events, stream=False), events)
    assert cm._speaking_hook is None
