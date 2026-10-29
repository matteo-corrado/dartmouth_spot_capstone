"""B11: filler-only utterances are ignored before the LLM call."""
from src.voice_control.intent import is_filler_only


def test_filler_only():
    assert is_filler_only("uh I think my uh") is False   # "I think my" = 3 words
    assert is_filler_only("um uh hmm") is True
    assert is_filler_only("okay so like") is True
    assert is_filler_only("") is True


def test_one_word_commands_are_not_filler():
    for word in ("hello", "sit", "dance", "yes"):
        assert is_filler_only(word) is False
    assert is_filler_only("um sit") is False


def test_okay_is_an_answer_mid_conversation():
    assert is_filler_only("Okay.") is True                       # fresh / default: still filler
    assert is_filler_only("Okay.", mid_conversation=True) is False
    assert is_filler_only("ok", mid_conversation=True) is False
    assert is_filler_only("um uh", mid_conversation=True) is True


def test_bare_okay_reply_reaches_the_brain(monkeypatch):
    import numpy as np
    import src.voice_control.client_mic as cm
    sent = []
    monkeypatch.setattr(cm, "send_to_asr", lambda stub, pcm: "Okay.")
    monkeypatch.setattr(cm, "check_safety_command", lambda t: None)
    monkeypatch.setattr(cm, "get_recorder", lambda: None)
    monkeypatch.setattr(cm, "_resume_talk_after_halt", lambda gen: None)
    monkeypatch.setattr(cm, "parse_intent", lambda t: sent.append(t) or None)
    info = {}
    cm.process_utterance(None, None, [np.zeros(16000, np.float32)], info=info, has_wake_detector=True)
    assert info.get("reached_llm") and sent == ["Okay"]
