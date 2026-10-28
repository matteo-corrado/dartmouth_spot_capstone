"""B11: filler-only utterances are ignored before the LLM call."""
from src.voice_control.intent import is_filler_only


def test_filler_only():
    assert is_filler_only("uh I think my uh") is False   # "I think my" = 3 words
    assert is_filler_only("um uh hmm") is True
    assert is_filler_only("okay so like") is True
    assert is_filler_only("hello") is True
