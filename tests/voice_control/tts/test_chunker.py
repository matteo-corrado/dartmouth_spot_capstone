# tests/voice_control/tts/test_chunker.py
"""Stage 2F P1 chunker TDD: abbreviation-safe split + suppress/describe path."""
from src.voice_control.text_segment import split_sentences


def test_simple_two_sentences():
    complete, remainder = split_sentences("Hello there. How are you?")
    assert complete == ["Hello there.", "How are you?"]
    assert remainder == ""


def test_keeps_trailing_partial_as_remainder():
    complete, remainder = split_sentences("I am walking. And then")
    assert complete == ["I am walking."]
    assert remainder == "And then"


def test_does_not_split_on_title_abbreviation():
    complete, remainder = split_sentences("Dr. Smith is here. Hello.")
    assert "Dr. Smith is here." in complete
    assert not any(c.strip() == "Dr." for c in complete)


def test_does_not_split_on_eg_ie():
    complete, remainder = split_sentences("Bring tools, e.g. a wrench. Done.")
    assert any("e.g. a wrench." in c for c in complete)


def test_does_not_split_on_us_acronym():
    complete, remainder = split_sentences("I visited the U.S. last year. It was great.")
    assert any("U.S. last year." in c for c in complete)


def test_no_split_without_boundary():
    complete, remainder = split_sentences("just a fragment with no end")
    assert complete == []
    assert remainder == "just a fragment with no end"


def test_clause_fallback_on_long_unpunctuated_buffer():
    # > 80 chars, no sentence end, but commas — split on the clause boundary so
    # TTS latency stays bounded.
    long = ("first we will walk to the door, then we will turn around slowly, "
            "and finally we will sit")
    complete, remainder = split_sentences(long)
    assert complete  # at least one clause emitted
    assert remainder  # tail kept


def test_exclamation_and_question():
    complete, remainder = split_sentences("Watch out! Are you ok? Yes")
    assert complete == ["Watch out!", "Are you ok?"]
    assert remainder == "Yes"


def test_decimal_not_split():
    complete, remainder = split_sentences("Move 3.5 meters forward. Stop.")
    assert any("3.5 meters forward." in c for c in complete)


def test_multiple_spaces_normalized():
    complete, _ = split_sentences("Done.   Next.")
    assert complete == ["Done.", "Next."]


def test_empty_input():
    assert split_sentences("") == ([], "")


def test_abbrev_then_real_end():
    complete, remainder = split_sentences("Meet Mr. Lee. Then go.")
    assert any("Mr. Lee." in c for c in complete)
    assert not any(c.strip() == "Mr." for c in complete)


from src.voice_control.spot_tts import TTSChunker


class _FakeTTS:
    """Captures speak() calls instead of rendering audio."""
    def __init__(self):
        self.spoken = []
    def speak(self, text, voice=None):
        self.spoken.append(text)


def _feed(chunker, s):
    for ch in s:
        chunker.accept(ch)


def test_chunker_emits_response_sentences_from_json_stream():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": "Hello there. How are you?"}')
    c.flush()
    assert tts.spoken == ["Hello there.", "How are you?"]


def test_chunker_does_not_split_abbreviation_in_response():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": "Meet Dr. Lee now. Then we walk."}')
    c.flush()
    assert "Meet Dr. Lee now." in tts.spoken
    assert "Dr." not in tts.spoken


def test_chunker_suppresses_on_describe_action():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [{"action": "describe"}], "response": "I see a red chair."}')
    c.flush()
    assert c.suppressed is True
    assert tts.spoken == []  # nothing spoken — client_mic speaks the stock ack


def test_chunker_does_not_suppress_without_describe():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [{"action": "stand"}], "response": "Standing up now."}')
    c.flush()
    assert c.suppressed is False
    assert tts.spoken == ["Standing up now."]


def test_chunker_flush_emits_trailing_partial():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": "No terminal punctuation here"')
    c.flush()
    assert tts.spoken == ["No terminal punctuation here"]


def test_chunker_handles_escaped_quote_in_response():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": "She said \\"hi\\" to me. Bye."}')
    c.flush()
    assert any("hi" in s for s in tts.spoken)
    assert "Bye." in tts.spoken


def test_chunker_callable_alias():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    for ch in '{"actions": [], "response": "One. Two."}':
        c(ch)  # __call__ == accept
    c.flush()
    assert tts.spoken == ["One.", "Two."]


def test_chunker_empty_response():
    tts = _FakeTTS()
    c = TTSChunker(tts)
    _feed(c, '{"actions": [], "response": ""}')
    c.flush()
    assert tts.spoken == []
