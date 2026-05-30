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
