"""vision.phrases: go-to phrase parsing moved out of spot_dispatch (M6)."""
import pytest

from src.voice_control.vision import phrases as ph


@pytest.mark.parametrize("desc,person", [
    ("me", True), ("Me.", True), ("person in red shirt", True), ("person on my left", True),
    ("the one on the left", True), ("the man in the blue jacket", True),
    ("someone wearing a hat", True), ("the guy with the red shirt", True),
    ("persons", True), ("the kids", True), ("children", True), ("the ladies", True),
    ("those guys", True), ("humans", True),
    ("red chair", False), ("the chair on my left", False), ("the cup on the floor", False),
    ("door", False), ("the bag with the stripes", False), ("the manhole", False)])
def test_is_person_phrase(desc, person):
    assert ph.is_person_phrase(desc) is person


@pytest.mark.parametrize("desc,label", [
    ("red chair", "chair"), ("the cup on the floor", "cup"), ("dining table", "dining table"),
    ("backpacks", "backpack"), ("persons", "person"), ("the kids", "person"),
    ("flibbertigibbet", None), ("", None)])
def test_object_label(desc, label):
    hit = ph.object_label(desc)
    assert (hit[1] if hit else None) == label


@pytest.mark.parametrize("desc,want", [
    ("persons", True), ("the kids", True), ("me", True), ("a man", True),
    ("chair", False), ("the red backpack", False), ("flibbertigibbet", False)])
def test_wants_person(desc, want):
    assert ph.wants_person(desc) is want


def test_clean_phrase():
    assert ph.clean_phrase("  The RED, chair! ") == "the red chair"
    assert ph.clean_phrase(None) == ""


def test_dispatcher_delegates():
    import src.voice_control.spot_dispatch as d
    assert not hasattr(d, "_PERSON_RE")  # parsing lives in vision.phrases now
    assert d._is_person_phrase("the kids") is True
    assert d._vision_object_label("red chair") == ph.object_label("red chair")
