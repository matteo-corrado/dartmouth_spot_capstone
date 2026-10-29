from src.voice_control.vision.vocab import COCO_NAMES, YOLOE_NAMES, resolve_label


def test_resolve_coco():
    assert resolve_label("a backpack") == ("coco", "backpack")
    assert resolve_label("Bags") == ("coco", "backpack")
    assert resolve_label("the people") == ("coco", "person")


def test_resolve_yoloe():
    assert resolve_label("whiteboard") == ("yoloe", "whiteboard")
    assert resolve_label("posters") == ("yoloe", "poster")


def test_resolve_unknown():
    assert resolve_label("unicorn") is None
    assert resolve_label("") is None


def test_yoloe_names_match_spike():
    assert len(YOLOE_NAMES) == 150
    assert YOLOE_NAMES[:3] == ["person", "door", "stairs"]


def test_coco_names():
    assert len(COCO_NAMES) == 80
    assert COCO_NAMES[0] == "person" and COCO_NAMES[79] == "toothbrush"


def test_synonyms_and_plurals():
    assert resolve_label("the sofa") == ("coco", "couch")
    assert resolve_label("some boxes") == ("yoloe", "box")
    assert resolve_label("rubbish bin") == ("yoloe", "trash can")
    assert resolve_label("phone") == ("coco", "cell phone")
