"""Detector vocabularies and spoken-phrase -> label resolution.

COCO_NAMES is the Ultralytics COCO order (class index = list index). YOLOE_NAMES
is the open-vocabulary list baked into the exported yoloe-26l-seg ONNX; its
order IS the model's class index, so never reorder it without re-exporting.
"""
from __future__ import annotations

COCO_NAMES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
]

YOLOE_NAMES = [
    'person', 'door', 'stairs', 'staircase', 'elevator', 'window', 'wall', 'floor',
    'ceiling light', 'poster', 'sign', 'banner', 'whiteboard', 'blackboard',
    'projector screen', 'monitor', 'tv', 'laptop', 'computer', 'keyboard', 'mouse',
    'phone', 'tablet', 'table', 'desk', 'chair', 'office chair', 'stool', 'bench',
    'couch', 'shelf', 'bookshelf', 'cabinet', 'drawer', 'locker', 'cart', 'trash can',
    'recycling bin', 'box', 'cardboard box', 'crate', 'backpack', 'bag', 'handbag',
    'suitcase', 'briefcase', 'bottle', 'water bottle', 'cup', 'mug', 'coffee cup',
    'can', 'plate', 'bowl', 'food', 'pizza', 'snack', 'fridge', 'microwave', 'sink',
    'plant', 'potted plant', 'flower', 'tree', 'lamp', 'clock', 'fan', 'heater',
    'fire extinguisher', 'exit sign', 'outlet', 'cable', 'power strip', 'toolbox',
    'tool', 'screwdriver', 'wrench', 'hammer', 'drill', 'scissors', 'tape', 'ladder',
    'bucket', 'robot', 'robot arm', 'drone', '3d printer', 'printer', 'oscilloscope',
    'soldering iron', 'circuit board', 'battery', 'helmet', 'safety glasses', 'gloves',
    'jacket', 'coat', 'hat', 'cap', 'shoe', 'umbrella', 'glasses', 'book', 'notebook',
    'paper', 'pen', 'marker', 'folder', 'envelope', 'badge', 'lanyard', 'name tag',
    'camera', 'tripod', 'microphone', 'speaker', 'headphones', 'guitar', 'ball', 'toy',
    'dog', 'cat', 'bicycle', 'scooter', 'skateboard', 'wheelchair', 'stroller', 'car',
    'truck', 'bus', 'traffic cone', 'barrier', 'rope', 'flag', 'balloon', 'cake',
    'award', 'trophy', 'easel', 'podium', 'stage', 'curtain', 'mirror', 'picture frame',
    'painting', 'sculpture', 'vending machine', 'water fountain', 'coat rack',
    'umbrella stand',
]

SYNONYMS = {
    "bag": "backpack", "rucksack": "backpack", "tv": "tv", "television": "tv",
    "telly": "tv", "mobile": "cell phone", "phone": "cell phone", "sofa": "couch",
    "bin": "trash can", "rubbish bin": "trash can", "people": "person",
    "someone": "person", "man": "person", "woman": "person", "kid": "person",
    "child": "person",
}

_ARTICLES = {"a", "an", "the", "some"}


def _lookup(name: str) -> tuple[str, str] | None:
    name = SYNONYMS.get(name, name)
    if name in COCO_NAMES:
        return ("coco", name)
    if name in YOLOE_NAMES:
        return ("yoloe", name)
    return None


def resolve_label(phrase: str) -> tuple[str, str] | None:
    """Map a spoken noun phrase to ("coco"|"yoloe", canonical_name), or None."""
    words = [w for w in phrase.lower().split() if w not in _ARTICLES]
    if not words:
        return None
    text = " ".join(words)
    candidates = [text]
    if text.endswith("es"):
        candidates.append(text[:-2])
    if text.endswith("s"):
        candidates.append(text[:-1])
    for cand in candidates:
        hit = _lookup(cand)
        if hit:
            return hit
    return None
