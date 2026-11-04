"""CR #6: the chain-wait sets in conversation/loop.py come from the generated
locomotion flag, and with today's registry they are exactly the base
(81fedfe client_mic.py) literal tuples, walk_to included (no wait)."""
from src.spot.conversation import loop as _loop
from src.spot.core.core_intents import all_actions
from src.spot.core.registry import ActionSpec, Feature, Registry
from src.spot.features import REGISTRY

# Copied from base client_mic.py (the inter-action chain wait).
BASE_NAV = ("go_to", "go_to_object", "follow_me", "tour", "patrol", "come_back", "emote")
BASE_MOVEMENT = ("walk", "strafe", "turn")
BASE_POSTURE = ("sit", "stand", "selfright", "body_height")


def _base_kind(cmd):
    if cmd in BASE_NAV:
        return "nav"
    if cmd in BASE_MOVEMENT:
        return "movement"
    if cmd in BASE_POSTURE:
        return "posture"
    return None


def test_chain_wait_equals_base_for_every_action():
    names = {a.name for a in all_actions(REGISTRY)} | {"unknown", ""}
    assert "walk_to" in names
    for name in sorted(names):
        assert _loop.chain_wait_kind(name) == _base_kind(name), name
    assert _loop.chain_wait_kind("walk_to") is None  # base: no wait (G1 replaces walk_to)


def test_new_locomotion_feature_action_gets_the_nav_wait(monkeypatch):
    import src.spot.features as _f
    reg = Registry()
    reg.add(Feature("g", actions=(ActionSpec("glide", lambda p, c: True, None, None,
                                             locomotion=True),)))
    monkeypatch.setattr(_f, "REGISTRY", reg)
    assert _loop.chain_wait_kind("glide") == "nav"
    assert _loop.chain_wait_kind("go_to") is None  # navigation not in this registry
    assert _loop.chain_wait_kind("emote") == "nav"
