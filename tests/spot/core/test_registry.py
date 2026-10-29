import importlib
import logging
import re
from types import SimpleNamespace

import pytest

from src.spot.core.registry import ActionSpec, Feature, IntentRule, Registry


def act(name, **kw):
    return ActionSpec(name, lambda p, c: True, None, kw.pop("prompt_line", None), **kw)


def test_duplicate_action_rejected():
    r = Registry()
    r.add(Feature("a", actions=(act("x"),)))
    with pytest.raises(ValueError):
        r.add(Feature("b", actions=(act("x"),)))


def test_load_skips_broken_feature(caplog):
    good = Feature("good", actions=(act("g"),))

    def imp(path):
        if path.endswith("bad"):
            raise ImportError("nope")
        return SimpleNamespace(FEATURE=good)

    r = Registry()
    with caplog.at_level(logging.ERROR):
        r.load(["bad", "good"], import_fn=imp)
    assert r.action("g") is not None
    assert r.action("bad") is None
    assert "[Features] bad skipped: nope" in caplog.text


def test_names_by_flag():
    r = Registry()
    r.add(Feature("f", actions=(act("a", locomotion=True), act("b"), act("c", locomotion=True))))
    assert r.names("locomotion") == frozenset({"a", "c"})
    assert r.grammar_action_names() == ["a", "b", "c"]


def test_prompt_lines():
    r = Registry()
    r.add(Feature("f", actions=(act("a", prompt_line="A"), act("b"))))
    assert r.prompt_lines() == ["A"]


def test_intent_rules_priority_stable():
    b = lambda m: {}
    r = Registry()
    r.add(Feature("f1", intent_rules=(IntentRule(5, "a", b), IntentRule(1, "b", b))))
    r.add(Feature("f2", intent_rules=(IntentRule(5, "c", b), IntentRule(1, "d", b))))
    assert [x.pattern for x in r.intent_rules()] == ["b", "d", "a", "c"]


# --- Task 16 safety review S1: Registry.add rejects unsafe feature actions ---
@pytest.mark.parametrize("spec, msg", [
    (act("stop"), "core intent"),
    (act("status"), "core intent"),
    (act("x", aborting=True), "aborting=True"),
    (act("x", unlocked=True), "requires motion_free"),
    (act("x", locomotion=True, unlocked=True, motion_free=True), "locomotion=True"),
    (act("x", locomotion=True, motion_free=True), "locomotion=True"),
    (act("x", locomotion=True, state_only=True), "locomotion=True"),
])
def test_add_rejects_unsafe_flags(spec, msg):
    r = Registry()
    with pytest.raises(ValueError, match=msg):
        r.add(Feature("bad", actions=(spec,)))
    assert r.action(spec.name) is None and r.features == []


def test_add_accepts_safe_combinations():
    r = Registry()
    r.add(Feature("ok", actions=(act("a", unlocked=True, motion_free=True),
                                 act("b", locomotion=True, short_path_blocked=True),
                                 act("c", state_only=True, unlocked=True, motion_free=True))))
    assert r.names("unlocked") == {"a", "c"}


def test_enabled_features_pass_the_checks():
    from src.spot.features.enabled import ENABLED
    r = Registry()
    for name in ENABLED:
        r.add(importlib.import_module(f"src.spot.features.{name}").FEATURE)
    assert [f.name for f in r.features] == ENABLED


def test_duplicate_action_within_one_feature_rejected():
    r = Registry()
    with pytest.raises(ValueError, match="duplicate action: x"):
        r.add(Feature("a", actions=(act("x"), act("x"))))
    assert r.features == [] and r.action("x") is None
