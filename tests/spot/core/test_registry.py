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


def test_load_returns_failures_and_skips_loaded():
    """Task 15: main calls load(ENABLED) again after the import-time load;
    loaded features are skipped (no duplicate-action error) and the call's
    failures come back with a traceback for the loud startup report (C6)."""
    good = Feature("good", actions=(act("g"),))
    calls = []

    def imp(path):
        calls.append(path)
        if path.endswith("bad"):
            raise ImportError("nope")
        return SimpleNamespace(FEATURE=good)

    r = Registry()
    first = r.load(["good", "bad"], import_fn=imp)
    assert [(n, str(e)) for n, e, _ in first] == [("bad", "nope")]
    assert "ImportError: nope" in first[0][2]
    second = r.load(["good", "bad"], import_fn=imp)  # good is not re-added
    assert [n for n, _, _ in second] == ["bad"]
    assert calls.count("src.spot.features.good") == 1
    assert len(r.failures) == 2
