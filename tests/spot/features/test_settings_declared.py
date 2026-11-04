from pathlib import Path

import src.spot.features.settings.handlers as settings_h
import src.spot.robot.dispatcher as d
from src.spot.features import REGISTRY
from src.spot.features.settings import FEATURE
from tests.spot.features._legacy_sets import LLM_PROMPT_SOURCE, declared_flags, legacy_flags

ROOT = Path(__file__).resolve().parents[3]


def test_flags_match_legacy_sets():
    assert [a.name for a in FEATURE.actions] == ["set_persona", "add_persona",
                                                 "set_volume", "set_backend"]
    for spec in FEATURE.actions:
        assert declared_flags(spec) == legacy_flags(spec.name), spec.name
        assert spec.arm_policy is None


def test_registered_and_prompt_lines_verbatim():
    prompt = (ROOT / LLM_PROMPT_SOURCE).read_text()
    for spec in FEATURE.actions:
        assert REGISTRY.action(spec.name) is spec
        assert spec.prompt_line in prompt
        assert spec.grammar_params is None
    assert FEATURE.intent_rules == ()


def test_settings_never_bring_up_a_session(monkeypatch):
    from src.spot.robot import CONNECTION
    import pytest
    monkeypatch.setattr(CONNECTION, "ensure", lambda *a, **k: pytest.fail("no bring-up"))
    assert d.dispatch_intent({"intent": "set_backend", "params": {"name": "bogus"}}) is False
    assert d.dispatch_intent({"intent": "set_volume", "params": {}}) is False
    assert d.dispatch_intent({"intent": "set_persona", "params": {"name": "x"}}) is False
    assert d.dispatch_intent({"intent": "add_persona", "params": {}}) is False
