from pathlib import Path

import src.spot.features.mouth.handlers as mouth_h
import src.voice_control.spot_dispatch as d
from src.spot.features import REGISTRY
from src.spot.features.mouth import FEATURE
from tests.spot.features._legacy_sets import LLM_PROMPT_SOURCE, declared_flags, legacy_flags

ROOT = Path(__file__).resolve().parents[3]


def test_flags_match_legacy_sets():
    assert [a.name for a in FEATURE.actions] == ["enable_mouth", "disable_mouth"]
    for spec in FEATURE.actions:
        assert declared_flags(spec) == legacy_flags(spec.name), spec.name
        assert spec.arm_policy is None  # dispatcher arm_policy_for decides, as today


def test_registered_and_prompt_lines_verbatim():
    prompt = (ROOT / LLM_PROMPT_SOURCE).read_text()
    for spec in FEATURE.actions:
        assert REGISTRY.action(spec.name) is spec
        assert spec.prompt_line in prompt
        assert spec.grammar_params is None
    assert FEATURE.intent_rules == ()
    assert FEATURE.start is mouth_h.start


def test_auto_arm_globals_read_live_through_spot_dispatch(monkeypatch):
    for name in ("_auto_arm", "_auto_arm_pending"):
        assert name not in vars(d)  # no stale binding shadows the forward
        for value in (True, False):
            monkeypatch.setattr(mouth_h, name, value)
            assert getattr(d, name) is value
    sentinel = object()
    monkeypatch.setattr(mouth_h, "_auto_arm_thread", sentinel)
    assert d._auto_arm_thread is sentinel


def test_core_callbacks_resolve_to_feature():
    import src.spot.robot as r
    sd = r.legacy_features()
    for name in ("_cancel_auto_arm", "_auto_arm_after_stand", "start_auto_arm",
                 "_handle_disable_mouth", "set_auto_arm", "auto_arm_if_ready",
                 "_handle_enable_mouth"):
        assert getattr(sd, name) is getattr(mouth_h, name)
    assert sd.AUTO_ARM_STAND_WAIT_S == mouth_h.AUTO_ARM_STAND_WAIT_S == 8.0


def test_enable_mouth_ensure_failure_touches_no_state(monkeypatch):
    """Legacy order: the session comes up before the handler runs, so a
    failed bring-up returns False with the auto-arm opt-in untouched."""
    from src.spot.robot import ARM, CONNECTION

    def boom(*a, **k):
        raise RuntimeError("no robot")
    monkeypatch.setattr(CONNECTION, "ensure", boom)
    monkeypatch.setattr(mouth_h, "_auto_arm", True)
    monkeypatch.setattr(ARM, "_mouth_resume", True)
    assert d.dispatch_intent({"intent": "disable_mouth", "params": {}}) is False
    assert mouth_h._auto_arm is True and ARM._mouth_resume is True
