"""B5/B7/E1: static prompt first, persona + state last, no volatile keys."""
import pytest

import src.voice_control.llm_brain as llm_brain
from src.voice_control.llm_brain import LLMBrain


class _FakeBackend:
    name = "fake"

    def __init__(self, up=True):
        self.up = up

    def is_available(self):
        return self.up

    def chat(self, *a, **k):
        return '{"actions": [], "response": "ok"}'


@pytest.fixture
def brain(monkeypatch, tmp_path):
    from src.voice_control.chatbot import conversation_log
    monkeypatch.setattr(conversation_log, "DEFAULT_LOG_DIR", tmp_path / "conversations")
    monkeypatch.setattr(llm_brain, "_BACKEND", _FakeBackend())
    return LLMBrain()


def test_static_prompt_prefix_stable_across_personas(brain):
    a = brain._build_messages("hi", state={})[0]["content"]; brain.session_state.current_persona = "pirate"
    b = brain._build_messages("hi", state={})[0]["content"]
    assert a[:4000] == b[:4000] and "uptime_s" not in b


def test_full_static_portion_stable_across_add_persona(brain):
    static = llm_brain.SYSTEM_PROMPT           # includes knowledge packs
    assert "{{" not in static                  # no unfilled placeholders
    a = brain._build_messages("hi", state={})[0]["content"]
    default = next(iter(brain._persona_registry.values()))
    brain._persona_registry["zz_cowboy"] = default   # runtime add_persona
    brain.session_state.current_persona = "zz_cowboy"
    b = brain._build_messages("hi", state={})[0]["content"]
    assert a.startswith(static) and b.startswith(static)
    assert "zz_cowboy" not in a and "zz_cowboy" in b[len(static):]


def test_emote_parts_gated_on_grammar_and_section():
    assert llm_brain._emote_prompt_parts("action-name ::= \"stop\"", "EMOTES:\n- dance") == ("", "")
    assert llm_brain._emote_prompt_parts('"\\"emote\\""', None) == ("", "")
    bullet, section = llm_brain._emote_prompt_parts('"\\"emote\\""', "EMOTES:\n- dance\n")
    assert bullet.startswith("- emote:") and section == "EMOTES:\n- dance"


def test_live_prompt_emote_matches_grammar():
    has = "emote" in llm_brain._GRAMMAR_TEXT and bool(llm_brain._load_emote_section())
    assert ("- emote:" in llm_brain.SYSTEM_PROMPT) == has
    assert "EMOTES:\nEMOTES:" not in llm_brain.SYSTEM_PROMPT


def test_wait_available_retries_until_up(brain, monkeypatch):
    class Flaky(_FakeBackend):
        def __init__(self):
            self.n = 0
        def is_available(self):
            self.n += 1
            return self.n >= 3      # 503 / refused twice, then up
    flaky = Flaky()
    monkeypatch.setattr(llm_brain, "_BACKEND", flaky)
    assert brain.wait_available(timeout_s=2.0, poll_s=0.01) is True and flaky.n == 3


def test_wait_available_times_out(brain, monkeypatch):
    monkeypatch.setattr(llm_brain, "_BACKEND", _FakeBackend(up=False))
    assert brain.wait_available(timeout_s=0.05, poll_s=0.01) is False


def test_prompt_starts_with_static_system_prompt(brain):
    content = brain._build_messages(
        "hi", state={"uptime_s": 12, "process_pid": 99, "battery_percent": 80})[0]["content"]
    assert content.startswith(llm_brain.SYSTEM_PROMPT[:200])
    assert "uptime_s" not in content and "process_pid" not in content
    assert "battery_percent: 80" in content


def test_prompt_has_mouth_emote_and_check_rules():
    p = llm_brain.SYSTEM_PROMPT
    assert "enable_mouth" in p and "disable_mouth" in p
    assert 'reply ONLY "Let me check."' in p
    assert "Battery is looking good" not in p


def test_is_available_uses_active_backend(brain, monkeypatch):
    monkeypatch.setattr(llm_brain, "_BACKEND", _FakeBackend(up=False))
    assert brain.is_available() is False
    monkeypatch.setattr(llm_brain, "_BACKEND", _FakeBackend(up=True))
    assert brain.is_available() is True


def test_describe_persona_rewarms_cache(brain, monkeypatch):
    calls = []
    monkeypatch.setattr(brain, "warm_up_async", lambda: calls.append(1))
    brain._available = True
    brain.describe_persona("cowboy")
    assert calls == [1]
