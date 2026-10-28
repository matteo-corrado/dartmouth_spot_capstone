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


def test_prompt_starts_with_static_system_prompt(brain):
    content = brain._build_messages(
        "hi", state={"uptime_s": 12, "process_pid": 99, "battery_percent": 80})[0]["content"]
    assert content.startswith(llm_brain.SYSTEM_PROMPT[:200])
    assert "uptime_s" not in content and "process_pid" not in content
    assert "battery_percent: 80" in content


def test_prompt_has_mouth_emote_and_check_rules():
    p = llm_brain.SYSTEM_PROMPT
    assert "enable_mouth" in p and "disable_mouth" in p and "- emote:" in p
    assert 'reply ONLY "Let me check."' in p
    assert "Battery is looking good" not in p


def test_is_available_uses_active_backend(brain, monkeypatch):
    monkeypatch.setattr(llm_brain, "_BACKEND", _FakeBackend(up=False))
    assert brain.is_available() is False
    monkeypatch.setattr(llm_brain, "_BACKEND", _FakeBackend(up=True))
    assert brain.is_available() is True
