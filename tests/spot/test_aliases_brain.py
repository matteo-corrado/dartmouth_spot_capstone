import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

PAIRS = [
    ("intent", "brain.intent"),
    ("llm_brain", "brain.llm_brain"),
    ("brain", "brain.backends"),
    ("brain.llamacpp_backend", "brain.backends.llamacpp_backend"),
    ("brain.ollama_backend", "brain.backends.ollama_backend"),
    ("chatbot.persona", "brain.persona"),
    ("chatbot.voice_discovery", "brain.voice_discovery"),
    ("chatbot.conversation_log", "conversation.conversation_log"),
    ("chatbot.session_state", "conversation.session_state"),
    ("speaker_lock", "conversation.speaker_lock"),
    ("speaker_embed", "conversation.voiceprint"),
]


@pytest.mark.parametrize("old,new", PAIRS)
def test_alias_identity(old, new):
    o = importlib.import_module(f"src.voice_control.{old}")
    n = importlib.import_module(f"src.spot.{new}")
    assert o is n


def test_chatbot_package_importable():
    from src.voice_control import chatbot
    from src.spot.brain import persona
    assert chatbot.persona is persona


def test_data_path_constants_exist():
    from src.spot.brain import llm_brain
    from src.spot.brain.backends import llamacpp_backend
    from src.spot.brain import persona
    from src.spot.conversation import conversation_log, voiceprint
    assert llm_brain._GRAMMAR_PATH.is_file()
    assert llm_brain._KNOWLEDGE_DIR.is_dir()
    for name in llm_brain._KNOWLEDGE_FILES:
        assert (llm_brain._KNOWLEDGE_DIR / name).is_file()
    assert Path(llamacpp_backend.GRAMMAR_PATH).is_file()
    assert (persona._REPO_ROOT / "config" / "personas.yaml").is_file()
    assert Path(conversation_log.DEFAULT_LOG_DIR).parent == ROOT / "logs"
    assert Path(voiceprint.DEFAULT_MODEL_PATH).parent.parent == ROOT / "models"
    # old data paths still resolve (symlinks) for non-owned readers
    assert (ROOT / "src/voice_control/grammar/spot_action.gbnf").is_file()
    assert (ROOT / "src/voice_control/knowledge/tour_route.md").is_file()
