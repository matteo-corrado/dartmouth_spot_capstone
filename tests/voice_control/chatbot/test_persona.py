import os
import pytest
from src.voice_control.chatbot.persona import (
    Persona,
    load_registry,
    get_persona,
    list_personas,
    default_persona_name,
    PersonaRegistryError,
)


REGISTRY_PATH = "config/personas.yaml"


def test_load_registry_returns_six_personas():
    reg = load_registry(REGISTRY_PATH)
    assert {
        "tour_guide", "pirate", "snarky", "butler", "shakespeare", "gen_z",
    } <= set(reg.keys())


def test_persona_has_prompt_prefix_and_voices():
    reg = load_registry(REGISTRY_PATH)
    pirate = reg["pirate"]
    assert isinstance(pirate, Persona)
    assert "pirate" in pirate.prompt_prefix.lower()
    assert pirate.voices["kokoro_v1"] == "am_michael"
    assert pirate.voices["elevenlabs"] == "Xq2dbIWNPChFB77imiDe"
    # Persona with no override returns empty dict; pirate carries a vlm override.
    assert reg["butler"].sampling_overrides == {}
    assert pirate.sampling_overrides["vlm"]["temperature"] == 0.8
    assert pirate.ack_template  # non-empty in-character ACK


def test_get_persona_returns_default_on_unknown_name(caplog):
    reg = load_registry(REGISTRY_PATH)
    p = get_persona("nonexistent_persona", reg)
    assert p.prompt_prefix == reg["default"].prompt_prefix
    # Warning should be logged
    assert any("nonexistent_persona" in rec.message for rec in caplog.records)


def test_list_personas_returns_sorted_names():
    reg = load_registry(REGISTRY_PATH)
    names = list_personas(reg)
    assert names == sorted(names)
    assert "tour_guide" in names


def test_default_persona_name_from_env(monkeypatch):
    monkeypatch.setenv("SPOT_PERSONA", "pirate")
    assert default_persona_name() == "pirate"


def test_default_persona_name_fallback(monkeypatch):
    monkeypatch.delenv("SPOT_PERSONA", raising=False)
    assert default_persona_name() == "default"


def test_load_registry_missing_file_raises():
    with pytest.raises(PersonaRegistryError):
        load_registry("nonexistent/path.yaml")


def test_voice_id_for():
    from src.voice_control.chatbot.persona import voice_id_for
    reg = load_registry(REGISTRY_PATH)
    pirate = reg["pirate"]
    assert voice_id_for(pirate, "kokoro") == "am_michael"
    assert voice_id_for(pirate, "elevenlabs") == "Xq2dbIWNPChFB77imiDe"


def test_persona_mouth_intensity_loaded_and_defaulted(tmp_path):
    reg = load_registry()  # real config/personas.yaml
    assert get_persona("pirate", reg).mouth_intensity == 1.3
    assert get_persona("tour_guide", reg).mouth_intensity == 1.0  # default
