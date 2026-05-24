"""Persona registry — declarative YAML loader + runtime lookup.

Loads `config/personas.yaml` at boot. Provides:
- Persona dataclass (prompt_prefix, voices dict keyed by backend name)
- load_registry(path) -> dict[name, Persona]
- get_persona(name, registry) -> Persona  (falls back to default on miss + warns)
- list_personas(registry) -> sorted list[str]
- default_persona_name() -> str  (from SPOT_PERSONA env or "tour_guide")
"""
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

DEFAULT_PERSONA = "tour_guide"


class PersonaRegistryError(Exception):
    """Raised when the persona registry YAML cannot be loaded."""


@dataclass
class Persona:
    name: str
    prompt_prefix: str
    voices: dict   # backend_name -> voice_id
    sampling_overrides: dict = field(default_factory=dict)
    # sampling_overrides shape: {"vlm": {"temperature": 0.8}, ...}
    # Keys: action | freeform | vlm. Values merge on top of
    # SAMPLING_PROFILES from brain/llamacpp_backend.py.


def load_registry(path: str = "config/personas.yaml") -> dict:
    """Load persona registry from YAML file.

    Raises PersonaRegistryError if the file is missing or malformed.
    Returns dict[persona_name, Persona].
    """
    p = Path(path)
    if not p.exists():
        raise PersonaRegistryError(f"Persona registry not found: {path}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as e:
        raise PersonaRegistryError(f"Malformed persona YAML: {e}") from e
    if not isinstance(raw, dict):
        raise PersonaRegistryError(f"Top-level YAML must be a mapping, got {type(raw)}")

    registry = {}
    for name, entry in raw.items():
        if not isinstance(entry, dict):
            logger.warning(f"Skipping malformed persona entry: {name}")
            continue
        prefix = entry.get("prompt_prefix", "").strip()
        voices = entry.get("voices", {}) or {}
        sampling_overrides = entry.get("sampling_overrides", {}) or {}
        if not prefix:
            logger.warning(f"Persona {name} has empty prompt_prefix; skipping")
            continue
        registry[name] = Persona(
            name=name,
            prompt_prefix=prefix,
            voices=voices,
            sampling_overrides=sampling_overrides,
        )
    return registry


def get_persona(name: str, registry: dict) -> Persona:
    """Look up persona by name. Falls back to default + logs warning on miss."""
    if name in registry:
        return registry[name]
    fallback = default_persona_name()
    logger.warning(
        f"Unknown persona '{name}', falling back to '{fallback}'"
    )
    if fallback in registry:
        return registry[fallback]
    # Last-resort: return first available persona
    first = next(iter(registry.values()))
    logger.warning(f"Default '{fallback}' also missing; using '{first.name}'")
    return first


def list_personas(registry: dict) -> list:
    """Return sorted list of persona names."""
    return sorted(registry.keys())


def default_persona_name() -> str:
    """Return boot-time default persona from SPOT_PERSONA env or hardcoded fallback."""
    return os.environ.get("SPOT_PERSONA", DEFAULT_PERSONA)
