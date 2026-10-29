"""Feature registry: actions, intent rules, prompt lines."""
from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    handler: Callable[[dict, "Ctx"], bool]
    grammar_params: str | None
    prompt_line: str | None
    locomotion: bool = False
    aborting: bool = False
    unlocked: bool = False
    short_path_blocked: bool = False
    info: bool = False
    state_only: bool = False
    motion_free: bool = False
    persona: bool = False
    arm_policy: tuple[str, str] | None = None
    needs_session: bool = True
    # Position in the generated GBNF action-name rule and in the LLM prompt's
    # action list (src/spot/brain/grammar/generate.py). The two orders differ
    # at base; ties keep core-then-registration order.
    grammar_rank: int = 10_000
    prompt_rank: int = 10_000


@dataclass(frozen=True)
class IntentRule:
    priority: int
    pattern: str
    build: Callable[[re.Match], dict]


@dataclass(frozen=True)
class Feature:
    name: str
    actions: tuple[ActionSpec, ...] = ()
    intent_rules: tuple[IntentRule, ...] = ()
    spotctl_help: tuple[str, ...] = ()
    subscriptions: tuple[tuple[str, Callable], ...] = ()
    start: Callable[["Ctx"], None] | None = None
    stop: Callable[["Ctx"], None] | None = None
    requires: tuple[str, ...] = ()


class Registry:
    def __init__(self) -> None:
        self.features: list[Feature] = []
        self._actions: dict[str, ActionSpec] = {}

    def add(self, feature: Feature) -> None:
        for a in feature.actions:
            if a.name in self._actions:
                raise ValueError(f"duplicate action: {a.name}")
        for a in feature.actions:
            self._actions[a.name] = a
        self.features.append(feature)

    def load(self, names: list[str], import_fn=importlib.import_module) -> None:
        for name in names:
            try:
                mod = import_fn(f"src.spot.features.{name}")
                self.add(mod.FEATURE)
            except Exception as exc:
                log.error("[Features] %s skipped: %s", name, exc)

    def action(self, name: str) -> ActionSpec | None:
        return self._actions.get(name)

    def actions(self) -> list[ActionSpec]:
        return list(self._actions.values())

    def names(self, flag: str) -> frozenset[str]:
        return frozenset(n for n, a in self._actions.items() if getattr(a, flag))

    def grammar_action_names(self) -> list[str]:
        return list(self._actions)

    def intent_rules(self) -> list[IntentRule]:
        rules = [r for f in self.features for r in f.intent_rules]
        return sorted(rules, key=lambda r: r.priority)

    def prompt_lines(self) -> list[str]:
        return [a.prompt_line for a in self._actions.values() if a.prompt_line]
