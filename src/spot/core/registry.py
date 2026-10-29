"""Feature registry: actions, intent rules, prompt lines."""
from __future__ import annotations

import importlib
import logging
import re
import traceback
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
        self._loaded: set[str] = set()   # feature module names already added
        # (name, exception, traceback text) for every feature that failed to
        # load; main prints these loudly at startup (safety condition C6).
        self.failures: list[tuple[str, BaseException, str]] = []

    def add(self, feature: Feature) -> None:
        for a in feature.actions:
            if a.name in self._actions:
                raise ValueError(f"duplicate action: {a.name}")
        for a in feature.actions:
            self._actions[a.name] = a
        self.features.append(feature)

    def load(self, names: list[str], import_fn=importlib.import_module) -> list:
        """Import and add each feature not already loaded (so main can call it
        again after the import-time load). Returns this call's failures as
        (name, exception, traceback text); they are also kept in .failures."""
        failed = []
        for name in names:
            if name in self._loaded:
                continue
            try:
                mod = import_fn(f"src.spot.features.{name}")
                self.add(mod.FEATURE)
                self._loaded.add(name)
                self.failures[:] = [x for x in self.failures if x[0] != name]
            except Exception as exc:
                log.error("[Features] %s skipped: %s", name, exc)
                failed.append((name, exc, traceback.format_exc()))
        for f in failed:  # one entry per feature: a retry replaces the old one
            self.failures[:] = [x for x in self.failures if x[0] != f[0]]
            self.failures.append(f)
        return failed

    def action(self, name: str) -> ActionSpec | None:
        return self._actions.get(name)

    def names(self, flag: str) -> frozenset[str]:
        return frozenset(n for n, a in self._actions.items() if getattr(a, flag))

    def grammar_action_names(self) -> list[str]:
        return list(self._actions)

    def intent_rules(self) -> list[IntentRule]:
        rules = [r for f in self.features for r in f.intent_rules]
        return sorted(rules, key=lambda r: r.priority)

    def prompt_lines(self) -> list[str]:
        return [a.prompt_line for a in self._actions.values() if a.prompt_line]
