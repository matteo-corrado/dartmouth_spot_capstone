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
    subscriptions: tuple[tuple[str, Callable], ...] = ()
    start: Callable[["Ctx"], None] | None = None
    # Called once at shutdown by main's _shutdown_runtime, AFTER the robot
    # session is closed (stop/stow/sit/power_off order untouched). Never
    # raises out of shutdown. No feature defines one yet.
    # RULE: stop must never send robot RPCs (no motion, arm, gripper or power
    # command). Release only the feature's own threads/files. main skips every
    # stop if the session close failed (CONNECTION._session still set).
    stop: Callable[["Ctx"], None] | None = None
    requires: tuple[str, ...] = ()


def _check_flags(feature: str, a: ActionSpec) -> None:
    """Safety rules for a feature action (Task 16 safety review S1)."""
    from src.spot.core.core_intents import CORE_INTENTS  # lazy: core_intents imports this module
    where = f"feature {feature!r} action {a.name!r}"
    if a.name in CORE_INTENTS:
        raise ValueError(f"{where}: name is a core intent (core intents are declared only in core_intents)")
    if a.aborting:
        raise ValueError(f"{where}: aborting=True is not allowed (ABORTING_INTENTS is a core constant)")
    if a.unlocked and not a.motion_free:
        raise ValueError(f"{where}: unlocked=True requires motion_free=True "
                         "(an action that skips the dispatch lock must not move the robot)")
    if a.locomotion and (a.unlocked or a.motion_free or a.state_only):
        raise ValueError(f"{where}: locomotion=True cannot be combined with "
                         "unlocked, motion_free or state_only")


class Registry:
    def __init__(self) -> None:
        self.features: list[Feature] = []
        self._actions: dict[str, ActionSpec] = {}
        self._loaded: set[str] = set()   # feature module names already added
        # (name, exception, traceback text) for every feature that failed to
        # load; main prints these loudly at startup (safety condition C6).
        self.failures: list[tuple[str, BaseException, str]] = []

    def add(self, feature: Feature) -> None:
        seen: set[str] = set()
        for a in feature.actions:
            if a.name in self._actions or a.name in seen:
                raise ValueError(f"duplicate action: {a.name}")
            seen.add(a.name)
            _check_flags(feature.name, a)
        for a in feature.actions:
            self._actions[a.name] = a
        self.features.append(feature)

    def load(self, names: list[str], import_fn=importlib.import_module) -> list:
        """Import and add each feature not already loaded (so main can call it
        again after the import-time load). Returns this call's failures as
        (name, exception, traceback text); they are also kept in .failures.

        Feature.requires is enforced: a feature whose required feature is not
        loaded (absent from `names` and earlier loads, or failed itself) is a
        load failure too, reported the same way (C6: fatal on the robot). All
        names are imported first, so ENABLED order does not matter; survivors
        are added in `names` order (grammar/prompt order is registration order)."""
        failed = []
        imported = []  # (module name, Feature), in names order
        for name in names:
            if name in self._loaded or any(n == name for n, _ in imported):
                continue
            try:
                mod = import_fn(f"src.spot.features.{name}")
                imported.append((name, mod.FEATURE))
            except Exception as exc:
                log.error("[Features] %s skipped: %s", name, exc)
                failed.append((name, exc, traceback.format_exc()))
        ok = {name for name, _ in imported}
        changed = True
        while changed:  # a feature whose dependency fails fails too
            changed = False
            have = {f.name for f in self.features} | {f.name for n, f in imported if n in ok}
            for name, feature in imported:
                missing = [r for r in feature.requires if r not in have]
                if name in ok and missing:
                    ok.discard(name)
                    changed = True
                    exc = ImportError(f"feature {feature.name!r} requires {', '.join(map(repr, missing))} "
                                      "(not loaded: add it to enabled.py, or remove this feature)")
                    log.error("[Features] %s skipped: %s", name, exc)
                    failed.append((name, exc, f"{exc}\n"))
        added = []
        for name, feature in imported:
            if name not in ok:
                continue
            try:
                self.add(feature)
                self._loaded.add(name)
                added.append((name, feature))
            except Exception as exc:
                log.error("[Features] %s skipped: %s", name, exc)
                failed.append((name, exc, traceback.format_exc()))
        # add() can reject a dependency that imported fine (duplicate action,
        # flag rule): re-run the cascade so its dependents come out too.
        changed = True
        while changed:
            changed = False
            have = {f.name for f in self.features}
            for name, feature in list(added):
                missing = [r for r in feature.requires if r not in have]
                if missing:
                    self._remove(name, feature)
                    added.remove((name, feature))
                    changed = True
                    exc = ImportError(f"feature {feature.name!r} requires {', '.join(map(repr, missing))} "
                                      "(it failed to register)")
                    log.error("[Features] %s skipped: %s", name, exc)
                    failed.append((name, exc, f"{exc}\n"))
        for name, _ in added:
            self.failures[:] = [x for x in self.failures if x[0] != name]
        for f in failed:  # one entry per feature: a retry replaces the old one
            self.failures[:] = [x for x in self.failures if x[0] != f[0]]
            self.failures.append(f)
        return failed

    def _remove(self, module_name: str, feature: Feature) -> None:
        """Undo add() for one feature (its actions, its entry, its loaded mark)."""
        self.features = [f for f in self.features if f is not feature]
        for a in feature.actions:
            if self._actions.get(a.name) is a:
                del self._actions[a.name]
        self._loaded.discard(module_name)

    def has(self, name: str) -> bool:
        """True when the feature module `name` is loaded (e.g. "gaze", which
        registers no actions)."""
        return name in self._loaded

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
