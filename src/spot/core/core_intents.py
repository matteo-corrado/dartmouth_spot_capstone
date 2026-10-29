"""Core intents: declared here, never by a feature, so disabling a feature can
never remove them (spec 6.2). No bosdyn import: the brain, the conversation
loop and the tests read this table without the robot stack.

The branches themselves stay in src/spot/robot/dispatcher.py (_dispatch_core),
so every CORE_ACTIONS handler is None. The generated grammar, prompt, intent
table and intent sets are CORE_ACTIONS / CORE_INTENT_RULES plus the feature
registry (src/spot/brain/grammar/generate.py, intent_names below).
"""
from __future__ import annotations

from src.spot.core.registry import ActionSpec, IntentRule

# The safety family stays a frozen constant: the safety path never looks at
# the feature registry, and "shutdown" is no action anywhere (base value).
SAFETY_INTENTS = frozenset({"stop", "freeze", "estop"})
SHUTDOWN_INTENTS = frozenset({"power_off", "shutdown", "sit"})  # stop, stow, then act
# Intents that abort any locomotion still being set up on another thread.
ABORTING_INTENTS = SAFETY_INTENTS | SHUTDOWN_INTENTS | {"stow"}

_SAFETY = dict(aborting=True, unlocked=True, motion_free=True)
_INFO = dict(unlocked=True, info=True, motion_free=True)


def _core(name, prompt_line, rank, **flags):
    # grammar and prompt orders agree for every core action
    return ActionSpec(name=name, handler=None, grammar_params=None, prompt_line=prompt_line,
                      grammar_rank=rank, prompt_rank=rank, **flags)


CORE_ACTIONS: tuple[ActionSpec, ...] = (
    _core("stop", "- stop: Stop all movement. No params.", 0, **_SAFETY),
    _core("freeze", "- freeze: Hold current position. No params.", 10, **_SAFETY),
    _core("estop", "- estop: Emergency stop (urgent only). No params.", 20, **_SAFETY),
    _core("stand", "- stand: Stand up from sitting. No params.", 30),
    _core("sit", "- sit: Sit down. No params.", 40, aborting=True),
    _core("selfright", "- selfright: Recover from a fall. No params.", 50),
    _core("battery_status", "- battery_status: Check battery level. No params.", 240, **_INFO),
    _core("status", "- status: Full robot status report. No params.", 250, **_INFO),
    _core("power_off", "- power_off: Safely power off. No params.", 260,
          aborting=True, short_path_blocked=True),
    _core("stow", None, 10_000, aborting=True),  # spotctl only: not in the grammar or prompt
)

CORE_INTENTS = frozenset(a.name for a in CORE_ACTIONS)


def _fixed(intent_name):
    def build(m):
        return {"intent": intent_name, "params": {}}
    return build


# The core rows of the old intent.COMMANDS table (priority = old index x 10).
CORE_INTENT_RULES: tuple[IntentRule, ...] = (
    # === POSTURE ===
    IntentRule(0, r"\b(?:stand|stand up|get up)\b", _fixed("stand")),
    IntentRule(10, r"\b(?:sit|sit down|lay down)\b", _fixed("sit")),
    IntentRule(20, r"\b(?:self[\s-]?right|get up from fall|recover)\b", _fixed("selfright")),
    # === STATUS & POWER ===
    IntentRule(330, r"\b(?:battery|battery status|how much battery|charge level|power level)\b",
               _fixed("battery_status")),
    IntentRule(340, r"\b(?:status|status check|how are you|robot status)\b", _fixed("status")),
    IntentRule(350, r"\b(?:power off|shut\s*down|turn off)\b", _fixed("power_off")),
)


def _registry(registry):
    if registry is None:
        import src.spot.features as _features  # call time: tests swap the package attribute
        registry = _features.REGISTRY
    return registry


def all_actions(registry=None) -> list[ActionSpec]:
    """Core actions first, then the registered feature actions."""
    return list(CORE_ACTIONS) + _registry(registry).actions()


def intent_names(flag: str, registry=None) -> frozenset[str]:
    """Core intents with `flag` set, plus the registered feature intents with
    it. Read at call time, so a disabled feature drops out."""
    core = frozenset(a.name for a in CORE_ACTIONS if getattr(a, flag))
    return core | _registry(registry).names(flag)


def intent_rules(registry=None) -> list[IntentRule]:
    """The regex table: core rows plus feature rows, by priority (stable)."""
    rules = list(CORE_INTENT_RULES) + _registry(registry).intent_rules()
    return sorted(rules, key=lambda r: r.priority)
