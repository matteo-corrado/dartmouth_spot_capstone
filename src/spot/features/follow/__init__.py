"""follow feature: follow_me (Task 13).

Flags copied from the hand-written sets at base: LOCOMOTION_INTENTS and
intent.SHORT_PATH_BLOCKED_INTENTS. arm_policy stays None (dispatcher
arm_policy_for / Arm.policy_for still decides). No GBNF params rule at base.
prompt_line: the llm_brain.SYSTEM_PROMPT line, verbatim. Intent rule:
intent.COMMANDS[24], priority 240.
"""
from src.spot.core.registry import ActionSpec, Feature, IntentRule
from src.spot.features.follow import handlers as _h


def _build(m):
    return {"intent": "follow_me", "params": {}}


FEATURE = Feature(
    name="follow",
    actions=(
        ActionSpec(
            name="follow_me", handler=_h.handle_follow_me, grammar_params=None,
            prompt_line='- follow_me: Follow the nearest person, maintaining distance. No params. Use for "follow me", "come with me", "tag along".',
            locomotion=True, short_path_blocked=True),
    ),
    intent_rules=(
        IntentRule(priority=240, pattern=r"\b(?:follow me|come with me|tag along)\b", build=_build),
    ),
)
