"""Spot dance feature: the `emote` action (built-in choreography routines).

Flags copied from the hand-written sets (spec 6.3): emote is in none of
LOCOMOTION / ABORTING / _UNLOCKED / INFO / SHORT_PATH_BLOCKED /
_STATE_ONLY / _PERSONA / _MOTION_FREE. Its arm policy ("stowed", current
pose) stays in the core arm_policy_for (spec 7.1b), so arm_policy is None.
The GBNF has no emote-specific params rule (generic `params`), so
grammar_params is None. intent.py has no emote regex rule.
"""
from src.spot.core.registry import ActionSpec, Feature
from src.spot.features.dance.handlers import emote as _emote

# Verbatim from src/spot/brain/llm_brain.py _EMOTE_ACTION_LINE.
EMOTE_PROMPT_LINE = (
    '- emote: Perform a fun built-in routine (dance, bow, wag, etc.). '
    'Params: {"name": "<routine name>"}. Use only a routine name from the '
    'EMOTES list below.\n'
)

FEATURE = Feature(
    name="dance",
    actions=(
        ActionSpec(
            name="emote",
            handler=_emote,
            grammar_params=None,
            prompt_line=EMOTE_PROMPT_LINE,
            locomotion=False,
            aborting=False,
            unlocked=False,
            short_path_blocked=False,
            info=False,
            state_only=False,
            motion_free=False,
            persona=False,
            arm_policy=None,
            needs_session=True,
        ),
    ),
)
