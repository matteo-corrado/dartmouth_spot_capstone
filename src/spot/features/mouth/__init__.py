"""Spot mouth animation feature: enable_mouth / disable_mouth and the
automatic fold + mouth once standing. Per spec 7.1b close-mouth-before-stow
and the talk pose stay in src/spot/robot/arm.py."""
from src.spot.core.registry import ActionSpec, Feature
from src.spot.features.mouth import handlers as _h

FEATURE = Feature(
    name="mouth",
    actions=(
        ActionSpec(
            name="enable_mouth",
            handler=_h.enable_mouth_action,
            grammar_params=None,  # spot_action.gbnf has one generic params rule
            prompt_line='- enable_mouth: Turn on your gripper "mouth" that moves while you talk. No params. Use for "use your mouth", "talk with your mouth".',
        ),
        ActionSpec(
            name="disable_mouth",
            handler=_h.disable_mouth_action,
            grammar_params=None,
            prompt_line='- disable_mouth: Turn off the gripper mouth. No params. Use for "stop moving your mouth", "close your mouth".',
        ),
    ),
    intent_rules=(),  # intent.py has no regex rule for these intents
    start=_h.start,
)
