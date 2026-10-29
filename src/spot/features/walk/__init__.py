"""walk feature: walk, turn, strafe, body_height, set_speed (Task 13).

Flags copied from the hand-written sets at base: LOCOMOTION_INTENTS (walk,
turn, strafe) and intent.SHORT_PATH_BLOCKED_INTENTS (walk, turn). arm_policy
stays None: the arm policy is still dispatcher.arm_policy_for (Arm.policy_for),
which depends on the live pose. The base GBNF has no per-action params rules,
so grammar_params is None. prompt_line: the llm_brain.SYSTEM_PROMPT line,
verbatim. Intent rules: intent.COMMANDS entries, priority = index x 10.
"""
from src.spot.core.registry import ActionSpec, Feature, IntentRule
from src.spot.features.walk import handlers as _h


def _rule(index, pattern, intent, template):
    def build(m, _intent=intent, _template=template):
        from src.spot.brain.intent import _resolve_param
        return {"intent": _intent,
                "params": {k: _resolve_param(v, m) for k, v in _template.items()}}
    return IntentRule(priority=index * 10, pattern=pattern, build=build)


FEATURE = Feature(
    name="walk",
    actions=(
        ActionSpec(
            name="walk", handler=_h.handle_walk, grammar_params=None,
            prompt_line='- walk: Walk forward/backward. Params: {"direction": "forward"|"backward", "distance": <meters>}. Default distance 1.0.',
            locomotion=True, short_path_blocked=True),
        ActionSpec(
            name="strafe", handler=_h.handle_strafe, grammar_params=None,
            prompt_line='- strafe: Move sideways. Params: {"direction": "left"|"right", "distance": <meters>}. Default distance 0.5.',
            locomotion=True),
        ActionSpec(
            name="turn", handler=_h.handle_turn, grammar_params=None,
            prompt_line='- turn: Rotate in place. Params: {"deg": <degrees 0-360>, "dir": "left"|"right"}. Default 90 degrees. "turn around" = 180.',
            locomotion=True, short_path_blocked=True),
        ActionSpec(
            name="body_height", handler=_h.handle_body_height, grammar_params=None,
            prompt_line='- body_height: Adjust height. Params: {"height": <-0.15 to 0.1>}. -0.15=crouch, 0=normal, 0.1=tall.'),
        ActionSpec(
            name="set_speed", handler=_h.handle_set_speed, grammar_params=None,
            prompt_line='- set_speed: Set speed. Params: {"speed": "slow"|"normal"|"fast"}.'),
    ),
    intent_rules=(
        # === BODY HEIGHT ===
        _rule(3, r"\b(?:crouch|lower body|get low|duck)\b",    "body_height", {"height": -0.15}),
        _rule(4, r"\b(?:stand tall|raise body|stretch up|max height)\b", "body_height", {"height": 0.1}),
        _rule(5, r"\b(?:normal height|default height|regular height)\b", "body_height", {"height": 0.0}),

        # === WALKING WITH DISTANCE ===
        _rule(6, r"\b(?:walk|move|go)\s+forward\s+(?P<dist>\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)?\b", "walk", {"direction": "forward", "distance": "$dist"}),
        _rule(7, r"\b(?:walk|move|go)\s+(?:backward|back)\s+(?P<dist>\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)?\b", "walk", {"direction": "backward", "distance": "$dist"}),
        _rule(8, r"\b(?:walk|move|go)\s+forward\b",            "walk", {"direction": "forward", "distance": 1.0}),
        _rule(9, r"\b(?:walk|move|go)\s+(?:backward|back)\b",  "walk", {"direction": "backward", "distance": 1.0}),
        _rule(10, r"\bback\s*up\b",                             "walk", {"direction": "backward", "distance": 1.0}),

        # === STRAFING ===
        _rule(11, r"\b(?:strafe|move|step)\s+left\s+(?P<dist>\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)?\b", "strafe", {"direction": "left", "distance": "$dist"}),
        _rule(12, r"\b(?:strafe|move|step)\s+right\s+(?P<dist>\d+(?:\.\d+)?)\s*(?:m(?:eters?)?|ft|feet)?\b", "strafe", {"direction": "right", "distance": "$dist"}),
        _rule(13, r"\b(?:strafe|step)\s+left\b",                "strafe", {"direction": "left", "distance": 0.5}),
        _rule(14, r"\b(?:strafe|step)\s+right\b",               "strafe", {"direction": "right", "distance": 0.5}),

        # === TURNING ===
        _rule(15, r"\bturn\s+left\s+(?P<deg>\d+(?:\.\d+)?)\s*(?:deg(?:rees)?)?\b",  "turn", {"deg": "$deg", "dir": "left"}),
        _rule(16, r"\bturn\s+right\s+(?P<deg>\d+(?:\.\d+)?)\s*(?:deg(?:rees)?)?\b", "turn", {"deg": "$deg", "dir": "right"}),
        _rule(17, r"\bturn\s+left\b",                           "turn", {"deg": 90, "dir": "left"}),
        _rule(18, r"\bturn\s+right\b",                          "turn", {"deg": 90, "dir": "right"}),
        _rule(19, r"\b(?:turn around|spin around|about face|one eighty|180)\b", "turn", {"deg": 180, "dir": "left"}),

        # === SPEED CONTROL ===
        _rule(20, r"\b(?:slow\s+(?:down|mode)|walk\s+slowly|go\s+slow)\b", "set_speed", {"speed": "slow"}),
        _rule(21, r"\b(?:normal\s+speed|regular\s+speed|default\s+speed)\b", "set_speed", {"speed": "normal"}),
        _rule(22, r"\b(?:fast\s+(?:mode)?|walk\s+fast|go\s+fast|speed\s+up)\b", "set_speed", {"speed": "fast"}),
    ),
)
