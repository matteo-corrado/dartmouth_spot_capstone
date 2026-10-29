"""Spot settings feature: persona, speaking volume and TTS backend. None of
these needs a Spot session (state-only, unlocked, motion-free)."""
from src.spot.core.registry import ActionSpec, Feature
from src.spot.features.settings import handlers as _h

_FLAGS = dict(unlocked=True, state_only=True, motion_free=True, needs_session=False)

FEATURE = Feature(
    name="settings",
    actions=(
        ActionSpec(
            name="set_persona",
            handler=_h.set_persona_action,
            grammar_params=None,  # spot_action.gbnf has one generic params rule
            prompt_line='- set_persona: Switch the robot\'s personality. Params: {"name": "<persona_name>"}. Available personas: see KNOWN PERSONAS at the end of this prompt. Emit when the user says "be a pirate", "switch to butler", "act like a butler", "be yourself" (resets to default), etc. The persona controls speaking style AND voice. After emitting, your "response" field will be spoken in the NEW voice — write it in-character.',
            persona=True,
            **_FLAGS,
        ),
        ActionSpec(
            name="add_persona",
            handler=_h.add_persona_action,
            grammar_params=None,
            prompt_line=('- add_persona: Create a NEW persona on the fly when the user requests one NOT in your known persona list. Params: {"name": "<persona_name>", "description": "<5-10 word voice + style description>"}. Description should cover voice traits (gender, accent, tone) AND style. Examples:\n'
                '  • User: "be a cowboy" → if cowboy NOT in known personas → add_persona({"name":"cowboy", "description":"deep gravelly American male cowboy"})\n'
                '  • User: "talk like a French waitress" → add_persona({"name":"french_waitress", "description":"soft warm French female"})\n'
                '  Use set_persona for known personas (listed in the set_persona bullet above); add_persona only for new ones.'),
            persona=True,
            **_FLAGS,
        ),
        ActionSpec(
            name="set_volume",
            handler=_h.set_volume_action,
            grammar_params=None,
            prompt_line='- set_volume: Set your speaking volume. Params: {"level": <1-100>} as a percentage where 100=full, 50=half, 10=quiet. For "louder"/"quieter" without a number, pick a sensible delta from current (e.g. +20).',
            **_FLAGS,
        ),
        ActionSpec(
            name="set_backend",
            handler=_h.set_backend_action,
            grammar_params=None,
            prompt_line='- set_backend: Switch the TTS engine. Params: {"name": "kokoro"|"elevenlabs"}. "kokoro" is local/offline (lower quality), "elevenlabs" is cloud/premium. Emit for "switch to local voice", "go offline", "use kokoro", "switch to premium", "use elevenlabs".',  # llm_brain prefixes this line with {{EMOTE_ACTION}} (dance's)
            **_FLAGS,
        ),
    ),
    intent_rules=(),  # intent.py has no regex rule for these intents
)
