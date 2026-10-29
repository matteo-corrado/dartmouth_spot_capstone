"""Built-in Spot choreography routines (emotes and dances) for the voice brain.

Pure catalogue + name resolution. The dispatcher's `emote` action (wired in a
later lane) calls resolve_routine() and runs the routine via ChoreographyClient;
llm_brain includes prompt_section() so natural phrasings map to a routine name.
"""
import random

# name -> one-line description (keys are the robot's built-in routine names)
ROUTINES: dict[str, str] = {
    "hey_join_me": "an upbeat inviting dance",
    "performance_bow": "a theatrical bow",
    "smooth_sailing": "a smooth swaying dance",
    "happy_gesture": "a happy little wiggle",
    "tail_wag": "wags its rear like a tail",
    "upright_sit": "sits up nicely",
    "wanna_play": "a playful play-bow",
}

DANCES = ("hey_join_me", "performance_bow", "smooth_sailing")

# name -> example phrasings that should trigger it
_TRIGGERS: dict[str, tuple[str, ...]] = {
    "hey_join_me": ("join me", "let's dance together"),
    "performance_bow": ("take a bow", "do a trick", "bow for us"),
    "smooth_sailing": ("smooth sailing", "do a smooth dance"),
    "happy_gesture": ("are you happy", "show me you're happy"),
    "tail_wag": ("wag your tail", "are you a good dog"),
    "upright_sit": ("sit up nicely", "sit pretty"),
    "wanna_play": ("want to play", "let's play"),
}

_RANDOM_NAMES = (None, "dance", "random")


def resolve_routine(name: str | None, last: str | None, rng=random) -> str:
    """Map a requested routine name to a concrete ROUTINES key.

    Exact key -> that key. None/"dance"/"random" -> a random dance != last.
    Anything else raises ValueError.
    """
    if name in ROUTINES:
        return name
    if name in _RANDOM_NAMES:
        return rng.choice([d for d in DANCES if d != last])
    raise ValueError(f"unknown routine: {name!r}")


def prompt_section() -> str:
    """Brain prompt text listing every routine, its description and triggers."""
    lines = [
        'EMOTES: use action "emote" with params {"name": <routine>}.',
        'Use name "dance" (or "random") to pick a random dance. Triggers: '
        '"dance", "show off", "bust a move", "dance again".',
    ]
    for name, desc in ROUTINES.items():
        phrases = ", ".join(f'"{p}"' for p in _TRIGGERS[name])
        lines.append(f"- {name}: {desc}. Triggers: {phrases}")
    return "\n".join(lines)
