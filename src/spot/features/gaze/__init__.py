"""Spot gaze feature: turn the folded arm toward the locked speaker.

No actions (no intent). No start hook: the gaze worker thread is started
lazily by client_mic (Task 10), not by spot_dispatch. The speaker_locked
subscription is declared here and wired to the bus in Task 15.
"""
from src.spot.core.registry import Feature
from src.spot.features.gaze.handlers import on_speaker_locked

FEATURE = Feature(
    name="gaze",
    actions=(),
    subscriptions=(("speaker_locked", on_speaker_locked),),
)
