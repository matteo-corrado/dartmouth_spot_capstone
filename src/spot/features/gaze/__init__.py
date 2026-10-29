"""Spot gaze feature: turn the folded arm toward the locked speaker.

No actions (no intent). No start hook: the gaze worker thread is started
lazily by the conversation loop (Task 10), not by spot_dispatch. The
speaker_locked subscription is declared here but main does NOT subscribe it
(Task 15, safety condition C5): on_speaker_locked runs the arm RPC on the
caller's thread, and the bus would call it on the main/audio loop. The loop
keeps its direct _gaze_at_speaker call, which hands off to the gaze worker.
"""
from src.spot.core.registry import Feature
from src.spot.features.gaze.handlers import on_speaker_locked

FEATURE = Feature(
    name="gaze",
    actions=(),
    subscriptions=(("speaker_locked", on_speaker_locked),),
)
