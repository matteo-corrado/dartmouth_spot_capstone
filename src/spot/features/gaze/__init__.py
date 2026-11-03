"""Spot gaze feature: turn the folded arm toward the locked speaker.

No actions (no intent), no start hook and no bus subscriptions. The gaze
worker thread is started lazily by the conversation loop (Task 10), and the
loop calls gaze directly (_gaze_at_speaker hands off to that one worker, so
the arm RPC never runs on the main/audio loop: safety condition C5). The
feature's registration is what REGISTRY.has("gaze") checks.
"""
from src.spot.core.registry import Feature

FEATURE = Feature(
    name="gaze",
    actions=(),
)
