"""Synchronous in-process event bus."""
import logging
from typing import Callable

log = logging.getLogger(__name__)

EVENTS: frozenset[str] = frozenset({
    "speaking_started", "speaking_stopped", "voice_state_changed",
    "speaker_locked", "speaker_released", "robot_stood", "robot_sat",
    "safety_halt", "person_seen",
})


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Callable[..., None]]] = {}

    def subscribe(self, event: str, handler: Callable[..., None]) -> None:
        self._subs.setdefault(event, []).append(handler)

    def publish(self, event: str, **payload) -> None:
        if event not in EVENTS:
            raise ValueError(f"unknown event: {event}")
        for handler in list(self._subs.get(event, ())):
            try:
                handler(**payload)
            except Exception as exc:
                log.error("[Events] handler for %s failed: %s", event, exc)
