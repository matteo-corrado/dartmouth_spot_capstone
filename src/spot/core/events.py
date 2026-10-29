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


# The process bus: main sets it (step 2); None until then and in unit tests,
# so publish() below is a no-op there. Publishers read it at call time
# through this module (same owner pattern as src.spot.robot). A deliberate,
# documented exception to spec section 5: one slot instead of threading a
# bus reference through the TTS speaker, conversation, dispatcher and safety
# listener constructors (which the golden harness fakes by name).
BUS: EventBus | None = None


def publish(event: str, **payload) -> None:
    """Publish on BUS if main has set one. Never raises: every caller is a
    publisher that must not break (the safety listener calls it after the
    halt has already returned)."""
    bus = BUS
    if bus is None:
        return
    try:
        bus.publish(event, **payload)
    except Exception as exc:  # noqa: BLE001
        log.error("[Events] publish %s failed: %s", event, exc)
