"""SafetyListener (Task 10): the halt is a direct call, never the event bus."""
import threading

import src.spot.conversation.loop as _loop
import src.spot.safety.listener as _sl
from src.spot.core.events import EventBus


def test_halt_does_not_touch_bus(monkeypatch):
    """Review Focus 4: a keyword halt never goes through the event bus, so a
    failing publish can't delay or block it (and on_keyword never raises)."""
    order = []
    hushed = threading.Event()

    def no_bus(self, event, **payload):
        order.append(("publish", event))
        raise AssertionError(f"halt published {event}")

    def halt(intent):
        order.append(("halt", intent["intent"]))
        return True

    def hush():
        order.append(("hush",))
        hushed.set()

    monkeypatch.setattr(EventBus, "publish", no_bus)
    monkeypatch.setattr(_loop.CONVERSATION, "_abort_chain", threading.Event())
    listener = _sl.SafetyListener(halt=halt, hush=hush)
    listener.on_keyword("stop")
    assert hushed.wait(1.0)
    assert [o for o in order if o[0] == "halt"] == [("halt", "stop")]
    assert not [o for o in order if o[0] == "publish"]
    assert _loop.CONVERSATION._abort_chain.is_set()
    assert listener._last_safety_at > float("-inf")


def test_halt_sent_even_if_hush_thread_cannot_start(monkeypatch):
    sent = []

    class Boom:
        def __init__(self, *a, **k):
            pass

        def start(self):
            raise RuntimeError("can't start new thread")

    monkeypatch.setattr(_loop.CONVERSATION, "_abort_chain", threading.Event())
    monkeypatch.setattr(_sl.threading, "Thread", Boom)
    listener = _sl.SafetyListener(halt=lambda i: sent.append(i["intent"]) or True, hush=lambda: None)
    listener.on_keyword("freeze")
    assert sent == ["freeze"]


def test_singleton_halt_and_hush_are_late_bound(monkeypatch):
    """LISTENER's halt/hush look up _dispatch_safety/_hush_player at call time,
    so a patch on the module reaches the KWS path (tests + golden harness)."""
    seen, hushed = [], threading.Event()
    monkeypatch.setattr(_loop.CONVERSATION, "_abort_chain", threading.Event())
    monkeypatch.setattr(_sl, "_dispatch_safety", lambda intent: seen.append(intent["intent"]) or True)
    monkeypatch.setattr(_sl, "_hush_player", hushed.set)
    _sl.LISTENER.on_keyword("estop")
    assert seen == ["estop"]
    assert hushed.wait(1.0)


def test_unmapped_label_is_ignored(monkeypatch):
    sent = []
    monkeypatch.setattr(_sl, "check_safety_command", lambda t: None)
    listener = _sl.SafetyListener(halt=lambda i: sent.append(i) or True, hush=lambda: None)
    listener.on_keyword("banana")
    assert sent == []
