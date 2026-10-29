import logging

import pytest

from src.spot.core.events import EventBus


def test_publish_calls_in_order():
    bus, seen = EventBus(), []
    bus.subscribe("robot_stood", lambda **p: seen.append(("a", p)))
    bus.subscribe("robot_stood", lambda **p: seen.append(("b", p)))
    bus.publish("robot_stood", x=1)
    assert seen == [("a", {"x": 1}), ("b", {"x": 1})]


def test_handler_exception_isolated(caplog):
    bus, seen = EventBus(), []

    def bad(**p):
        raise RuntimeError("boom")

    bus.subscribe("safety_halt", bad)
    bus.subscribe("safety_halt", lambda **p: seen.append(1))
    with caplog.at_level(logging.ERROR):
        bus.publish("safety_halt")
    assert seen == [1]
    assert "[Events] handler for safety_halt failed: boom" in caplog.text


def test_unknown_event_raises():
    with pytest.raises(ValueError):
        EventBus().publish("nope")
