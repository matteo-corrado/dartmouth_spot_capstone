"""Ruling R24: the robot-contact guard fires on every real entry point."""
import pytest

import tests.robot_guard as guard
from src.spot.robot import CONNECTION


@pytest.fixture
def expect_hits():
    n = len(guard.HITS)
    yield
    del guard.HITS[n:]  # these hits are the point of the test, not a leak


def test_guard_fires_on_every_entry_point(expect_hits, monkeypatch):
    import bosdyn.client.channel as channel
    import bosdyn.client.robot as robot
    import bosdyn.client.sdk as sdk
    import src.spot.robot.session as session

    monkeypatch.setattr(CONNECTION, "_session", None)
    monkeypatch.setattr(CONNECTION, "_context", None)
    monkeypatch.setattr(CONNECTION, "_closing", False)
    calls = [
        lambda: CONNECTION.ensure(),
        lambda: session.spot_session(),
        lambda: session.quick_robot("guard-test"),
        lambda: sdk.create_standard_sdk("guard-test").create_robot("192.0.2.1"),
        lambda: robot.Robot.authenticate(object(), "u", "p"),
        lambda: channel.create_secure_channel("192.0.2.1", None, None),
    ]
    for call in calls:
        with pytest.raises(guard.RobotContactBlocked, match="test tried to contact the robot"):
            call()
    assert CONNECTION._session is None
    assert len(set(guard.HITS)) == len(guard.GUARDED)


def test_guard_not_swallowed_by_dispatch_catch_all(expect_hits, monkeypatch):
    """A registered intent with no fake session reaches the real ensure():
    the block escapes dispatch's `except Exception` instead of looking like a
    normal failure."""
    import src.voice_control.spot_dispatch as d
    monkeypatch.setattr(CONNECTION, "_session", None)
    monkeypatch.setattr(CONNECTION, "_context", None)
    monkeypatch.setattr(CONNECTION, "_closing", False)
    with pytest.raises(guard.RobotContactBlocked):
        d.dispatch_intent({"intent": "set_speed", "params": {}})
