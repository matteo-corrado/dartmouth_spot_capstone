"""Tests can never contact the robot (Ruling R24).

install() wraps every real robot-contact entry point so it raises
RobotContactBlocked the moment its real body would run. Tests that fake the
session (CONNECTION._session set, or a monkeypatched spot_session) never reach
these, and a per-test monkeypatch simply replaces the guard for that test.

RobotContactBlocked is a BaseException so no `except Exception` catch-all in
the dispatch code can swallow it; every hit is also recorded in HITS, and
tests/conftest.py fails the session if any hit was recorded.

Used by tests/conftest.py (pytest process) and by the golden harness child
(tests/integration/harness.py), which does not load conftest.
"""
HITS: list[str] = []


class RobotContactBlocked(BaseException):
    pass


def _blocker(what):
    def blocked(*args, **kwargs):
        HITS.append(what)
        raise RobotContactBlocked(f"test tried to contact the robot: {what}")
    blocked.__robot_guard__ = what
    return blocked


def install():
    """Wrap the entry points; returns an undo callable. Idempotent."""
    import bosdyn.client.channel as channel
    import bosdyn.client.robot as robot
    import bosdyn.client.sdk as sdk
    import src.spot.robot.connection as connection
    import src.spot.robot.session as session

    targets = [
        (sdk.Sdk, "create_robot", "bosdyn Sdk.create_robot"),
        (robot.Robot, "authenticate", "bosdyn Robot.authenticate"),
        (channel, "create_secure_channel", "bosdyn channel.create_secure_channel"),
        (session, "spot_session", "src.spot.robot.session.spot_session"),
        (session, "quick_robot", "src.spot.robot.session.quick_robot"),
        (connection, "spot_session", "Connection.ensure bring-up (connection.spot_session)"),
    ]
    undo = []
    for owner, attr, what in targets:
        orig = owner.__dict__[attr] if isinstance(owner, type) else getattr(owner, attr)
        if getattr(orig, "__robot_guard__", None):
            continue
        setattr(owner, attr, _blocker(what))
        undo.append((owner, attr, orig))

    def uninstall():
        for owner, attr, orig in reversed(undo):
            setattr(owner, attr, orig)
    return uninstall


GUARDED = (
    "bosdyn Sdk.create_robot",
    "bosdyn Robot.authenticate",
    "bosdyn channel.create_secure_channel",
    "src.spot.robot.session.spot_session",
    "src.spot.robot.session.quick_robot",
    "Connection.ensure bring-up (connection.spot_session)",
)
