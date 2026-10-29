"""Shared pytest fixtures for Stage 2E.1 tests."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Ruling R24: no test can contact the robot (see tests/robot_guard.py).
# ---------------------------------------------------------------------------
import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _no_robot_contact():
    from tests.robot_guard import install
    uninstall = install()
    yield
    uninstall()


def pytest_sessionfinish(session, exitstatus):
    from tests.robot_guard import HITS
    if HITS:
        print(f"\n[robot_guard] {len(HITS)} blocked robot-contact attempt(s): {sorted(set(HITS))}")
        session.exitstatus = 1
