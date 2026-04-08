# src/estop.py
import time
from contextlib import contextmanager

from bosdyn.client import create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.time_sync import TimeSyncClient
from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD


def claim_estop(estop_client: EstopClient, name: str, timeout_sec: int) -> EstopEndpoint:
    """Claim the E-Stop by deregistering existing endpoints and registering ours.

    Uses force_simple_setup to replace the entire E-Stop configuration,
    taking over from any other client (tablet, other scripts, etc.).
    """
    # Check who currently holds the E-Stop
    try:
        status = estop_client.get_status()
        active_endpoints = status.endpoints
        if active_endpoints:
            names = [ep.endpoint.name for ep in active_endpoints]
            print(f"[E-Stop] Active endpoints found: {names}")
            print(f"[E-Stop] Claiming E-Stop from existing holders...")
        else:
            print("[E-Stop] No active endpoints, registering fresh.")
    except Exception as e:
        print(f"[E-Stop] Could not query status ({e}), proceeding with claim...")

    # force_simple_setup replaces the entire config with just our endpoint
    endpoint = EstopEndpoint(estop_client, name=name, estop_timeout=timeout_sec)
    endpoint.force_simple_setup()
    print(f"[E-Stop] Claimed successfully as '{name}'")
    return endpoint


@contextmanager
def estop_session(hostname: str = BOSDYN_ROBOT_IP,
                  username: str = BOSDYN_CLIENT_USERNAME,
                  password: str = BOSDYN_CLIENT_PASSWORD,
                  name: str = "dartmouth_estop",
                  timeout_sec: int = 3,
                  cut_on_exit: bool = True):
    """Claim Spot's E-Stop and yield the live keepalive.

    Args:
        cut_on_exit: When True (default), the context manager issues
            ``keepalive.stop()`` (the E-Stop CUT) on exit before releasing
            the endpoint. Safe default for callers that just want a hard
            stop on teardown. Pass ``False`` when the caller drives its own
            shutdown sequence (e.g. wakespot's graceful sit/power-off path)
            and wants to release the endpoint WITHOUT cutting motors —
            cutting after a clean ``power_off`` is harmless but conceptually
            contradicts a "graceful" shutdown. The caller is then
            responsible for calling ``keepalive.stop()`` itself when an
            emergency cut is actually wanted.
    """
    sdk = create_standard_sdk("dartmouth_spot_capstone_estop")
    robot = sdk.create_robot(hostname)
    robot.authenticate(username, password)

    # Time sync recommended before E-Stop registration
    ts_client = robot.ensure_client(TimeSyncClient.default_service_name)
    for _ in range(5):
        try:
            ts_client.get_time_sync_update()
            break
        except Exception:
            time.sleep(0.2)

    estop_client: EstopClient = robot.ensure_client(EstopClient.default_service_name)
    endpoint = claim_estop(estop_client, name, timeout_sec)

    keepalive = EstopKeepAlive(endpoint)
    keepalive.allow()  # ALLOW = not stopping robot

    try:
        yield {"robot": robot, "estop_client": estop_client, "endpoint": endpoint, "keepalive": keepalive}
    finally:
        # Optionally return to safe STOP state, then release the endpoint.
        if cut_on_exit:
            try:
                keepalive.stop()  # issue STOP before exiting
            except Exception:
                pass
        try:
            keepalive.shutdown()
        except Exception:
            pass