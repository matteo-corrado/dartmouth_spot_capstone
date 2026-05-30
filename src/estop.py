# src/estop.py
import time
from contextlib import contextmanager

from bosdyn.client import create_standard_sdk
from bosdyn.client.estop import (
    EstopClient,
    EstopEndpoint,
    EstopKeepAlive,
    MotorsOnError,
)
from bosdyn.client.exceptions import RetryableRpcError, UnimplementedError
from bosdyn.client.lease import LeaseClient
from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD


def _safe_power_off(robot) -> None:
    """Sit Spot and de-energize motors via force-taken lease.

    Required when ``force_simple_setup`` hits ``MotorsOnError``: BD's
    ``SetEstopConfig`` refuses motors-on, and the public SDK has no
    runtime takeover mechanism. Tablet bypasses this with a pre-paired
    persistent endpoint — third-party clients have no equivalent.

    ``lease_client.take()`` force-takes the lease regardless of holder.
    ``robot.power_off(cut_immediately=False)`` issues SafePowerOff: Spot
    sits gracefully then de-energizes motors. Lease is then returned so
    voice control's child can acquire its own.
    """
    print("[E-Stop] Motors on — taking lease and SafePowerOff to recover...")
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease = lease_client.take()
    try:
        robot.power_off(cut_immediately=False, timeout_sec=20)
        print("[E-Stop] Motors de-energized.")
    finally:
        try:
            lease_client.return_lease(lease)
        except Exception as e:
            print(f"[E-Stop] WARN: lease return failed ({e}); continuing.")


def claim_estop(robot, estop_client: EstopClient, name: str,
                timeout_sec: int) -> EstopEndpoint:
    """Claim the E-Stop. On ``MotorsOnError``, SafePowerOff then retry.

    ``force_simple_setup`` is the only public-SDK way to install a new
    endpoint — and it requires motors off. On ``MotorsOnError`` we
    self-recover: force-take lease, SafePowerOff (Spot sits + motors
    de-energize), then retry the setup.
    """
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

    endpoint = EstopEndpoint(estop_client, name=name, estop_timeout=timeout_sec)
    try:
        endpoint.force_simple_setup()
    except MotorsOnError:
        _safe_power_off(robot)
        endpoint.force_simple_setup()
    print(f"[E-Stop] Claimed successfully as '{name}'.")
    return endpoint


@contextmanager
def estop_session(hostname: str = BOSDYN_ROBOT_IP,
                  username: str = BOSDYN_CLIENT_USERNAME,
                  password: str = BOSDYN_CLIENT_PASSWORD,
                  name: str = "dartmouth_estop",
                  timeout_sec: int = 3,
                  cut_on_exit: bool = True,
                  auth_timeout_sec: float = 90.0,
                  auth_retry_interval: float = 3.0):
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

    # Cold-boot race: if wakespot launches while Spot is still powering on,
    # authenticate() throws transient transport errors — ProxyConnectionError
    # before the robot proxy answers, UnimplementedError while the auth service
    # is still registering behind it. Retry until the robot finishes booting.
    # Credential rejections (InvalidLoginError, TemporarilyLockedOutError) are
    # ResponseError, not RpcError, so they propagate immediately — no pointless
    # retry and no risk of triggering an account lockout.
    auth_deadline = time.monotonic() + auth_timeout_sec
    attempt = 0
    while True:
        try:
            robot.authenticate(username, password)
            break
        except (RetryableRpcError, UnimplementedError) as e:
            attempt += 1
            remaining = auth_deadline - time.monotonic()
            if remaining <= 0:
                raise
            wait = min(auth_retry_interval, remaining)
            print(f"[E-Stop] Robot not ready yet ({type(e).__name__}, "
                  f"attempt {attempt}); retrying in {wait:.0f}s "
                  f"(~{remaining:.0f}s before giving up)...")
            time.sleep(wait)

    # Time sync required before any robot_command (power_off recovery path).
    # wait_for_sync() defaults to a 3s budget and RAISES on expiry; on a cold
    # boot the sync service may still be settling right after auth succeeds, so
    # share the remaining boot-wait budget instead of crashing here.
    robot.time_sync.wait_for_sync(timeout_sec=max(1.0, auth_deadline - time.monotonic()))

    estop_client: EstopClient = robot.ensure_client(EstopClient.default_service_name)
    endpoint = claim_estop(robot, estop_client, name, timeout_sec)

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
