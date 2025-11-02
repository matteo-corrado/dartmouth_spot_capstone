# src/session.py
from contextlib import contextmanager
import time

from bosdyn.client import create_standard_sdk
from bosdyn.client.auth import AuthResponseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.power import PowerClient
from bosdyn.client.time_sync import TimeSyncClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, blocking_sit
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.exceptions import RetryableRpcError

from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD


@contextmanager
def spot_session(hostname: str = BOSDYN_ROBOT_IP,
                 username: str = BOSDYN_CLIENT_USERNAME,
                 password: str = BOSDYN_CLIENT_PASSWORD,
                 stand_on_enter: bool = True,
                 sit_on_exit: bool = True):
    sdk = create_standard_sdk("dartmouth_spot_capstone")
    robot = sdk.create_robot(hostname)

    # Auth
    robot.authenticate(username, password)

    # Time sync
    robot.time_sync.wait_for_sync()

    # Clients
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
    power_client = robot.ensure_client(PowerClient.default_service_name)
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    # Lease with keepalive
    lease = lease_client.acquire()
    lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True)

    # Power on and stand (optional)
    if stand_on_enter:
        robot.power_on(timeout_sec=20)
        blocking_stand(cmd_client, timeout_sec=20)

    try:
        yield {
            "robot": robot,
            "lease_client": lease_client,
            "cmd": cmd_client,
            "power": power_client,
            "state": state_client,
        }
    finally:
        # Optional sit then power off
        if sit_on_exit:
            try:
                blocking_sit(cmd_client, timeout_sec=15)
            except Exception:
                pass
        try:
            robot.power_off(cut_immediately=False, timeout_sec=20)
        except Exception:
            pass
        lease_keepalive.shutdown()