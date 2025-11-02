# src/estop.py
import time
from contextlib import contextmanager

from bosdyn.client import create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.time_sync import TimeSyncClient
from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD


@contextmanager
def estop_session(hostname: str = BOSDYN_ROBOT_IP,
                  username: str = BOSDYN_CLIENT_USERNAME,
                  password: str = BOSDYN_CLIENT_PASSWORD,
                  name: str = "dartmouth_estop",
                  timeout_sec: int = 3):
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
    endpoint = EstopEndpoint(estop_client, name=name, estop_timeout=timeout_sec)
    endpoint.force_simple_setup()  # become sole E-Stop

    keepalive = EstopKeepAlive(endpoint)
    keepalive.allow()  # ALLOW = not stopping robot

    try:
        yield {"robot": robot, "estop_client": estop_client, "endpoint": endpoint, "keepalive": keepalive}
    finally:
        # Return to safe STOP state and release
        try:
            keepalive.stop()  # issue STOP before exiting
        except Exception:
            pass
        try:
            keepalive.shutdown()
        except Exception:
            pass