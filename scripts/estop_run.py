# scripts/estop_run.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import signal, time
from bosdyn.client import create_standard_sdk
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client.time_sync import TimeSyncClient
from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD

_running = True
def _handle(sig, frame):
    global _running
    _running = False

def start_estop():
    sdk = create_standard_sdk("dartmouth_spot_capstone_estop")
    robot = sdk.create_robot(BOSDYN_ROBOT_IP)
    robot.authenticate(BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD)

    # Best-effort time sync
    ts = robot.ensure_client(TimeSyncClient.default_service_name)
    for _ in range(5):
        try:
            ts.get_time_sync_update()
            break
        except Exception:
            time.sleep(0.2)

    estop_client: EstopClient = robot.ensure_client(EstopClient.default_service_name)
    endpoint = EstopEndpoint(estop_client, name="dartmouth_estop", estop_timeout=3.0)
    endpoint.force_simple_setup()  # become the active endpoint
    keepalive = EstopKeepAlive(endpoint)
    keepalive.allow()
    return robot, endpoint, keepalive

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print("Starting E-Stop keepalive. Press Ctrl+C to stop (will issue STOP).")
    robot = endpoint = keepalive = None

    robot, endpoint, keepalive = start_estop()

    try:
        while _running:
            try:
                keepalive.allow()  # refresh heartbeat
            except Exception:
                # If endpoint becomes unknown/cleared, re-register
                try:
                    if keepalive: keepalive.shutdown()
                except Exception:
                    pass
                print("E-Stop endpoint issue detected. Re-registering...")
                robot, endpoint, keepalive = start_estop()
            time.sleep(0.5)
    finally:
        try:
            if keepalive:
                keepalive.stop()
                keepalive.shutdown()
        except Exception:
            pass
        print("E-Stop stopped and released.")