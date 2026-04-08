#!/usr/bin/env python3
# scripts/estop_run.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import signal, time
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from src.session import quick_robot

_running = True
def _handle(sig, frame):
    global _running
    _running = False

def start_estop(claim=True):
    robot = quick_robot("dartmouth_spot_capstone_estop")
    estop_client: EstopClient = robot.ensure_client(EstopClient.default_service_name)

    # Check and report existing E-Stop holders before claiming
    if claim:
        try:
            status = estop_client.get_status()
            for ep_status in status.endpoints:
                ep = ep_status.endpoint
                print(f"[E-Stop] Found active endpoint: '{ep.name}' (role={ep.role})")
            if status.endpoints:
                print("[E-Stop] Claiming E-Stop from existing holders...")
        except Exception:
            pass

    endpoint = EstopEndpoint(estop_client, name="dartmouth_estop", estop_timeout=3.0)
    endpoint.force_simple_setup()  # claim: replace config and become sole endpoint
    keepalive = EstopKeepAlive(endpoint)
    keepalive.allow()
    print("[E-Stop] Claimed and active as 'dartmouth_estop'")
    return robot, estop_client, endpoint, keepalive

if __name__ == "__main__":
    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)

    print("Starting E-Stop keepalive. Press Ctrl+C to stop (will issue STOP).")
    print("This will claim the E-Stop even if another client holds it.\n")
    robot = estop_client = endpoint = keepalive = None

    robot, estop_client, endpoint, keepalive = start_estop(claim=True)

    try:
        while _running:
            try:
                keepalive.allow()  # refresh heartbeat
            except Exception:
                # Another client reclaimed the E-Stop - respect that and exit
                print("[E-Stop] Endpoint lost - another client has claimed the E-Stop.")
                print("[E-Stop] Exiting gracefully. Re-run this script to reclaim.")
                _running = False
            time.sleep(0.5)
    finally:
        try:
            if keepalive:
                keepalive.stop()
                keepalive.shutdown()
        except Exception:
            pass
        print("[E-Stop] Stopped and released.")