# src/session.py
from contextlib import contextmanager
import time

from bosdyn.client import create_standard_sdk
from bosdyn.client.auth import AuthResponseError
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.power import PowerClient
from bosdyn.client.time_sync import TimeSyncClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, blocking_sit
from bosdyn.client.robot_command import RobotCommandBuilder
from bosdyn.client.exceptions import RetryableRpcError
from bosdyn.client.keepalive import KeepaliveClient, remove_all_policies

from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD


@contextmanager
def spot_session(hostname: str = BOSDYN_ROBOT_IP,
                 username: str = BOSDYN_CLIENT_USERNAME,
                 password: str = BOSDYN_CLIENT_PASSWORD,
                 stand_on_enter: bool = True,
                 sit_on_exit: bool = True,
                 upload_map: bool = False,
                 map_path: str = None,
                 auto_localize: bool = False):
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

    # Clear stale keepalive policies BEFORE lease acquisition
    def _clear_keepalive_policies():
        try:
            keepalive_client = robot.ensure_client(KeepaliveClient.default_service_name)
            status = keepalive_client.get_status()
            if status.status:
                print(f"[Session] Clearing {len(status.status)} stale keepalive policy(ies)...")
                remove_all_policies(keepalive_client)
                time.sleep(2)
                print("[Session] ✓ Keepalive policies cleared")
        except Exception as e:
            print(f"[Session] Note: Could not clear keepalive policies: {e}")

    _clear_keepalive_policies()

    # Lease - force take to override any stale claims
    print("[Session] Taking lease forcefully...")
    lease = lease_client.take()
    lease_keepalive = LeaseKeepAlive(lease_client, must_acquire=False, return_at_exit=True)

    # Power on and stand (optional)
    if stand_on_enter:

        # Check if robot is already powered before trying to power on
        robot_state = state_client.get_robot_state()
        is_powered = robot_state.power_state.motor_power_state == robot_state.power_state.STATE_ON

        if not is_powered:
            try:
                robot.power_on(timeout_sec=20)
            except Exception as e:
                error_type = type(e).__name__
                if "KeepaliveMotorsOff" in error_type or "KeepaliveMotorsOffError" in str(e):
                    # Retry once: clear policies again and try power on
                    print("[Session] Keepalive still blocking — clearing policies again and retrying...")
                    _clear_keepalive_policies()
                    try:
                        robot.power_on(timeout_sec=20)
                    except Exception:
                        print("[Session] Warning: Cannot power on - keepalive still blocking motors.")
                        print("[Session] Try releasing e-stop from tablet, then retry.")
                        raise
                elif "Estopped" in error_type or "EstoppedError" in str(e):
                    print("[Session] Warning: Robot is e-stopped.")
                    print("[Session] Release e-stop from the tablet, then retry.")
                    raise
                else:
                    raise
        
        # Robot is powered (or we powered it on), now stand
        try:
            blocking_stand(cmd_client, timeout_sec=20)
        except Exception as e:
            print(f"[Session] Warning: Stand command failed: {e}")
            # Don't raise - robot might already be standing or estop might be blocking
            # Continue with session anyway

    # Optional: Upload map and localize
    if upload_map and map_path:
        try:
            from src.graph_nav_utils import upload_graph_and_snapshots, initialize_localization
            print("[Session] Uploading map...")
            if upload_graph_and_snapshots(robot, map_path):
                if auto_localize:
                    print("[Session] Auto-localizing...")
                    initialize_localization(robot, use_fiducial=True)
            else:
                print("[Session] Warning: Map upload failed")
        except Exception as e:
            print(f"[Session] Warning: Map upload/localization error: {e}")
            # Don't fail the session if map upload fails

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