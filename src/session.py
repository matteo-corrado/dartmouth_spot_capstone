
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
from bosdyn.client.exceptions import RetryableRpcError
from bosdyn.client.keepalive import KeepaliveClient, remove_all_policies

from src.config import BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD


def quick_robot(client_name: str,
                hostname: str = BOSDYN_ROBOT_IP,
                username: str = BOSDYN_CLIENT_USERNAME,
                password: str = BOSDYN_CLIENT_PASSWORD,
                time_sync_attempts: int = 5,
                time_sync_backoff: float = 0.2):
    """Connect, authenticate, and time-sync a Spot ``Robot`` object.

    Lighter-weight alternative to ``spot_session()`` for callers that need
    a connected ``Robot`` but **not** the full lease/power-on/stand
    lifecycle. Used by E-Stop scripts and the web panel's camera streamer
    — they only need to talk to one or two clients (e.g. ``EstopClient``,
    ``ImageClient``) and would otherwise duplicate the same SDK boilerplate
    in each call site.

    Returns the ``Robot`` instance ready for ``ensure_client(...)`` calls.
    Does NOT take a lease, does NOT power on motors, and does NOT do any
    teardown — the caller owns the lifetime.

    Time-sync uses a small retry loop because Spot's TimeSyncClient can
    occasionally raise on the very first call right after reboot.
    """
    sdk = create_standard_sdk(client_name)
    robot = sdk.create_robot(hostname)
    robot.authenticate(username, password)

    ts = robot.ensure_client(TimeSyncClient.default_service_name)
    for _ in range(time_sync_attempts):
        try:
            ts.get_time_sync_update()
            break
        except Exception:
            time.sleep(time_sync_backoff)

    return robot


@contextmanager
def spot_session(hostname: str = BOSDYN_ROBOT_IP,
                 username: str = BOSDYN_CLIENT_USERNAME,
                 password: str = BOSDYN_CLIENT_PASSWORD,
                 stand_on_enter: bool = True,
                 sit_on_exit: bool = True,
                 upload_map: bool = False,
                 map_path: str = None,
                 auto_localize: bool = False):
    # ChoreographyClient must be registered at SDK creation for the emote action.
    try:
        from bosdyn.choreography.client.choreography import ChoreographyClient
        sdk = create_standard_sdk("dartmouth_spot_capstone", [ChoreographyClient])
    except ImportError:
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

    try:
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


# Arm poses (Task 14). "stowed" / "talk" / "walk".
#
# WALK_POSE picks the tucked pose used while the legs move. Operator session 2
# chooses it on the robot with scripts/try_walk_pose.py; switch it here:
#   "stow"   (a) arm_stow_command (claw closed first)  <- current default
#   "carry"  (b) arm_carry_command
#   "joints" (c) joint pose WALK_JOINTS via arm_joint_move_helper (slow, >= 2 s)
WALK_POSE = "stow"
WALK_JOINTS = (0.0, -2.6, 2.6, 0.0, 0.0, 0.0)  # sh0, sh1, el0, el1, wr0, wr1
ARM_MAX_LINEAR_VEL = 0.3   # m/s (gaze; named poses expose no limit in SDK 5.x)
ARM_MAX_ANGULAR_VEL = 1.0  # rad/s (joint-move max_vel for the "joints" pose)
JOINT_MOVE_SECS = 2.5      # >= 2 s trajectory for joint-space moves


def walk_pose_command(variant: str = None):
    """RobotCommand for one walking-pose candidate ("stow" | "carry" | "joints")."""
    from bosdyn.client.robot_command import RobotCommandBuilder
    variant = variant or WALK_POSE
    if variant == "stow":
        return RobotCommandBuilder.arm_stow_command()
    if variant == "carry":
        return RobotCommandBuilder.arm_carry_command()
    if variant == "joints":
        return RobotCommandBuilder.arm_joint_move_helper(
            joint_positions=[list(WALK_JOINTS)], times=[JOINT_MOVE_SECS],
            max_vel=ARM_MAX_ANGULAR_VEL)
    raise ValueError(f"unknown walk pose variant: {variant!r}")


def send_arm_pose(cmd_client, pose: str, walk_variant: str = None):
    """Send the command(s) for an arm pose without waiting. Returns the arm cmd id.

    "stowed" closes the gripper first, then stows. "talk" = ready pose.
    "walk" = the WALK_POSE candidate (or walk_variant). The walk "stow"
    candidate also closes the claw first, like "stowed".
    """
    from bosdyn.client.robot_command import RobotCommandBuilder
    if pose == "stowed" or (pose == "walk" and (walk_variant or WALK_POSE) == "stow"):
        cmd_client.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(0.0))
        return cmd_client.robot_command(RobotCommandBuilder.arm_stow_command())
    if pose == "talk":
        return cmd_client.robot_command(RobotCommandBuilder.arm_ready_command())
    if pose == "walk":
        return cmd_client.robot_command(walk_pose_command(walk_variant))
    raise ValueError(f"unknown arm pose: {pose!r}")


def move_arm_to(cmd_client, pose: str, timeout_s=5.0) -> bool:
    """Move the arm to "stowed" | "talk" | "walk". Returns True on arrival.

    timeout_s <= 0 sends the command and returns True without waiting (used on
    the stop/freeze/estop path so the stow never blocks the caller).
    """
    from bosdyn.client.robot_command import block_until_arm_arrives
    cmd_id = send_arm_pose(cmd_client, pose)
    if timeout_s is None or timeout_s <= 0:
        return True
    return bool(block_until_arm_arrives(cmd_client, cmd_id, timeout_s))


def deploy_arm_safe(cmd_client, timeout_sec: float = 5.0):
    """Deploy the arm to the ready ("talk") pose. Caller must have motors
    powered and the robot standing. Returns True on arrival."""
    return move_arm_to(cmd_client, "talk", timeout_sec)


def stow_arm(cmd_client, timeout_sec: float = 5.0):
    """Close the gripper then stow the arm. Safe to call when already stowed."""
    return move_arm_to(cmd_client, "stowed", timeout_sec)
