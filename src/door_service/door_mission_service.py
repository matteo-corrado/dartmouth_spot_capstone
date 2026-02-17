"""Remote Mission Service for door opening during Spot AutoWalks.

This gRPC server runs on the Jetson and is called by Spot at configured
waypoints during AutoWalk playback. It handles two door types:
  1. Push-button doors (press handicap button, wait, walk through)
  2. Handle-based doors (grasp handle, push/pull, walk through)

Usage:
    1. Start this service on the Jetson
    2. On tablet: Record AutoWalk → at a door → "+" → "Create New Action" → "Remote GRPC"
    3. Select this service, configure parameters (button vs handle, push vs pull)
    4. On playback, Spot calls this service at the door waypoint

BD Support note: Handle-based door opening is complex. Start with push-button
doors, then iterate on handle-based opening with pre-configured positions.
"""

import logging
import time
import threading
from enum import Enum, auto
from typing import Optional

from bosdyn.api import robot_command_pb2
from bosdyn.api.mission import remote_pb2, remote_service_pb2_grpc
from bosdyn.api import arm_command_pb2, synchronized_command_pb2
from bosdyn.api import geometry_pb2
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME
from bosdyn.client import math_helpers

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Door service parameters (configurable from tablet)
# ---------------------------------------------------------------------------
DEFAULT_PARAMS = {
    "door_type": "button",       # "button" or "handle"
    "door_direction": "push",    # "push" or "pull" (for handle doors)
    "handle_height_m": 1.0,      # height of door handle from ground
    "button_height_m": 1.0,      # height of push button from ground
    "handle_offset_y_m": 0.0,    # lateral offset of handle from center
    "button_offset_y_m": -0.3,   # lateral offset of button (usually to the side)
    "door_width_m": 0.9,         # width of doorway
    "wait_time_s": 5.0,          # seconds to wait for door to open (button mode)
    "approach_distance_m": 0.8,  # distance to stand from the door
    "walk_through_distance_m": 2.0,  # distance to walk through doorway
}


# ---------------------------------------------------------------------------
# State machines
# ---------------------------------------------------------------------------
class ButtonState(Enum):
    """State machine for push-button door opening.

    Sequence: press button → walk through with arm holding door → stow arm after.
    """
    IDLE = auto()
    UNSTOW_ARM = auto()
    POSITION_ARM = auto()
    PRESS_BUTTON = auto()
    WALK_THROUGH = auto()
    STOW_ARM = auto()
    DONE = auto()
    FAILED = auto()


class HandleState(Enum):
    """State machine for handle-based door opening."""
    IDLE = auto()
    UNSTOW_ARM = auto()
    POSITION_ARM = auto()
    OPEN_GRIPPER = auto()
    GRASP_HANDLE = auto()
    CLOSE_GRIPPER = auto()
    PUSH_OR_PULL = auto()
    HOLD_OPEN = auto()
    WALK_THROUGH = auto()
    RELEASE_HANDLE = auto()
    STOW_ARM = auto()
    DONE = auto()
    FAILED = auto()


class DoorOpeningServicer(remote_service_pb2_grpc.RemoteMissionServiceServicer):
    """gRPC servicer that handles door opening for Spot AutoWalks.

    Implements the Remote Mission Service interface:
    - EstablishSession: acquire leases, parse parameters
    - Tick: execute door-opening state machine
    - Stop: halt safely
    - TeardownSession: release leases
    - GetRemoteMissionServiceInfo: describe service parameters
    """

    def __init__(self, robot):
        """Initialize with a connected Spot robot client.

        Args:
            robot: Authenticated bosdyn.client.Robot instance.
        """
        self.robot = robot
        self._lock = threading.Lock()

        # Session state
        self._lease_client = None
        self._lease_keepalive = None
        self._command_client = None
        self._params = dict(DEFAULT_PARAMS)

        # State machine
        self._button_state = ButtonState.IDLE
        self._handle_state = HandleState.IDLE
        self._state_start_time = None
        self._command_id = None

    # -----------------------------------------------------------------------
    # gRPC: EstablishSession
    # -----------------------------------------------------------------------
    def EstablishSession(self, request, context):
        """Called when the mission starts this action node."""
        logger.info("EstablishSession called")
        response = remote_pb2.EstablishSessionResponse()

        try:
            # Parse parameters from mission inputs
            self._parse_inputs(request.inputs)

            # Acquire lease for body + arm
            self._lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
            self._command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)

            # Take lease (sublease from the mission)
            lease = self._lease_client.take()
            self._lease_keepalive = LeaseKeepAlive(self._lease_client)

            # Reset state machine
            if self._params["door_type"] == "button":
                self._button_state = ButtonState.UNSTOW_ARM
            else:
                self._handle_state = HandleState.UNSTOW_ARM
            self._state_start_time = time.time()

            response.status = remote_pb2.EstablishSessionResponse.STATUS_OK
            logger.info(f"Session established — door_type={self._params['door_type']}")

        except Exception as e:
            logger.error(f"EstablishSession failed: {e}")
            response.status = remote_pb2.EstablishSessionResponse.STATUS_ERROR
            if hasattr(response, 'error_message'):
                response.error_message = str(e)

        return response

    # -----------------------------------------------------------------------
    # gRPC: Tick
    # -----------------------------------------------------------------------
    def Tick(self, request, context):
        """Called repeatedly by the mission to advance the state machine."""
        response = remote_pb2.TickResponse()

        try:
            with self._lock:
                if self._params["door_type"] == "button":
                    status = self._tick_button()
                else:
                    status = self._tick_handle()

            response.status = status

        except Exception as e:
            logger.error(f"Tick error: {e}")
            response.status = remote_pb2.TickResponse.STATUS_FAILURE

        return response

    # -----------------------------------------------------------------------
    # gRPC: Stop
    # -----------------------------------------------------------------------
    def Stop(self, request, context):
        """Called to safely halt the action."""
        logger.info("Stop called — halting door operation")
        response = remote_pb2.StopResponse()

        try:
            if self._command_client:
                self._command_client.robot_command(RobotCommandBuilder.stop_command())
            # Attempt to stow arm
            self._stow_arm()
        except Exception as e:
            logger.error(f"Stop error: {e}")

        response.status = remote_pb2.StopResponse.STATUS_OK
        return response

    # -----------------------------------------------------------------------
    # gRPC: TeardownSession
    # -----------------------------------------------------------------------
    def TeardownSession(self, request, context):
        """Called when the mission finishes with this action node."""
        logger.info("TeardownSession called")
        response = remote_pb2.TeardownSessionResponse()

        try:
            # Stow arm
            self._stow_arm()

            # Release lease
            if self._lease_keepalive:
                self._lease_keepalive.shutdown()
                self._lease_keepalive = None

            # Reset state
            self._button_state = ButtonState.IDLE
            self._handle_state = HandleState.IDLE

        except Exception as e:
            logger.error(f"TeardownSession error: {e}")

        response.status = remote_pb2.TeardownSessionResponse.STATUS_OK
        return response

    # -----------------------------------------------------------------------
    # gRPC: GetRemoteMissionServiceInfo
    # -----------------------------------------------------------------------
    def GetRemoteMissionServiceInfo(self, request, context):
        """Describe this service and its configurable parameters."""
        response = remote_pb2.GetRemoteMissionServiceInfoResponse()

        # Define custom parameters that can be set from the tablet
        # These show up in the AutoWalk recording UI
        custom_params = response.custom_params

        # Door type: "button" or "handle"
        door_type_spec = custom_params.specs["door_type"]
        door_type_spec.spec.default_value.string_value.value = "button"

        # Door direction: "push" or "pull" (for handle mode)
        direction_spec = custom_params.specs["door_direction"]
        direction_spec.spec.default_value.string_value.value = "push"

        # Button height
        button_height_spec = custom_params.specs["button_height_m"]
        button_height_spec.spec.default_value.double_value.value = 1.0

        # Handle height
        handle_height_spec = custom_params.specs["handle_height_m"]
        handle_height_spec.spec.default_value.double_value.value = 1.0

        # Wait time (for button doors)
        wait_spec = custom_params.specs["wait_time_s"]
        wait_spec.spec.default_value.double_value.value = 5.0

        # Walk-through distance
        walk_spec = custom_params.specs["walk_through_distance_m"]
        walk_spec.spec.default_value.double_value.value = 2.0

        return response

    # -----------------------------------------------------------------------
    # Push-button state machine
    # -----------------------------------------------------------------------
    def _tick_button(self) -> int:
        """Advance the push-button door state machine.

        Returns TickResponse status enum value.
        """
        state = self._button_state

        if state == ButtonState.IDLE:
            return remote_pb2.TickResponse.STATUS_SUCCESS

        elif state == ButtonState.UNSTOW_ARM:
            logger.info("[Button] Unstowing arm...")
            cmd = RobotCommandBuilder.arm_ready_command()
            self._command_id = self._command_client.robot_command(cmd)
            self._button_state = ButtonState.POSITION_ARM
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == ButtonState.POSITION_ARM:
            # Wait for arm ready, then position to button
            if time.time() - self._state_start_time < 2.0:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Button] Positioning arm to button...")
            button_x = self._params["approach_distance_m"]
            button_y = self._params["button_offset_y_m"]
            button_z = self._params["button_height_m"] - 0.5  # relative to body

            cmd = RobotCommandBuilder.arm_pose_command(
                x=button_x, y=button_y, z=button_z,
                qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                frame_name=BODY_FRAME_NAME,
                seconds=2.0
            )
            self._command_id = self._command_client.robot_command(cmd)
            self._button_state = ButtonState.PRESS_BUTTON
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == ButtonState.PRESS_BUTTON:
            # Wait for positioning, then press forward
            if time.time() - self._state_start_time < 2.5:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Button] Pressing button — arm will hold door open...")
            press_dist = self._params.get("press_distance_m", 0.15)
            press_x = self._params["approach_distance_m"] + press_dist
            button_y = self._params["button_offset_y_m"]
            button_z = self._params["button_height_m"] - 0.5

            cmd = RobotCommandBuilder.arm_pose_command(
                x=press_x, y=button_y, z=button_z,
                qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                frame_name=BODY_FRAME_NAME,
                seconds=1.0
            )
            self._command_id = self._command_client.robot_command(cmd)
            self._button_state = ButtonState.WALK_THROUGH
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == ButtonState.WALK_THROUGH:
            # Wait for press, then walk through with arm holding door
            if time.time() - self._state_start_time < 2.0:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Button] Walking through with arm holding door...")
            distance = self._params["walk_through_distance_m"]
            try:
                from bosdyn.client.robot_state import RobotStateClient
                state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
                robot_state = state_client.get_robot_state()
                frame_tree = robot_state.kinematic_state.transforms_snapshot

                cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    goal_x_rt_body=distance,
                    goal_y_rt_body=0.0,
                    goal_heading_rt_body=0.0,
                    frame_tree_snapshot=frame_tree
                )
                self._command_client.robot_command(cmd, end_time_secs=time.time() + 15.0)
            except Exception as e:
                logger.error(f"[Button] Walk-through failed: {e}")

            self._button_state = ButtonState.STOW_ARM
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == ButtonState.STOW_ARM:
            # Wait for walk-through, then stow arm
            if time.time() - self._state_start_time < 8.0:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Button] Through door, stowing arm...")
            self._stow_arm()
            self._button_state = ButtonState.DONE
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == ButtonState.DONE:
            # Give time to stow
            if time.time() - self._state_start_time < 2.5:
                return remote_pb2.TickResponse.STATUS_RUNNING
            logger.info("[Button] Door operation complete!")
            self._button_state = ButtonState.IDLE
            return remote_pb2.TickResponse.STATUS_SUCCESS

        elif state == ButtonState.FAILED:
            logger.error("[Button] Door operation failed")
            return remote_pb2.TickResponse.STATUS_FAILURE

        return remote_pb2.TickResponse.STATUS_RUNNING

    # -----------------------------------------------------------------------
    # Handle-based state machine
    # -----------------------------------------------------------------------
    def _tick_handle(self) -> int:
        """Advance the handle-based door state machine.

        Returns TickResponse status enum value.

        Note: Handle-based door opening is the harder problem. This implements
        a basic version using pre-configured handle positions. For production
        use, integrate gripper camera detection for handle localization.
        """
        state = self._handle_state

        if state == HandleState.IDLE:
            return remote_pb2.TickResponse.STATUS_SUCCESS

        elif state == HandleState.UNSTOW_ARM:
            logger.info("[Handle] Unstowing arm...")
            cmd = RobotCommandBuilder.arm_ready_command()
            self._command_id = self._command_client.robot_command(cmd)
            self._handle_state = HandleState.OPEN_GRIPPER
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == HandleState.OPEN_GRIPPER:
            if time.time() - self._state_start_time < 2.0:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Handle] Opening gripper...")
            cmd = RobotCommandBuilder.claw_gripper_open_command()
            self._command_id = self._command_client.robot_command(cmd)
            self._handle_state = HandleState.POSITION_ARM
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == HandleState.POSITION_ARM:
            if time.time() - self._state_start_time < 1.5:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Handle] Positioning arm to handle...")
            handle_x = self._params["approach_distance_m"]
            handle_y = self._params["handle_offset_y_m"]
            handle_z = self._params["handle_height_m"] - 0.5  # relative to body

            cmd = RobotCommandBuilder.arm_pose_command(
                x=handle_x, y=handle_y, z=handle_z,
                qw=0.707, qx=0.0, qy=0.707, qz=0.0,  # gripper pointing forward
                frame_name=BODY_FRAME_NAME,
                seconds=2.0
            )
            self._command_id = self._command_client.robot_command(cmd)
            self._handle_state = HandleState.CLOSE_GRIPPER
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == HandleState.CLOSE_GRIPPER:
            if time.time() - self._state_start_time < 2.5:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Handle] Closing gripper on handle...")
            cmd = RobotCommandBuilder.claw_gripper_close_command()
            self._command_id = self._command_client.robot_command(cmd)
            self._handle_state = HandleState.PUSH_OR_PULL
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == HandleState.PUSH_OR_PULL:
            if time.time() - self._state_start_time < 1.5:
                return remote_pb2.TickResponse.STATUS_RUNNING

            direction = self._params["door_direction"]
            logger.info(f"[Handle] {direction.capitalize()}ing door...")

            handle_y = self._params["handle_offset_y_m"]
            handle_z = self._params["handle_height_m"] - 0.5

            if direction == "push":
                # Push: move arm forward
                target_x = self._params["approach_distance_m"] + 0.4
            else:
                # Pull: move arm backward
                target_x = self._params["approach_distance_m"] - 0.4

            cmd = RobotCommandBuilder.arm_pose_command(
                x=target_x, y=handle_y, z=handle_z,
                qw=0.707, qx=0.0, qy=0.707, qz=0.0,
                frame_name=BODY_FRAME_NAME,
                seconds=2.0
            )
            self._command_id = self._command_client.robot_command(cmd)
            self._handle_state = HandleState.RELEASE_HANDLE
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == HandleState.RELEASE_HANDLE:
            if time.time() - self._state_start_time < 3.0:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Handle] Releasing handle...")
            cmd = RobotCommandBuilder.claw_gripper_open_command()
            self._command_id = self._command_client.robot_command(cmd)
            self._handle_state = HandleState.STOW_ARM
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == HandleState.STOW_ARM:
            if time.time() - self._state_start_time < 1.5:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Handle] Stowing arm...")
            self._stow_arm()
            self._handle_state = HandleState.WALK_THROUGH
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == HandleState.WALK_THROUGH:
            if time.time() - self._state_start_time < 2.0:
                return remote_pb2.TickResponse.STATUS_RUNNING

            logger.info("[Handle] Walking through doorway...")
            distance = self._params["walk_through_distance_m"]
            try:
                from bosdyn.client.robot_state import RobotStateClient
                state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
                robot_state = state_client.get_robot_state()
                frame_tree = robot_state.kinematic_state.transforms_snapshot

                cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                    goal_x_rt_body=distance,
                    goal_y_rt_body=0.0,
                    goal_heading_rt_body=0.0,
                    frame_tree_snapshot=frame_tree
                )
                self._command_client.robot_command(cmd, end_time_secs=time.time() + 15.0)
            except Exception as e:
                logger.error(f"[Handle] Walk-through failed: {e}")

            self._handle_state = HandleState.DONE
            self._state_start_time = time.time()
            return remote_pb2.TickResponse.STATUS_RUNNING

        elif state == HandleState.DONE:
            if time.time() - self._state_start_time < 8.0:
                return remote_pb2.TickResponse.STATUS_RUNNING
            logger.info("[Handle] Door operation complete!")
            self._handle_state = HandleState.IDLE
            return remote_pb2.TickResponse.STATUS_SUCCESS

        elif state == HandleState.FAILED:
            logger.error("[Handle] Door operation failed")
            return remote_pb2.TickResponse.STATUS_FAILURE

        return remote_pb2.TickResponse.STATUS_RUNNING

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _stow_arm(self):
        """Safely stow the arm."""
        try:
            if self._command_client:
                cmd = RobotCommandBuilder.arm_stow_command()
                self._command_client.robot_command(cmd)
                time.sleep(2.0)
        except Exception as e:
            logger.error(f"Arm stow failed: {e}")

    def _parse_inputs(self, inputs):
        """Parse mission node inputs into service parameters."""
        try:
            for key, value in inputs.items():
                if key in self._params:
                    if hasattr(value, 'string_value'):
                        self._params[key] = value.string_value.value
                    elif hasattr(value, 'double_value'):
                        self._params[key] = value.double_value.value
                    elif hasattr(value, 'int_value'):
                        self._params[key] = value.int_value.value
                    logger.info(f"  param {key} = {self._params[key]}")
        except Exception as e:
            logger.warning(f"Error parsing inputs: {e} — using defaults")
