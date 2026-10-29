"""Motion I/O for the vision actions: a small protocol plus the real-robot adapter.

``SpotNavIO`` is the only place the vision package sends motion. Every
``move_body`` / ``velocity`` re-checks the dispatcher's safety generation
immediately before ``robot_command`` (under ``_arm_lock``, like
``gaze_at_bearing``) and again right after it: if a stop landed while the
command was in flight, a ``stop_command`` follows so the stop always wins.
Spot's built-in obstacle avoidance is left on (default mobility params plus
the walk speed cap from ``capped_mobility_params``).
"""
from __future__ import annotations

import logging
import math
import time
from typing import Protocol

from .types import Detection

logger = logging.getLogger(__name__)

TRAJ_END_S = 4.0       # trajectory command lifetime; the go-to loop re-issues it
VEL_END_S = 0.5        # velocity command lifetime: a dead loop stops the robot
RPC_TIMEOUT_S = 0.5    # < spot_dispatch.SAFETY_LOCK_TIMEOUT_S, like the gaze send
LOCK_TIMEOUT_S = 0.5   # fail closed if the arm lock is busy (a stop's stow holds it)


class NavIO(Protocol):
    def snapshot(self, sources: list[str], labels: set[str] | None,
                 model: str) -> list[Detection]: ...

    def odom_pose(self) -> tuple[float, float, float]: ...  # x, y, yaw (rad) in odom

    def move_body(self, x: float, y: float, yaw: float) -> bool: ...

    def velocity(self, vx: float, vy: float, vrot: float) -> bool: ...

    def halt(self) -> None: ...

    def aborted(self) -> bool: ...

    def speak(self, text: str) -> None: ...


def _odom_tform_body(snapshot):
    from bosdyn.client.frame_helpers import get_odom_tform_body
    return get_odom_tform_body(snapshot)


class SpotNavIO:
    """NavIO on the real robot. ``session`` is spot_dispatch's session dict,
    ``gen`` the safety generation taken when the action was dispatched."""

    def __init__(self, session, gen, speak=None):
        self.session = session
        self.gen = gen
        self._speak = speak

    # -- sensing ------------------------------------------------------------
    def snapshot(self, sources, labels, model):
        from .world import snapshot
        return snapshot(self.session["robot"], list(sources), model=model, labels=labels)

    def odom_pose(self):
        state = self.session["state"].get_robot_state()
        tf = _odom_tform_body(state.kinematic_state.transforms_snapshot)
        return (float(tf.x), float(tf.y), float(tf.rot.to_yaw()))

    def aborted(self) -> bool:
        from src.voice_control import spot_dispatch as sd
        return sd._aborted(self.gen)

    # -- motion -------------------------------------------------------------
    def _send(self, cmd, end_s: float) -> bool:
        """Check the safety generation, send, check again. Never raises."""
        from src.voice_control import spot_dispatch as sd
        cmd_client = self.session["cmd"]
        if not sd._arm_lock.acquire(timeout=LOCK_TIMEOUT_S):
            logger.warning("navio: arm lock busy, motion not sent")
            return False
        try:
            if sd._aborted(self.gen):
                return False
            cmd_client.robot_command(cmd, end_time_secs=time.time() + end_s,
                                     timeout=RPC_TIMEOUT_S)
        except Exception as e:
            logger.warning("navio: motion command failed: %s", e)
            return False
        finally:
            sd._arm_lock.release()
        if sd._aborted(self.gen):
            # A stop raced our send: make sure the last word is a stop.
            self.halt()
            return False
        return True

    def move_body(self, x, y, yaw) -> bool:
        """Body-frame trajectory goal (m, m, rad). False if aborted or failed."""
        try:
            from bosdyn.client.robot_command import RobotCommandBuilder
            from src.graph_nav_utils import capped_mobility_params
            if self.aborted():
                return False
            state = self.session["state"].get_robot_state()
            cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                goal_x_rt_body=float(x), goal_y_rt_body=float(y),
                goal_heading_rt_body=float(yaw),
                frame_tree_snapshot=state.kinematic_state.transforms_snapshot,
                params=capped_mobility_params())
        except Exception as e:
            logger.warning("navio: move_body build failed: %s", e)
            return False
        return self._send(cmd, TRAJ_END_S)

    def velocity(self, vx, vy, vrot) -> bool:
        """Body-frame velocity (m/s, m/s, rad/s), lives VEL_END_S. False if aborted/failed."""
        try:
            from bosdyn.client.robot_command import RobotCommandBuilder
            from src.graph_nav_utils import MAX_TURN_RATE, capped_mobility_params, clamp_speed
            if self.aborted():
                return False
            vr = float(vrot)
            if not math.isfinite(vr):
                return False
            cmd = RobotCommandBuilder.synchro_velocity_command(
                v_x=clamp_speed(float(vx)), v_y=clamp_speed(float(vy)),
                v_rot=max(-MAX_TURN_RATE, min(MAX_TURN_RATE, vr)),
                params=capped_mobility_params())
        except Exception as e:
            logger.warning("navio: velocity build failed: %s", e)
            return False
        return self._send(cmd, VEL_END_S)

    def halt(self) -> None:
        """Unconditional stop_command. Never raises."""
        try:
            from bosdyn.client.robot_command import RobotCommandBuilder
            self.session["cmd"].robot_command(RobotCommandBuilder.stop_command(),
                                              timeout=RPC_TIMEOUT_S)
        except Exception as e:
            logger.warning("navio: halt failed: %s", e)

    def speak(self, text: str) -> None:
        if self._speak is None:
            return
        try:
            self._speak(text)
        except Exception as e:
            logger.warning("navio: speak failed: %s", e)
