"""Motion stop, end-of-motion arm restore, trajectory wait."""
import time

from bosdyn.client.robot_command import RobotCommandBuilder

import src.spot.robot as _r
from src.spot.robot.arm import RPC_TIMEOUT_S


def send_stop(cmd_client) -> bool:
    try:
        cmd_client.robot_command(RobotCommandBuilder.stop_command(), timeout=RPC_TIMEOUT_S)
        return True
    except Exception as e:
        print(f"[Spot] ✗ stop_command failed: {e}")
        return False


def finish_motion(natural: bool, stop_event):
    """End of a motion thread. Cancelled -> nothing (the stop or the next
    motion owns the robot). Natural finish -> restore the arm. Anything else
    -> stop the robot and keep the arm tucked."""
    if stop_event is not None and stop_event.is_set():
        return
    if natural:
        _r.ARM.restore_after_motion(stop_event)
        return
    print("[Spot] Motion did not finish cleanly — stopping; arm stays tucked")
    if _r.CONNECTION._session is not None:
        send_stop(_r.CONNECTION._session["cmd"])


def thread_then_restore(fn, stop_event):
    """Nav-thread target wrapper: fn() returns True only on a natural finish."""
    natural = False
    try:
        natural = bool(fn())
    finally:
        finish_motion(natural, stop_event)


def wait_trajectory_done(cmd_client, cmd_id, deadline, stop_event):
    """Poll a mobility trajectory until it reports AT_GOAL and SETTLED (True),
    or the deadline passes / stop_event is set (False)."""
    while time.time() < deadline and not stop_event.is_set():
        try:
            fb = cmd_client.robot_command_feedback(cmd_id)
            traj = (fb.feedback.synchronized_feedback
                    .mobility_command_feedback.se2_trajectory_feedback)
            if traj.status == 1 and traj.body_movement_status == 2:  # AT_GOAL + SETTLED
                return True
        except Exception:
            pass
        stop_event.wait(0.25)
    return False


def start_restore_watch(cmd_client, cmd_id, deadline, gen=None):
    """Register a watcher as the nav thread: waits for the trajectory, then
    restores the arm. is_navigating() is True meanwhile (gaze/emote gate)."""
    return _r.NAV.spawn(
        lambda ev: thread_then_restore(
            lambda: wait_trajectory_done(cmd_client, cmd_id, deadline, ev), ev),
        mode="walk", gen=gen)


def abort_motion(cmd_client, name):
    """A stop/sit/stow arrived while this locomotion was being set up."""
    print(f"[Spot] '{name}' cancelled by a stop")
    _r.NAV.cancel()
    send_stop(cmd_client)
    _r.ARM.stow_now(cmd_client, force=True)
    return False
