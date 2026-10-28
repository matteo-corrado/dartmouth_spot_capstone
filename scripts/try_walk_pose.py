#!/usr/bin/env python3
"""Operator scratch tool (Task 14, operator session 2): try the walking-pose
candidates on the robot and pick the tightest one where the claw can still
open and close without hitting the body.

    (a) "stow"   arm_stow_command
    (b) "carry"  arm_carry_command
    (c) "joints" sh0=0, sh1=-2.6, el0=2.6, el1=0, wr0=0, wr1=0 (slow joint move)

For each candidate the arm goes to the pose, then the claw opens and closes
twice. Press Enter to advance. Record the winner in src/session.py WALK_POSE.

REQUIRES a live E-Stop (run scripts/estop_run.py in another terminal first).
STOP voice control first: this takes the lease (spot_session forcefully takes
it). The E-Stop is checked before the robot powers on or stands. Writes nothing.

    python scripts/try_walk_pose.py [--variants stow carry joints] [--open 0.6]
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from bosdyn.client.estop import EstopClient
from bosdyn.client.robot_command import (
    RobotCommandBuilder, block_until_arm_arrives, blocking_stand)

from src.session import spot_session, move_arm_to, walk_pose_command

LABELS = {"stow": "(a) arm_stow_command", "carry": "(b) arm_carry_command",
          "joints": "(c) custom joint pose"}


def _estop_live(robot) -> bool:
    try:
        status = robot.ensure_client(EstopClient.default_service_name).get_status()
    except Exception as e:
        print(f"Could not read E-Stop status: {e}")
        return False
    if not status.endpoints:
        print("No E-Stop endpoint registered. Start scripts/estop_run.py first.")
        return False
    if robot.is_estopped():
        print("Robot is E-Stopped. Release the E-Stop, then rerun.")
        return False
    return True


def _claw(cmd, frac, pause):
    cmd.robot_command(RobotCommandBuilder.claw_gripper_open_fraction_command(frac))
    time.sleep(pause)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variants", nargs="+", default=["stow", "carry", "joints"],
                    choices=list(LABELS))
    ap.add_argument("--open", type=float, default=0.6, help="claw open fraction to test")
    args = ap.parse_args()

    with spot_session(stand_on_enter=False, sit_on_exit=False) as s:
        robot, cmd = s["robot"], s["cmd"]
        if not robot.has_arm():
            print("No arm on this robot.")
            return 1
        if not _estop_live(robot):  # BEFORE power-on / stand
            return 1
        print("E-Stop live. Powering on and standing...")
        robot.power_on(timeout_sec=20)
        blocking_stand(cmd, timeout_sec=20)
        try:
            print("Deploying to talk pose first...")
            move_arm_to(cmd, "talk", timeout_s=6.0)
            for v in args.variants:
                input(f"\nEnter -> move to {LABELS[v]} (Ctrl+C to abort) ")
                if v == "stow":
                    _claw(cmd, 0.0, 0.5)
                cmd_id = cmd.robot_command(walk_pose_command(v))
                arrived = block_until_arm_arrives(cmd, cmd_id, 6.0)
                print(f"  arrived={arrived}. Watching the claw: open/close x2...")
                for _ in range(2):
                    _claw(cmd, args.open, 0.8)
                    _claw(cmd, 0.0, 0.8)
                print("  Did the claw clear the body? Note it.")
        except KeyboardInterrupt:
            print("\nAborted.")
        finally:
            print("Stowing arm.")
            try:
                move_arm_to(cmd, "stowed", timeout_s=6.0)
            except Exception as e:
                print(f"Stow failed: {e}")
    print("\nSet WALK_POSE in src/session.py to the winner: 'stow' | 'carry' | 'joints'.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
