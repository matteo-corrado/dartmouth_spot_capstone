#!/usr/bin/env python3
"""Standalone door-opening test — no AutoWalk needed.

Position Spot in front of a door, pick a mode, and watch it go.

Usage:
    # Push-button door (default)
    python scripts/test_door_standalone.py --mode button

    # Handle door — push direction
    python scripts/test_door_standalone.py --mode handle --direction push

    # Handle door — pull direction
    python scripts/test_door_standalone.py --mode handle --direction pull

    # Adjust heights/offsets
    python scripts/test_door_standalone.py --mode button --button-height 1.1 --button-offset-y -0.25

    # Dry run — print state transitions without sending commands
    python scripts/test_door_standalone.py --mode button --dry-run

Requires:
    - E-Stop running (scripts/estop_run.py in separate terminal)
    - Robot powered on and standing
    - Spot positioned facing the door at ~0.8m distance
"""

import sys
import pathlib
import argparse
import logging
import time
import os

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import BODY_FRAME_NAME


def connect_to_spot():
    """Connect and authenticate with Spot."""
    load_dotenv(project_root / ".env")
    spot_ip = os.getenv("BOSDYN_ROBOT_IP", "192.168.80.3")
    spot_user = os.getenv("BOSDYN_CLIENT_USERNAME", "SpotDBEC")
    spot_pass = os.getenv("BOSDYN_CLIENT_PASSWORD")

    if not spot_pass:
        print("ERROR: Set BOSDYN_CLIENT_PASSWORD in .env file")
        sys.exit(1)

    print(f"Connecting to Spot at {spot_ip}...")
    sdk = create_standard_sdk("DoorTest")
    robot = sdk.create_robot(spot_ip)
    robot.authenticate(spot_user, spot_pass)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()
    print("Connected!")
    return robot


def stow_arm(command_client):
    """Safely stow the arm."""
    print("  Stowing arm...")
    cmd = RobotCommandBuilder.arm_stow_command()
    command_client.robot_command(cmd)
    time.sleep(2.0)


def test_button_door(robot, command_client, params, dry_run=False):
    """Test push-button door opening sequence."""
    button_x = params["approach_distance_m"]
    button_y = params["button_offset_y_m"]
    button_z = params["button_height_m"] - 0.5  # relative to body center

    steps = [
        ("Unstow arm (ready position)", "arm_ready"),
        ("Position arm near button", ("arm_pose", button_x, button_y, button_z)),
        ("Press button (extend forward)", ("arm_pose", button_x + 0.15, button_y, button_z)),
        ("Retract arm from button", ("arm_pose", button_x - 0.2, button_y, button_z)),
        (f"Wait {params['wait_time_s']}s for door to open", ("wait", params["wait_time_s"])),
        ("Stow arm", "arm_stow"),
        (f"Walk through ({params['walk_through_distance_m']}m forward)", ("walk", params["walk_through_distance_m"])),
    ]

    for i, (desc, action) in enumerate(steps, 1):
        print(f"\n  [{i}/{len(steps)}] {desc}")
        if dry_run:
            print("    [DRY RUN] Skipped")
            continue

        input("    Press Enter to execute (Ctrl+C to abort)...")

        if action == "arm_ready":
            cmd = RobotCommandBuilder.arm_ready_command()
            command_client.robot_command(cmd)
            time.sleep(2.5)

        elif action == "arm_stow":
            stow_arm(command_client)

        elif isinstance(action, tuple) and action[0] == "arm_pose":
            _, x, y, z = action
            cmd = RobotCommandBuilder.arm_pose_command(
                x=x, y=y, z=z,
                qw=1.0, qx=0.0, qy=0.0, qz=0.0,
                frame_name=BODY_FRAME_NAME,
                seconds=2.0
            )
            command_client.robot_command(cmd)
            time.sleep(2.5)

        elif isinstance(action, tuple) and action[0] == "wait":
            wait_time = action[1]
            for sec in range(int(wait_time)):
                print(f"    Waiting... {sec+1}/{int(wait_time)}s")
                time.sleep(1.0)

        elif isinstance(action, tuple) and action[0] == "walk":
            distance = action[1]
            state_client = robot.ensure_client(RobotStateClient.default_service_name)
            robot_state = state_client.get_robot_state()
            frame_tree = robot_state.kinematic_state.transforms_snapshot
            cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                goal_x_rt_body=distance,
                goal_y_rt_body=0.0,
                goal_heading_rt_body=0.0,
                frame_tree_snapshot=frame_tree
            )
            command_client.robot_command(cmd, end_time_secs=time.time() + 15.0)
            time.sleep(8.0)

        print("    Done")

    print("\n  Button door sequence complete!")


def test_handle_door(robot, command_client, params, dry_run=False):
    """Test handle-based door opening sequence."""
    handle_x = params["approach_distance_m"]
    handle_y = params["handle_offset_y_m"]
    handle_z = params["handle_height_m"] - 0.5
    direction = params["door_direction"]

    if direction == "push":
        target_x = handle_x + 0.4
    else:
        target_x = handle_x - 0.4

    steps = [
        ("Unstow arm (ready position)", "arm_ready"),
        ("Open gripper", "gripper_open"),
        ("Position arm at handle", ("arm_pose_grip", handle_x, handle_y, handle_z)),
        ("Close gripper (grasp handle)", "gripper_close"),
        (f"{direction.capitalize()} door", ("arm_pose_grip", target_x, handle_y, handle_z)),
        ("Open gripper (release handle)", "gripper_open"),
        ("Stow arm", "arm_stow"),
        (f"Walk through ({params['walk_through_distance_m']}m forward)", ("walk", params["walk_through_distance_m"])),
    ]

    for i, (desc, action) in enumerate(steps, 1):
        print(f"\n  [{i}/{len(steps)}] {desc}")
        if dry_run:
            print("    [DRY RUN] Skipped")
            continue

        input("    Press Enter to execute (Ctrl+C to abort)...")

        if action == "arm_ready":
            cmd = RobotCommandBuilder.arm_ready_command()
            command_client.robot_command(cmd)
            time.sleep(2.5)

        elif action == "arm_stow":
            stow_arm(command_client)

        elif action == "gripper_open":
            cmd = RobotCommandBuilder.claw_gripper_open_command()
            command_client.robot_command(cmd)
            time.sleep(1.5)

        elif action == "gripper_close":
            cmd = RobotCommandBuilder.claw_gripper_close_command()
            command_client.robot_command(cmd)
            time.sleep(1.5)

        elif isinstance(action, tuple) and action[0] == "arm_pose_grip":
            _, x, y, z = action
            cmd = RobotCommandBuilder.arm_pose_command(
                x=x, y=y, z=z,
                qw=0.707, qx=0.0, qy=0.707, qz=0.0,  # gripper pointing forward
                frame_name=BODY_FRAME_NAME,
                seconds=2.0
            )
            command_client.robot_command(cmd)
            time.sleep(2.5)

        elif isinstance(action, tuple) and action[0] == "walk":
            distance = action[1]
            state_client = robot.ensure_client(RobotStateClient.default_service_name)
            robot_state = state_client.get_robot_state()
            frame_tree = robot_state.kinematic_state.transforms_snapshot
            cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
                goal_x_rt_body=distance,
                goal_y_rt_body=0.0,
                goal_heading_rt_body=0.0,
                frame_tree_snapshot=frame_tree
            )
            command_client.robot_command(cmd, end_time_secs=time.time() + 15.0)
            time.sleep(8.0)

        print("    Done")

    print(f"\n  Handle door ({direction}) sequence complete!")


def main():
    parser = argparse.ArgumentParser(description="Standalone door-opening test")
    parser.add_argument("--mode", choices=["button", "handle"], default="button",
                        help="Door type: button (push-button) or handle (default: button)")
    parser.add_argument("--direction", choices=["push", "pull"], default="push",
                        help="Door direction for handle mode (default: push)")
    parser.add_argument("--button-height", type=float, default=1.0,
                        help="Button height in meters from ground (default: 1.0)")
    parser.add_argument("--handle-height", type=float, default=1.0,
                        help="Handle height in meters from ground (default: 1.0)")
    parser.add_argument("--button-offset-y", type=float, default=-0.3,
                        help="Button lateral offset in meters (default: -0.3)")
    parser.add_argument("--handle-offset-y", type=float, default=0.0,
                        help="Handle lateral offset in meters (default: 0.0)")
    parser.add_argument("--approach-distance", type=float, default=0.8,
                        help="Distance from door in meters (default: 0.8)")
    parser.add_argument("--walk-distance", type=float, default=2.0,
                        help="Walk-through distance in meters (default: 2.0)")
    parser.add_argument("--wait-time", type=float, default=5.0,
                        help="Seconds to wait for button door to open (default: 5.0)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print steps without sending commands to robot")
    args = parser.parse_args()

    params = {
        "door_direction": args.direction,
        "button_height_m": args.button_height,
        "handle_height_m": args.handle_height,
        "button_offset_y_m": args.button_offset_y,
        "handle_offset_y_m": args.handle_offset_y,
        "approach_distance_m": args.approach_distance,
        "walk_through_distance_m": args.walk_distance,
        "wait_time_s": args.wait_time,
    }

    print("=" * 60)
    print("SPOT DOOR OPENING — STANDALONE TEST")
    print("=" * 60)
    print(f"\n  Mode:      {args.mode}")
    if args.mode == "handle":
        print(f"  Direction: {args.direction}")
        print(f"  Handle height: {args.handle_height}m")
        print(f"  Handle offset Y: {args.handle_offset_y}m")
    else:
        print(f"  Button height: {args.button_height}m")
        print(f"  Button offset Y: {args.button_offset_y}m")
        print(f"  Wait time: {args.wait_time}s")
    print(f"  Approach distance: {args.approach_distance}m")
    print(f"  Walk-through: {args.walk_distance}m")
    if args.dry_run:
        print(f"\n  *** DRY RUN — no commands will be sent ***")
    print(f"\n  IMPORTANT: Make sure E-Stop is running!")
    print(f"  Position Spot ~{args.approach_distance}m from the door, facing it.")
    print(f"\n  Each step requires Enter to proceed — you can abort with Ctrl+C at any time.")

    if not args.dry_run:
        input("\nPress Enter when ready...")
        robot = connect_to_spot()

        # Acquire lease
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease = lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client)
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)

        # Make sure robot is standing
        print("Ensuring Spot is standing...")
        blocking_stand(command_client)
        time.sleep(1.0)
    else:
        robot = None
        command_client = None
        print("\n[DRY RUN] Skipping robot connection")

    try:
        if args.mode == "button":
            test_button_door(robot, command_client, params, dry_run=args.dry_run)
        else:
            test_handle_door(robot, command_client, params, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n\nAborted by user!")
        if command_client and not args.dry_run:
            print("Stowing arm and stopping...")
            stow_arm(command_client)
            command_client.robot_command(RobotCommandBuilder.stop_command())
    finally:
        if not args.dry_run and 'lease_keepalive' in locals():
            lease_keepalive.shutdown()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
