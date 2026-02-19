#!/usr/bin/env python3
"""Push-bar door opening using DoorService AutoPushCommand.

AutoPushCommand is BD's built-in command for push-bar doors. The robot
points its hand down and pushes the door open with its wrist. It needs
a 3D push point in the vision frame + hinge side.

Usage:
    # VLM auto-detects push bar, computes 3D point (recommended):
    python scripts/test_door_standalone.py --vlm
    python scripts/test_door_standalone.py --vlm --depth 0.5

    # Direct vision-frame coordinates (from previous VLM run):
    python scripts/test_door_standalone.py --vision-xyz 1.65,2.68,0.81

    # Change hinge side (default: left):
    python scripts/test_door_standalone.py --vlm --hinge right

Requires:
    - E-Stop running (scripts/estop_run.py)
    - Robot powered on and standing
    - Spot ~1m from push bar door, facing it
    - For --vlm: Ollama running with qwen2.5vl:7b
"""

import sys
import pathlib
import argparse
import time
import os
import re
import base64
import math

import numpy as np
import requests

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import (
    RobotCommandBuilder, RobotCommandClient, blocking_stand)
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient
from bosdyn.client.door import DoorClient
from bosdyn.client import frame_helpers
from bosdyn.api import geometry_pb2, image_pb2
from bosdyn.api.spot import door_pb2
from bosdyn import geometry as geo

OLLAMA_URL = "http://localhost:11434"
VLM_MODEL = "qwen2.5vl:7b"


def connect_to_spot():
    load_dotenv(project_root / ".env")
    spot_ip = os.getenv("BOSDYN_ROBOT_IP", "192.168.80.3")
    spot_user = os.getenv("BOSDYN_CLIENT_USERNAME", "SpotDBEC")
    spot_pass = os.getenv("BOSDYN_CLIENT_PASSWORD")
    if not spot_pass:
        print("ERROR: Set BOSDYN_CLIENT_PASSWORD in .env")
        sys.exit(1)

    print(f"Connecting to Spot at {spot_ip}...")
    sdk = create_standard_sdk("DoorTest")
    robot = sdk.create_robot(spot_ip)
    robot.authenticate(spot_user, spot_pass)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()
    print("Connected!")
    return robot


def clear_behavior_faults(robot, cmd_client):
    """Clear any behavior faults so commands can be sent."""
    state_client = robot.ensure_client(RobotStateClient.default_service_name)
    state = state_client.get_robot_state()
    faults = state.behavior_fault_state.faults
    if not faults:
        return
    print(f"  Clearing {len(faults)} behavior fault(s)...")
    for fault in faults:
        try:
            cmd_client.clear_behavior_fault(fault.behavior_fault_id)
        except Exception as e:
            print(f"    Warning: could not clear fault {fault.behavior_fault_id}: {e}")
    time.sleep(1.0)


# ── Image capture ──────────────────────────────────────────

def capture_and_decode(image_client, source_name):
    """Capture image, return (response, jpeg_bytes, rotated_np_array)."""
    from PIL import Image
    import io

    responses = image_client.get_image_from_sources([source_name])
    resp = responses[0]
    raw = resp.shot.image.data
    fmt = resp.shot.image.format

    if fmt == image_pb2.Image.FORMAT_JPEG:
        img = Image.open(io.BytesIO(raw)).convert("L")
    else:
        h, w = resp.shot.image.rows, resp.shot.image.cols
        img = Image.frombytes("L", (w, h), raw)

    rotated = img.rotate(-90, expand=True)  # fisheye mounted sideways
    buf = io.BytesIO()
    rotated.save(buf, format="JPEG")
    return resp, buf.getvalue(), np.array(rotated)


# ── VLM push bar detection ────────────────────────────────

def ask_vlm(image_bytes, prompt):
    """Send image to VLM, return text response."""
    b64 = base64.b64encode(image_bytes).decode()
    r = requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": VLM_MODEL,
        "messages": [{"role": "user", "content": prompt, "images": [b64]}],
        "stream": False, "keep_alive": "5m",
        "options": {"num_gpu": 99, "num_predict": 300},
    }, timeout=120)
    if r.status_code != 200:
        return None
    return r.json().get("message", {}).get("content", "").strip()


def vlm_find_push_bar_pixel(image_client, cmd_client, camera="left"):
    """Pitch up, capture, ask VLM for push bar pixel. Returns (img_resp, px_raw, py_raw)."""
    print("\n  [VLM] Pitching up to see door...")
    pitch_cmd = RobotCommandBuilder.synchro_stand_command(
        footprint_R_body=geo.EulerZXY(pitch=-0.4, roll=0.0, yaw=0.0))
    cmd_client.robot_command(pitch_cmd)
    time.sleep(2.0)

    source = f"front{camera}_fisheye_image"
    print(f"  [VLM] Capturing from {source}...")
    img_resp, jpeg_bytes, rotated = capture_and_decode(image_client, source)
    h, w = rotated.shape[:2]
    print(f"    Image: {w}x{h} (rotated)")

    # Save for inspection
    calib_dir = project_root / "door_calibration"
    calib_dir.mkdir(exist_ok=True)
    with open(calib_dir / "door_view.jpg", "wb") as f:
        f.write(jpeg_bytes)
    print(f"    Saved: {calib_dir / 'door_view.jpg'}")

    print("  [VLM] Asking VLM to find push bar...")
    vlm_prompt = (
        f"This image is {w}x{h} pixels from a robot's front camera. "
        f"The robot is facing a door with a horizontal push bar. "
        f"What are the pixel coordinates (x, y) of the CENTER of the push bar? "
        f"Reply with ONLY the coordinates in this format: (x, y)"
    )
    t0 = time.time()
    response = ask_vlm(jpeg_bytes, vlm_prompt)
    print(f"    VLM ({time.time()-t0:.1f}s): {response}")

    if not response:
        return None, None, None

    match = re.search(r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?', response)
    if not match:
        print(f"    ERROR: Could not parse coordinates")
        return None, None, None

    px, py = int(match.group(1)), int(match.group(2))
    px = max(0, min(w - 1, px))
    py = max(0, min(h - 1, py))
    print(f"    Push bar at pixel: ({px}, {py}) in rotated image")

    # Un-rotate: orig_x = rot_y, orig_y = H_orig - 1 - rot_x
    raw_rows = img_resp.shot.image.rows or w
    pixel_x_raw = py
    pixel_y_raw = raw_rows - 1 - px
    print(f"    Raw camera pixel: ({pixel_x_raw}, {pixel_y_raw})")

    # Reset pitch
    cmd_client.robot_command(RobotCommandBuilder.synchro_stand_command())
    time.sleep(0.5)

    return img_resp, pixel_x_raw, pixel_y_raw


# ── 3D point computation ──────────────────────────────────

def compute_3d_point_vision(img_resp, pixel_x_raw, pixel_y_raw, depth):
    """Compute 3D push point in vision frame from raw pixel + estimated depth."""
    pinhole = img_resp.source.pinhole
    fx = pinhole.intrinsics.focal_length.x
    fy = pinhole.intrinsics.focal_length.y
    cx = pinhole.intrinsics.principal_point.x
    cy = pinhole.intrinsics.principal_point.y
    print(f"    Intrinsics: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}")

    if fx <= 0 or fy <= 0:
        print("    WARNING: Invalid intrinsics")
        return None

    # Ray in camera frame
    dir_x = (pixel_x_raw - cx) / fx
    dir_y = (pixel_y_raw - cy) / fy
    dir_z = 1.0

    # Transform to vision frame
    snapshot = img_resp.shot.transforms_snapshot
    vision_tform_cam = frame_helpers.get_a_tform_b(
        snapshot, frame_helpers.VISION_FRAME_NAME,
        img_resp.shot.frame_name_image_sensor)

    cam_origin = np.array(vision_tform_cam.transform_point(0, 0, 0))
    ray_pt = np.array(vision_tform_cam.transform_point(dir_x, dir_y, dir_z))
    ray_dir = ray_pt - cam_origin
    ray_dir = ray_dir / np.linalg.norm(ray_dir)

    push_point = cam_origin + depth * ray_dir

    print(f"    Camera origin (vision): [{cam_origin[0]:.2f}, {cam_origin[1]:.2f}, {cam_origin[2]:.2f}]")
    print(f"    Ray dir: [{ray_dir[0]:.3f}, {ray_dir[1]:.3f}, {ray_dir[2]:.3f}]")
    print(f"    Push point (vision, depth={depth}m): "
          f"[{push_point[0]:.2f}, {push_point[1]:.2f}, {push_point[2]:.2f}]")

    return push_point


# ── AutoPushCommand via DoorService ───────────────────────

def auto_push_door(robot, push_point, hinge_side_str):
    """Send AutoPushCommand and monitor feedback."""
    door_client = robot.ensure_client(DoorClient.default_service_name)

    hinge = (door_pb2.DoorCommand.HINGE_SIDE_LEFT if hinge_side_str == "left"
             else door_pb2.DoorCommand.HINGE_SIDE_RIGHT)

    push_cmd = door_pb2.DoorCommand.AutoPushCommand()
    push_cmd.frame_name = frame_helpers.VISION_FRAME_NAME
    push_cmd.push_point_in_frame.CopyFrom(
        geometry_pb2.Vec3(x=push_point[0], y=push_point[1], z=push_point[2]))
    push_cmd.hinge_side = hinge

    door_command = door_pb2.DoorCommand.Request(auto_push_command=push_cmd)
    request = door_pb2.OpenDoorCommandRequest(door_command=door_command)

    print(f"\n  Sending AutoPushCommand...")
    print(f"    Point: [{push_point[0]:.2f}, {push_point[1]:.2f}, {push_point[2]:.2f}] (vision)")
    print(f"    Hinge: {hinge_side_str}")

    response = door_client.open_door(request)

    # Check initial response
    resp_status_names = {0: "UNKNOWN", 1: "OK", 2: "ROBOT_CMD_ERROR",
                         3: "DOOR_PLANE_NOT_DETECTED"}
    print(f"    Response: {resp_status_names.get(response.status, response.status)}")

    if response.status == 3:  # DOOR_PLANE_NOT_DETECTED
        print("    DOOR_PLANE_NOT_DETECTED — try different depth or position")
        return False

    if response.status != 1:  # not OK
        print(f"    Unexpected response status: {response.status}")
        return False

    # Poll feedback
    fb_request = door_pb2.OpenDoorFeedbackRequest()
    fb_request.door_command_id = response.door_command_id

    fb_names = {0: "UNKNOWN", 1: "COMPLETED", 2: "IN_PROGRESS",
                3: "STALLED", 4: "NOT_DETECTED"}

    t0 = time.time()
    timeout = 60.0
    while time.time() - t0 < timeout:
        elapsed = time.time() - t0
        fb = door_client.open_door_feedback(fb_request)
        status = fb.feedback.status
        print(f"    [{elapsed:.1f}s] {fb_names.get(status, status)}")

        if status == 1:  # COMPLETED
            print("\n  Door opened successfully!")
            return True
        if status in (3, 4):  # STALLED or NOT_DETECTED
            print(f"\n  Door command {fb_names.get(status, status)}")
            return False
        time.sleep(1.0)

    print(f"\n  Timed out after {timeout}s")
    return False


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Push-bar door opening via DoorService AutoPushCommand")
    parser.add_argument("--vlm", action="store_true",
                        help="Use VLM to auto-detect push bar location")
    parser.add_argument("--vision-xyz", type=str, default=None,
                        help="Direct vision-frame coords: 'x,y,z' (e.g. '1.65,2.68,0.81')")
    parser.add_argument("--hinge", choices=["left", "right"], default="left",
                        help="Hinge side when facing door (default: left)")
    parser.add_argument("--depth", type=float, default=0.5,
                        help="Estimated depth to door in meters for VLM mode (default: 0.5)")
    parser.add_argument("--camera", choices=["left", "right"], default="left",
                        help="Which front camera for VLM mode (default: left)")
    args = parser.parse_args()

    if not args.vlm and not args.vision_xyz:
        parser.error("Must specify --vlm or --vision-xyz")

    print("=" * 60)
    print("PUSH-BAR DOOR OPENING (AutoPushCommand)")
    print("=" * 60)
    print(f"\n  Make sure E-Stop is running!")
    print(f"  Hinge: {args.hinge}")
    if args.vlm:
        print(f"  VLM mode, depth={args.depth}m, camera={args.camera}")
    else:
        print(f"  Vision coords: {args.vision_xyz}")

    robot = connect_to_spot()

    # Clear stale keepalive policies
    from bosdyn.client.keepalive import KeepaliveClient, remove_all_policies
    try:
        kc = robot.ensure_client(KeepaliveClient.default_service_name)
        if kc.get_status().status:
            print("Clearing stale keepalive policies...")
            remove_all_policies(kc)
            time.sleep(2)
    except Exception:
        pass

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    keepalive = LeaseKeepAlive(lease_client)
    cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)

    print("Powering on...")
    robot.power_on(timeout_sec=20)
    clear_behavior_faults(robot, cmd_client)

    print("Ensuring Spot is standing...")
    blocking_stand(cmd_client)
    time.sleep(1.0)

    input("\nPress Enter when ready...")

    try:
        push_point = None

        if args.vision_xyz:
            # Direct vision-frame coordinates
            parts = args.vision_xyz.split(",")
            push_point = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            print(f"\n  Using vision coords: [{push_point[0]:.2f}, {push_point[1]:.2f}, {push_point[2]:.2f}]")

        elif args.vlm:
            # VLM detection
            image_client = robot.ensure_client(ImageClient.default_service_name)
            img_resp, px_raw, py_raw = vlm_find_push_bar_pixel(
                image_client, cmd_client, camera=args.camera)
            if img_resp is None:
                print("  VLM failed to detect push bar")
                return 1

            print(f"\n  Computing 3D push point (depth={args.depth}m)...")
            push_point = compute_3d_point_vision(img_resp, px_raw, py_raw, args.depth)
            if push_point is None:
                print("  Failed to compute 3D point")
                return 1

        # Send AutoPushCommand
        success = auto_push_door(robot, push_point, args.hinge)

        if not success:
            print("\n  Tip: try adjusting --depth (0.3-1.0) or repositioning Spot")

    except KeyboardInterrupt:
        print("\n\nAborted!")
        cmd_client.robot_command(RobotCommandBuilder.stop_command())
        time.sleep(0.5)
    finally:
        clear_behavior_faults(robot, cmd_client)
        keepalive.shutdown()

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
