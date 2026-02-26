#!/usr/bin/env python3
"""Calibrate door handle/hinge position for DoorService-based door opening.

Captures front fisheye images from Spot, lets the user identify the push bar
handle and hinge positions, and saves calibration to door_config.json.

Modes:
  capture      Capture fisheye images and save to door_calibration/ for inspection
  set          Set handle/hinge pixel coords from inspecting the saved images
  interactive  All-in-one GUI mode (requires OpenCV with display)

Usage:
  # Step 1: Position robot ~1-2m from door, facing it.
  python scripts/calibrate_door.py capture

  # Step 2: Open door_calibration/side_by_side.jpg (or individual images).
  #          Note the handle pixel position and which side the hinge is on.
  #
  #   In the side-by-side image:
  #     LEFT half  = frontright camera
  #     RIGHT half = frontleft camera
  #
  #   Provide the pixel coords from whichever half shows the door best.

  python scripts/calibrate_door.py set \\
      --handle-x 400 --handle-y 250 \\
      --hinge left \\
      --camera left

  # Or all-in-one (requires display + OpenCV):
  python scripts/calibrate_door.py interactive
"""

import sys
import pathlib
import json
import math
import argparse
import time
import os

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

CONFIG_PATH = project_root / "door_config.json"
CALIB_DIR = project_root / "door_calibration"


def connect():
    """Connect to Spot and return (robot, cmd_client, image_client, keepalive)."""
    from dotenv import load_dotenv
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
    from bosdyn.client.image import ImageClient

    load_dotenv(project_root / ".env")
    spot_ip = os.getenv("BOSDYN_ROBOT_IP", "192.168.80.3")
    spot_user = os.getenv("BOSDYN_CLIENT_USERNAME", "")
    spot_pass = os.getenv("BOSDYN_CLIENT_PASSWORD")
    if not spot_pass:
        print("ERROR: Set BOSDYN_CLIENT_PASSWORD in .env")
        sys.exit(1)

    print(f"Connecting to Spot at {spot_ip}...")
    sdk = create_standard_sdk("DoorCalibration")
    robot = sdk.create_robot(spot_ip)
    robot.authenticate(spot_user, spot_pass)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    keepalive = LeaseKeepAlive(lease_client)
    cmd_client = robot.ensure_client(RobotCommandClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    # Power on and stand if needed
    from bosdyn.client.robot_state import RobotStateClient
    state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_state = state_client.get_robot_state()
    is_powered = (robot_state.power_state.motor_power_state
                  == robot_state.power_state.STATE_ON)
    if not is_powered:
        print("Powering on...")
        robot.power_on(timeout_sec=20)
    print("Standing...")
    blocking_stand(cmd_client, timeout_sec=10)

    return robot, cmd_client, image_client, keepalive


def capture_images(cmd_client, image_client):
    """Pitch robot up and capture front fisheye images. Returns dict of responses."""
    from bosdyn.client.robot_command import RobotCommandBuilder
    from bosdyn import geometry

    print("Pitching robot up to see the door...")
    pitch_cmd = RobotCommandBuilder.synchro_stand_command(
        footprint_R_body=geometry.EulerZXY(pitch=-0.4, roll=0.0, yaw=0.0)
    )
    cmd_client.robot_command(pitch_cmd)
    time.sleep(2.0)

    sources = ["frontleft_fisheye_image", "frontright_fisheye_image"]
    print(f"Capturing from {sources}...")
    responses = image_client.get_image_from_sources(sources)

    # Reset pitch
    cmd_client.robot_command(RobotCommandBuilder.synchro_stand_command())

    return {resp.source.name: resp for resp in responses}


def decode_image(img_response):
    """Decode an image response to a numpy array."""
    import numpy as np

    raw = img_response.shot.image.data
    fmt = img_response.shot.image.format  # 1=RAW, 2=JPEG

    from bosdyn.api import image_pb2 as img_pb2

    if fmt == img_pb2.Image.FORMAT_JPEG:
        try:
            import cv2
            arr = np.frombuffer(raw, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        except ImportError:
            from PIL import Image
            import io
            return np.array(Image.open(io.BytesIO(raw)).convert("L"))
    else:
        h = img_response.shot.image.rows
        w = img_response.shot.image.cols
        return np.frombuffer(raw, dtype=np.uint8).reshape((h, w))


def rotate_90cw(img):
    """Rotate image 90 degrees clockwise."""
    import numpy as np
    return np.rot90(img, k=-1)


# ── capture mode ──────────────────────────────────────────────────────────

def do_capture():
    robot, cmd_client, image_client, keepalive = connect()
    try:
        image_dict = capture_images(cmd_client, image_client)
        import numpy as np

        CALIB_DIR.mkdir(exist_ok=True)

        # Save individual rotated images
        for name, resp in image_dict.items():
            img = decode_image(resp)
            rotated = rotate_90cw(img)
            path = CALIB_DIR / f"{name}.jpg"
            try:
                import cv2
                cv2.imwrite(str(path), rotated)
            except ImportError:
                from PIL import Image
                Image.fromarray(rotated).save(str(path))
            print(f"  Saved: {path} ({rotated.shape[1]}x{rotated.shape[0]})")

        # Save side-by-side: frontright on LEFT, frontleft on RIGHT
        imgs = []
        for name in ["frontright_fisheye_image", "frontleft_fisheye_image"]:
            imgs.append(rotate_90cw(decode_image(image_dict[name])))
        sbs = np.hstack(imgs)
        sbs_path = CALIB_DIR / "side_by_side.jpg"
        try:
            import cv2
            cv2.imwrite(str(sbs_path), sbs)
        except ImportError:
            from PIL import Image
            Image.fromarray(sbs).save(str(sbs_path))
        print(f"  Saved side-by-side: {sbs_path} ({sbs.shape[1]}x{sbs.shape[0]})")

        half_w = imgs[0].shape[1]
        print(f"\n  Side-by-side layout:")
        print(f"    LEFT  half (x: 0 to {half_w-1})   = frontright camera")
        print(f"    RIGHT half (x: {half_w} to {sbs.shape[1]-1}) = frontleft camera")
        print(f"\n  Next steps:")
        print(f"    1. Open {sbs_path}")
        print(f"    2. Note the (x, y) pixel of the push bar handle")
        print(f"    3. Note which side the door hinge is on (left or right of handle)")
        print(f"    4. Note which half of the image the handle is in (determines camera)")
        print(f"    5. Run:")
        print(f"       python scripts/calibrate_door.py set \\")
        print(f"         --handle-x <X> --handle-y <Y> \\")
        print(f"         --hinge left|right \\")
        print(f"         --camera left|right")
        print(f"\n    --camera left  = frontleft  (RIGHT half of side-by-side)")
        print(f"    --camera right = frontright (LEFT  half of side-by-side)")
        print(f"\n    If using coords from the side-by-side image, add --from-sbs")
    finally:
        keepalive.shutdown()


# ── set mode ──────────────────────────────────────────────────────────────

def do_set(args):
    """Save calibration from user-provided pixel coordinates."""
    camera_source = ("frontleft_fisheye_image" if args.camera == "left"
                     else "frontright_fisheye_image")

    hx = args.handle_x
    hy = args.handle_y

    # If coords are from side-by-side, convert to single-image coords
    if args.from_sbs:
        # Try to get actual dimensions from saved images
        sbs_path = CALIB_DIR / "side_by_side.jpg"
        half_w = 640  # default
        if sbs_path.exists():
            try:
                import cv2
                sbs = cv2.imread(str(sbs_path))
                half_w = sbs.shape[1] // 2
            except ImportError:
                try:
                    from PIL import Image
                    sbs = Image.open(sbs_path)
                    half_w = sbs.width // 2
                except ImportError:
                    pass

        if args.camera == "left":
            # frontleft = right half of side-by-side
            hx = hx - half_w
        # frontright = left half, no adjustment

    config = {
        "camera_source": camera_source,
        "handle_pixel_x_rotated": float(hx),
        "handle_pixel_y_rotated": float(hy),
        "hinge_side": args.hinge,
    }

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Calibration saved to {CONFIG_PATH}")
    print(f"  Camera: {camera_source}")
    print(f"  Handle (rotated img coords): ({hx:.0f}, {hy:.0f})")
    print(f"  Hinge side: {args.hinge}")


# ── interactive mode ──────────────────────────────────────────────────────

def do_interactive():
    """GUI click-based calibration (requires OpenCV with display)."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        print("ERROR: OpenCV required for interactive mode.")
        print("  pip install opencv-python")
        print("  Or use 'capture' + 'set' modes instead.")
        sys.exit(1)

    robot, cmd_client, image_client, keepalive = connect()
    try:
        image_dict = capture_images(cmd_client, image_client)

        # Decode and rotate
        imgs = {}
        for name in ["frontright_fisheye_image", "frontleft_fisheye_image"]:
            img = decode_image(image_dict[name])
            imgs[name] = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        # Side-by-side: frontright LEFT, frontleft RIGHT
        sbs = np.hstack([imgs["frontright_fisheye_image"],
                         imgs["frontleft_fisheye_image"]])
        display = cv2.cvtColor(sbs, cv2.COLOR_GRAY2BGR)
        half_w = imgs["frontright_fisheye_image"].shape[1]

        clicks = []

        def on_click(event, x, y, flags, param):
            if event != cv2.EVENT_LBUTTONDOWN:
                return
            clicks.append((x, y))
            # Draw marker
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            if len(clicks) == 1:
                cv2.putText(display, "HANDLE", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                print(f"  Handle: ({x}, {y}) — now click on the HINGE...")
            elif len(clicks) == 2:
                cv2.putText(display, "HINGE", (x + 8, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                print(f"  Hinge: ({x}, {y})")

        cv2.namedWindow("Click HANDLE, then HINGE", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Click HANDLE, then HINGE", on_click)

        print("\nClick on the push bar HANDLE position...")
        while len(clicks) < 2:
            cv2.imshow("Click HANDLE, then HINGE", display)
            key = cv2.waitKey(100)
            if key == 27:  # ESC
                print("Cancelled.")
                cv2.destroyAllWindows()
                return
        time.sleep(0.5)
        cv2.destroyAllWindows()

        handle_sbs = clicks[0]
        hinge_sbs = clicks[1]

        # Determine camera source from which half handle is in
        if handle_sbs[0] >= half_w:
            camera_source = "frontleft_fisheye_image"
            local_x = handle_sbs[0] - half_w
        else:
            camera_source = "frontright_fisheye_image"
            local_x = handle_sbs[0]
        local_y = handle_sbs[1]

        # Hinge side: if handle is left of hinge, hinge is on right
        hinge_side = "right" if handle_sbs[0] < hinge_sbs[0] else "left"

        config = {
            "camera_source": camera_source,
            "handle_pixel_x_rotated": float(local_x),
            "handle_pixel_y_rotated": float(local_y),
            "hinge_side": hinge_side,
        }

        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

        print(f"\nCalibration saved to {CONFIG_PATH}")
        print(f"  Camera: {camera_source}")
        print(f"  Handle (rotated): ({local_x}, {local_y})")
        print(f"  Hinge side: {hinge_side}")

    finally:
        keepalive.shutdown()


# ── main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate door handle position for DoorService-based opening")
    sub = parser.add_subparsers(dest="mode")

    sub.add_parser("capture", help="Capture fisheye images and save for inspection")

    sp = sub.add_parser("set", help="Set handle coords from inspected images")
    sp.add_argument("--handle-x", type=float, required=True,
                    help="Handle X pixel in rotated single-camera image")
    sp.add_argument("--handle-y", type=float, required=True,
                    help="Handle Y pixel in rotated single-camera image")
    sp.add_argument("--hinge", choices=["left", "right"], required=True,
                    help="Which side the door hinge is on (facing the door)")
    sp.add_argument("--camera", choices=["left", "right"], default="left",
                    help="Camera: left=frontleft, right=frontright (default: left)")
    sp.add_argument("--from-sbs", action="store_true",
                    help="Coords are from the side-by-side image (auto-adjusts)")

    sub.add_parser("interactive",
                   help="Interactive GUI mode (requires OpenCV + display)")

    args = parser.parse_args()
    if args.mode == "capture":
        do_capture()
    elif args.mode == "set":
        do_set(args)
    elif args.mode == "interactive":
        do_interactive()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
