#!/usr/bin/env python3
"""Test YOLO detection on Spot's camera feed.

Captures a frame from Spot's front camera, runs YOLO-World (open-vocabulary)
and YOLOv8n (person detection), prints results + steering math.

Usage:
    python scripts/test_visual_nav.py                          # detect "person"
    python scripts/test_visual_nav.py --target "red chair"     # detect specific object
    python scripts/test_visual_nav.py --save                   # save annotated image
"""

import sys
import pathlib
import argparse
import time
import io

project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from PIL import Image


def main():
    parser = argparse.ArgumentParser(description="Test YOLO detection on Spot camera")
    parser.add_argument("--target", default="person", help="Object to search for with YOLO-World")
    parser.add_argument("--save", action="store_true", help="Save annotated image to /tmp/")
    parser.add_argument("--no-spot", action="store_true", help="Use a test image instead of Spot camera")
    args = parser.parse_args()

    # --- Step 1: Get an image ---
    if args.no_spot:
        # Generate a simple test image
        print("[Test] Using synthetic test image (no Spot)")
        img = Image.new("RGB", (640, 480), (128, 128, 128))
    else:
        print("[Test] Connecting to Spot and capturing frame...")
        from src.session import spot_session
        from bosdyn.client.image import ImageClient

        with spot_session(stand_on_enter=False, sit_on_exit=False) as session:
            robot = session["robot"]
            img_client = robot.ensure_client(ImageClient.default_service_name)
            resps = img_client.get_image_from_sources(["frontleft_fisheye_image"])
            if not resps:
                print("[Test] ERROR: No image returned from camera")
                return
            data = resps[0].shot.image.data
            img = Image.open(io.BytesIO(data))
            print(f"[Test] Raw image: {img.size[0]}x{img.size[1]}")

            # Rotate 90° CW (fisheye is mounted sideways)
            img = img.transpose(Image.Transpose.ROTATE_270)
            print(f"[Test] Rotated image: {img.size[0]}x{img.size[1]}")

    w, h = img.size

    # --- Step 2: YOLO-World (open-vocabulary) ---
    print(f"\n[Test] Loading YOLO-World and searching for: '{args.target}'")
    t0 = time.time()
    from ultralytics import YOLOWorld
    model_world = YOLOWorld("yolov8s-worldv2.pt")
    load_time = time.time() - t0
    print(f"[Test] YOLO-World loaded in {load_time:.1f}s")

    model_world.set_classes([args.target])
    t0 = time.time()
    results_world = model_world.predict(img, device="cpu", conf=0.25, verbose=False)
    infer_time = time.time() - t0
    print(f"[Test] YOLO-World inference: {infer_time:.2f}s")

    if results_world and len(results_world[0].boxes) > 0:
        boxes = results_world[0].boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].item()
            cx = (bbox[0] + bbox[2]) / 2.0
            bh = bbox[3] - bbox[1]
            offset = (cx - w / 2.0) / (w / 2.0)
            yaw = -1.0 * offset
            frac = bh / h
            print(f"  [{i}] bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}] "
                  f"conf={conf:.2f} offset={offset:+.2f} yaw={yaw:+.2f}rad "
                  f"size={frac:.0%}")
        print(f"  -> Best detection: conf={boxes.conf.max().item():.2f}")
    else:
        print(f"  -> No '{args.target}' detected")

    # --- Step 3: YOLOv8n (person detection) ---
    print(f"\n[Test] Loading YOLOv8n for person detection...")
    t0 = time.time()
    from ultralytics import YOLO
    model_person = YOLO("yolov8n.pt")
    load_time = time.time() - t0
    print(f"[Test] YOLOv8n loaded in {load_time:.1f}s")

    t0 = time.time()
    results_person = model_person.predict(img, device="cpu", classes=[0], conf=0.25, verbose=False)
    infer_time = time.time() - t0
    print(f"[Test] YOLOv8n inference: {infer_time:.2f}s")

    if results_person and len(results_person[0].boxes) > 0:
        boxes = results_person[0].boxes
        for i in range(len(boxes)):
            bbox = boxes.xyxy[i].cpu().numpy()
            conf = boxes.conf[i].item()
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            print(f"  [{i}] bbox=[{bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f}] "
                  f"conf={conf:.2f} area={area:.0f}px²")
        print(f"  -> {len(boxes)} person(s) detected")
    else:
        print("  -> No persons detected")

    # --- Step 4: Save annotated image ---
    if args.save:
        out_path = "/tmp/visual_nav_test.jpg"
        # Use ultralytics built-in annotation
        if results_world and len(results_world[0].boxes) > 0:
            annotated = Image.fromarray(results_world[0].plot())
            annotated.save(out_path)
            print(f"\n[Test] Annotated image saved to {out_path}")
            print(f"  scp spotdog@<jetson-ip>:{out_path} . && open visual_nav_test.jpg")
        else:
            img.save(out_path)
            print(f"\n[Test] Image saved to {out_path} (no detections to annotate)")

    print("\n[Test] Done!")


if __name__ == "__main__":
    main()
