"""Measure vision snapshot latency on the robot: camera capture + YOLO person detection.

Read-only: takes no lease and sends no motion. Follow must not be used near people
until the 2-camera median is below ~0.5 s (docs/project/todo/vision-actions.md, 1.1).
Stop the demo first, or the numbers include its GPU and camera load.

Usage:
    python scripts/vision_snapshot_latency.py [--host 192.168.50.3] [--runs 8]
        [--env-dir /home/spotdog/spot/dartmouth_spot_capstone]
"""
import argparse
import pathlib
import statistics
import sys
import time

from dotenv import load_dotenv

REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

MAIN_CHECKOUT = "/home/spotdog/spot/dartmouth_spot_capstone"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--host", default="192.168.50.3")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--env-dir", default=MAIN_CHECKOUT,
                    help="folder holding the robot credentials file (worktrees have none)")
    args = ap.parse_args()

    load_dotenv(pathlib.Path(args.env_dir) / ".env")
    from src.session import quick_robot
    from src.voice_control.vision import cameras, runtime, world

    robot = quick_robot("vision-latency", hostname=args.host)
    model = runtime.load_model("yolo26l-seg")
    print("provider", model.provider)

    front = ["frontleft_fisheye_image", "frontright_fisheye_image"]
    for sources in (front, front + ["left_fisheye_image"], list(cameras.BODY_SOURCES)):
        cap, det, n = [], [], 0
        for _ in range(args.runs):
            t0 = time.time()
            frames = cameras.capture(robot, sources, with_depth=True)
            t1 = time.time()
            dets = world.detect_frames(frames, labels={"person"})
            t2 = time.time()
            cap.append(t1 - t0)
            det.append(t2 - t1)
            n = len(dets)
        tot = [a + b for a, b in zip(cap, det)]
        print(f"{len(sources)} cams: capture med {statistics.median(cap) * 1000:.0f} ms, "
              f"detect med {statistics.median(det) * 1000:.0f} ms, "
              f"total med {statistics.median(tot) * 1000:.0f} ms, max {max(tot) * 1000:.0f} ms, "
              f"persons {n}")


if __name__ == "__main__":
    main()
