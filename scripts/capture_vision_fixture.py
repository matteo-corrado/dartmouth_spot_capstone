#!/usr/bin/env python3
"""Save one serialized GetImageResponse (all body sources + hand) for offline tests.

Read-only: no lease, no power, no motion. Output belongs under
tests/fixtures/vision/private/ because people may be in view.

Usage: python scripts/capture_vision_fixture.py --out PATH [--host 192.168.50.3] [--with-depth]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bosdyn.api import image_pb2  # noqa: E402
from bosdyn.client.image import ImageClient, build_image_request  # noqa: E402

from src.session import quick_robot  # noqa: E402
from src.voice_control.vision.cameras import BODY_SOURCES, DEPTH_SOURCE, HAND_SOURCE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--host", default="192.168.50.3")
    ap.add_argument("--with-depth", action="store_true")
    a = ap.parse_args()

    robot = quick_robot("vision_fixture", hostname=a.host)
    client = robot.ensure_client(ImageClient.default_service_name)
    sources = list(BODY_SOURCES) + [HAND_SOURCE]
    reqs = [build_image_request(s, image_format=image_pb2.Image.FORMAT_JPEG) for s in sources]
    if a.with_depth:
        reqs += [build_image_request(DEPTH_SOURCE[s],
                                     pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
                                     image_format=image_pb2.Image.FORMAT_RAW)
                 for s in sources]
    resps = client.get_image(reqs)
    msg = image_pb2.GetImageResponse(image_responses=resps)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(msg.SerializeToString())
    for r in resps:
        print(r.source.name, r.shot.image.cols, "x", r.shot.image.rows, "status", r.status)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
