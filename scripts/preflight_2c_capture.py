#!/usr/bin/env python3
"""Stage 2C preflight: capture one frame per camera, save to /tmp/preflight_2c.

Read-only. No lease, no power-on, no motion. Body fisheyes are rotated
ROTATE_270 to match production (visual_nav._capture_with_response). Used to
(a) eyeball fisheye warp severity for the undistort decision and (b) provide
real greyscale frames to feed the GPU YOLO sidecar.
"""
import io
import sys
from pathlib import Path

sys.path.insert(0, ".")
from PIL import Image

from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient
from src.session import quick_robot

BODY = ["frontleft_fisheye_image", "frontright_fisheye_image",
        "left_fisheye_image", "right_fisheye_image", "back_fisheye_image"]
HAND = "hand_color_image"
OUT = Path("/tmp/preflight_2c")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    robot = quick_robot("preflight-2c-capture")
    print(f"Connected: {robot._name}")
    client = robot.ensure_client(ImageClient.default_service_name)

    reqs = [image_pb2.ImageRequest(image_source_name=s, quality_percent=90,
                                   image_format=image_pb2.Image.FORMAT_JPEG)
            for s in BODY + [HAND]]
    resps = {r.source.name: r for r in client.get_image(reqs)}

    for name, resp in resps.items():
        pil = Image.open(io.BytesIO(resp.shot.image.data))
        if name in BODY:
            pil = pil.transpose(Image.Transpose.ROTATE_270)  # match prod
        path = OUT / f"{name}.png"
        pil.save(path)
        print(f"  {name}: {pil.size} -> {path}")
    print(f"saved {len(resps)} frames to {OUT}")


if __name__ == "__main__":
    main()
