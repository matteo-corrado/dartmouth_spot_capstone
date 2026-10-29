"""THROWAWAY SPIKE: time a full look-around (all cameras -> upright -> VLM).

Read-only: no lease, no motion. Run from repo root:
  spot-env/bin/python <this> --host 192.168.50.3
"""
import argparse, base64, io, pathlib, sys, time
import requests
from dotenv import load_dotenv

REPO = pathlib.Path("/home/spotdog/spot/dartmouth_spot_capstone")
sys.path.insert(0, str(REPO))
load_dotenv(REPO / ".env")
from PIL import Image
from bosdyn.api import image_pb2
from bosdyn.client.image import ImageClient
from src.session import quick_robot

OUT = pathlib.Path(__file__).parent / "lookaround"
OUT.mkdir(exist_ok=True)
# BD get_image.py ROTATION_ANGLE (degrees CCW)
ROT = {"back_fisheye_image": 0, "frontleft_fisheye_image": -78,
       "frontright_fisheye_image": -102, "left_fisheye_image": 0,
       "right_fisheye_image": 180, "hand_color_image": 0}
URL = "http://localhost:11435/v1/chat/completions"
SYSTEM = ("You are Spot, a robot. These are your own camera views right now. "
          "Answer in first person, never mention images or cameras.")


def vlm(content, max_tokens=200):
    t = time.time()
    r = requests.post(URL, json={"messages": [{"role": "system", "content": SYSTEM},
                                              {"role": "user", "content": content}],
                                 "max_tokens": max_tokens, "temperature": 0.5,
                                 "chat_template_kwargs": {"enable_thinking": False}}, timeout=180)
    r.raise_for_status()
    j = r.json()
    return time.time() - t, j["choices"][0]["message"]["content"].strip(), j.get("usage", {})


def img_part(jpeg):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.50.3")
    a = ap.parse_args()
    robot = quick_robot("lookaround-spike", hostname=a.host)
    ic = robot.ensure_client(ImageClient.default_service_name)
    reqs = [image_pb2.ImageRequest(image_source_name=s, quality_percent=75,
                                   image_format=image_pb2.Image.FORMAT_JPEG) for s in ROT]
    t = time.time()
    resps = ic.get_image(reqs)
    print(f"capture {len(resps)} sources: {time.time()-t:.2f}s")

    up = {}
    t = time.time()
    for r in resps:
        s = r.source.name
        im = Image.open(io.BytesIO(r.shot.image.data)).convert("RGB")
        im = im.rotate(ROT[s], expand=True)
        buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
        up[s] = buf.getvalue()
        im.save(OUT / f"{s}.jpg")
        print(f"  {s}: raw {r.shot.image.cols}x{r.shot.image.rows} -> upright {im.size}")
    print(f"rotate+encode: {time.time()-t:.2f}s")

    # contact sheet, each tile labelled by source
    tiles = [Image.open(OUT / f"{s}.jpg") for s in up]
    h = 360
    tiles = [t_.resize((int(t_.width * h / t_.height), h)) for t_ in tiles]
    sheet = Image.new("RGB", (sum(t_.width for t_ in tiles), h + 20), "white")
    from PIL import ImageDraw
    d = ImageDraw.Draw(sheet); x = 0
    for s, t_ in zip(up, tiles):
        sheet.paste(t_, (x, 20)); d.text((x + 4, 4), s, fill="black"); x += t_.width
    sheet.save(OUT / "contact_sheet.jpg")
    print("saved", OUT / "contact_sheet.jpg")

    # A: one VLM call per camera
    total = 0
    for s, jpg in up.items():
        dt, txt, u = vlm([{"type": "text", "text": "In one sentence, list the main objects you see."}, img_part(jpg)], 80)
        total += dt
        print(f"[A] {s}: {dt:.1f}s prompt_tok={u.get('prompt_tokens')} -> {txt}")
    print(f"[A] per-camera sequential total: {total:.1f}s")

    # B: one call, all images, labelled
    parts = [{"type": "text", "text": "Each view is labelled. Briefly say what you see in each direction, 2-3 sentences total."}]
    for s, jpg in up.items():
        parts += [{"type": "text", "text": f"View: {s}"}, img_part(jpg)]
    dt, txt, u = vlm(parts, 200)
    print(f"[B] single call, {len(up)} images: {dt:.1f}s prompt_tok={u.get('prompt_tokens')} -> {txt}")


if __name__ == "__main__":
    main()
