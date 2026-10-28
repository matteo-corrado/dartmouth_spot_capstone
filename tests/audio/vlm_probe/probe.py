"""VLM probe: hallucination + latency.

Runs query_vlm() against:
 - Synthetic images (solid color, geometric shape, blank canvas)
 - Wikimedia public-domain photos (cat, chair, hallway)
 - Spot live camera frames (front/left/right/back)

Prints per-call latency and the VLM's reply. Manual eyeball check for
hallucinated objects.
"""
from __future__ import annotations

import io
import os
import sys
import time
import pathlib
import urllib.request

import requests
from PIL import Image, ImageDraw

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("SPOT_BRAIN_BACKEND", "llamacpp")

from src.voice_control.llm_brain import LLMBrain  # noqa: E402


PROBE_DIR = pathlib.Path(__file__).resolve().parent
WIKIMEDIA_PROBES = [
    # (filename, url, ground-truth caption used to score hallucinations)
    ("cat.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/640px-Cat03.jpg",
     "single grey/white tabby cat against a black background"),
    ("chair.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/9/97/Chair_01.jpg/640px-Chair_01.jpg",
     "single wooden chair on a plain background"),
    ("hallway.jpg",
     "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Corridor_in_school.jpg/640px-Corridor_in_school.jpg",
     "interior school hallway with lockers / doors"),
]


def synth_solid(color: tuple[int, int, int], size: int = 512) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (size, size), color).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def synth_circle() -> bytes:
    img = Image.new("RGB", (512, 512), (255, 255, 255))
    d = ImageDraw.Draw(img)
    d.ellipse([128, 128, 384, 384], fill=(0, 100, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def fetch_or_cache(name: str, url: str) -> bytes:
    path = PROBE_DIR / name
    if not path.exists():
        print(f"  [fetch] {url}")
        urllib.request.urlretrieve(url, path)
    return path.read_bytes()


def run_probe(brain: LLMBrain, label: str, image_bytes: bytes, question: str,
              ground_truth: str = "") -> dict:
    print(f"\n=== {label} ({len(image_bytes)} bytes JPEG) ===")
    if ground_truth:
        print(f"  ground truth: {ground_truth}")
    print(f"  Q: {question}")
    t0 = time.time()
    reply = brain.query_vlm(image_bytes, question)
    dt = time.time() - t0
    print(f"  A ({dt:.2f}s): {reply}")
    return {"label": label, "latency_s": dt, "reply": reply, "ground_truth": ground_truth}


def maybe_capture_spot(brain: LLMBrain, cameras: list[str]) -> list[dict]:
    try:
        from src.voice_control.spot_dispatch import capture_frame, ensure_spot_session
    except Exception as e:
        print(f"\n[spot] import failed: {e}")
        return []
    try:
        ensure_spot_session()
    except Exception as e:
        print(f"\n[spot] session unavailable — skipping live frames ({e})")
        return []
    results = []
    for cam in cameras:
        print(f"\n--- Spot live: {cam} ---")
        t_cap = time.time()
        frame = capture_frame(cam)
        cap_dt = time.time() - t_cap
        if not frame:
            print(f"  capture failed ({cap_dt:.2f}s)")
            continue
        out_path = PROBE_DIR / f"spot_{cam}.jpg"
        out_path.write_bytes(frame)
        print(f"  saved -> {out_path} ({len(frame)} bytes, capture {cap_dt:.2f}s)")
        results.append(
            run_probe(brain, f"spot_{cam}", frame, "What do you see?",
                      ground_truth="live Spot camera frame")
        )
    return results


def main():
    brain = LLMBrain()
    results = []

    print("######## Synthetic ########")
    results.append(run_probe(brain, "solid_red", synth_solid((220, 30, 30)),
                             "What do you see?",
                             ground_truth="solid red — NO objects, NO scene"))
    results.append(run_probe(brain, "solid_white", synth_solid((255, 255, 255)),
                             "What do you see?",
                             ground_truth="solid white — NO objects, NO scene"))
    results.append(run_probe(brain, "blue_circle_on_white", synth_circle(),
                             "What do you see?",
                             ground_truth="one blue circle on white background"))

    print("\n######## Wikimedia ########")
    for name, url, truth in WIKIMEDIA_PROBES:
        try:
            data = fetch_or_cache(name, url)
        except Exception as e:
            print(f"\n[fetch] {name} failed: {e}")
            continue
        label = name.replace(".jpg", "")
        results.append(run_probe(brain, label, data, "What do you see?", ground_truth=truth))

    print("\n######## Spot live ########")
    results.extend(maybe_capture_spot(brain, ["front", "left", "right", "back"]))

    print("\n\n######## Latency summary ########")
    for r in results:
        print(f"  {r['latency_s']:5.2f}s  {r['label']}")
    if results:
        lats = [r["latency_s"] for r in results]
        print(f"  --- mean {sum(lats)/len(lats):.2f}s | min {min(lats):.2f}s | max {max(lats):.2f}s")


if __name__ == "__main__":
    main()
