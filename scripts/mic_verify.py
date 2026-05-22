#!/usr/bin/env python3
"""Stage 2A T12 — mic-verify end-to-end.

Replays tests/audio/*_<room>.wav through the live pipeline (wake detector +
ASR gRPC server) and asserts:

  - hey_spot_*.wav         → wake detector fires
  - adv_*.wav, idle_*.wav  → wake detector does NOT fire
  - command clips          → ASR transcript covers expected keywords
  - hey_spot_stand_up,     → ASR transcript covers expected keywords
    hey_spot_walk_forward    (single-breath path)

Writes JSON report to /tmp/stage2a_mic_verify_<timestamp>.json and exits
non-zero on any failure. Designed to compare gates against the T0 baseline
snapshot (/tmp/stage2a_preflight_*.json) on a per-run basis.

Usage:
    spot-env/bin/python scripts/mic_verify.py --room lab
"""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import grpc
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "voice_control"))

from asr_pb2 import StreamingRequest, StreamingConfig, AudioChunk  # noqa: E402
from asr_pb2_grpc import ASRStub  # noqa: E402
from src.voice_control.wake import make_wake_detector  # noqa: E402

SAMPLE_RATE = 16000
FRAME_MS = 20
BYTES_PER_FRAME = SAMPLE_RATE * 2 * FRAME_MS // 1000  # int16 mono
AUDIO_DIR = PROJECT_ROOT / "tests" / "audio"

# Expected transcript keywords per clip (lowercased word sets).
# Command clips: every listed word must appear in the transcript.
EXPECTED_KEYWORDS = {
    "stand_up":                       ["stand", "up"],
    "sit_down":                       ["sit", "down"],
    "walk_forward_1m":                ["walk", "forward"],
    "walk_forward_two_meters":        ["walk", "forward"],
    "turn_left_90":                   ["turn", "left"],
    "turn_around":                    ["turn", "around"],
    "go_to_the_kitchen":              ["go", "kitchen"],
    "come_here":                      ["come", "here"],
    "follow_me":                      ["follow", "me"],
    "stop":                           ["stop"],
    "battery_status":                 ["battery"],
    "what_do_you_see":                ["what", "see"],
    "describe_surroundings":          ["describe"],
    "tell_me_about_dartmouth_hall":   ["dartmouth"],
    "find_a_chair":                   ["find", "chair"],
    "hey_spot_stand_up":              ["stand", "up"],
    "hey_spot_walk_forward":          ["walk", "forward"],
}

WAKE_CLIPS = [
    "hey_spot_close", "hey_spot_mid", "hey_spot_far",
    "hey_spot_stand_up", "hey_spot_walk_forward",
]
NO_WAKE_CLIPS = [
    "adv_spot_the_difference", "adv_on_the_spot", "adv_x_marks_the_spot",
    "adv_hey_dog", "adv_hey_stop", "adv_hot_spot",
    "idle_01", "idle_02", "idle_03", "idle_04", "idle_05",
]

WORD_RE = re.compile(r"[a-z0-9]+")


def load_wav(path: Path) -> bytes:
    with wave.open(str(path), "rb") as w:
        if w.getframerate() != SAMPLE_RATE or w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"{path.name}: expected 16kHz mono PCM16")
        return w.readframes(w.getnframes())


def normalize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def run_wake(detector, pcm: bytes) -> bool:
    """Feed PCM through detector frame-by-frame; return True if it ever fired."""
    detector.reset() if hasattr(detector, "reset") else None
    fired = False
    for off in range(0, len(pcm), BYTES_PER_FRAME):
        frame = pcm[off:off + BYTES_PER_FRAME]
        if len(frame) < BYTES_PER_FRAME:
            break
        if detector.process_frame(frame):
            fired = True
    return fired


def run_asr(stub, pcm: bytes) -> tuple[str, float]:
    def gen():
        yield StreamingRequest(config=StreamingConfig(
            language_code="en", sample_rate_hz=SAMPLE_RATE, enable_punctuation=True,
        ))
        for off in range(0, len(pcm), BYTES_PER_FRAME):
            yield StreamingRequest(audio=AudioChunk(pcm16=pcm[off:off + BYTES_PER_FRAME]))

    t0 = time.time()
    transcript = ""
    for resp in stub.StreamingRecognize(gen()):
        if resp.is_final:
            transcript = resp.transcript.strip()
            break
    return transcript, time.time() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="lab", help="Room suffix used at record time")
    ap.add_argument("--asr", default="localhost:50055", help="ASR gRPC endpoint")
    args = ap.parse_args()

    sfx = f"_{args.room}"
    print(f"[mic-verify] room={args.room} corpus={AUDIO_DIR}\n")

    detector = make_wake_detector()
    if not detector.is_available():
        print("ERROR: wake detector unavailable")
        return 2

    channel = grpc.insecure_channel(args.asr)
    stub = ASRStub(channel)

    results = {"wake_should_fire": [], "wake_should_not_fire": [], "asr": []}
    failures = 0

    # 1) Wake clips: detector must fire.
    print("=== Wake (expect FIRE) ===")
    for name in WAKE_CLIPS:
        path = AUDIO_DIR / f"{name}{sfx}.wav"
        if not path.exists():
            print(f"  {name}: MISSING — {path.name}")
            failures += 1
            continue
        fired = run_wake(detector, load_wav(path))
        ok = fired
        print(f"  {name}: {'OK' if ok else 'FAIL'} (fired={fired})")
        results["wake_should_fire"].append({"clip": name, "fired": fired, "ok": ok})
        if not ok:
            failures += 1

    # 2) Non-wake clips: detector must NOT fire.
    print("\n=== Adversarial + idle (expect NO fire) ===")
    for name in NO_WAKE_CLIPS:
        path = AUDIO_DIR / f"{name}{sfx}.wav"
        if not path.exists():
            print(f"  {name}: MISSING — {path.name}")
            failures += 1
            continue
        fired = run_wake(detector, load_wav(path))
        ok = not fired
        print(f"  {name}: {'OK' if ok else 'FAIL'} (fired={fired})")
        results["wake_should_not_fire"].append({"clip": name, "fired": fired, "ok": ok})
        if not ok:
            failures += 1

    # 3) ASR clips: transcript must cover expected keywords.
    print("\n=== ASR keyword coverage ===")
    latencies = []
    for name, expected in EXPECTED_KEYWORDS.items():
        path = AUDIO_DIR / f"{name}{sfx}.wav"
        if not path.exists():
            print(f"  {name}: MISSING — {path.name}")
            failures += 1
            continue
        transcript, latency_s = run_asr(stub, load_wav(path))
        words = set(normalize(transcript))
        missing = [w for w in expected if w not in words]
        ok = not missing
        print(f"  {name}: {'OK' if ok else 'FAIL'} ({latency_s*1000:.0f}ms) '{transcript}'"
              f"{' missing=' + str(missing) if missing else ''}")
        results["asr"].append({
            "clip": name, "transcript": transcript, "expected": expected,
            "missing": missing, "latency_s": latency_s, "ok": ok,
        })
        latencies.append(latency_s)
        if not ok:
            failures += 1

    # Summary + dump.
    median_ms = float(np.median(latencies) * 1000) if latencies else 0.0
    p95_ms = float(np.percentile(latencies, 95) * 1000) if latencies else 0.0
    print(f"\n=== Summary ===")
    print(f"  wake fire   : {sum(1 for r in results['wake_should_fire'] if r['ok'])}/{len(results['wake_should_fire'])}")
    print(f"  wake reject : {sum(1 for r in results['wake_should_not_fire'] if r['ok'])}/{len(results['wake_should_not_fire'])}")
    print(f"  ASR pass    : {sum(1 for r in results['asr'] if r['ok'])}/{len(results['asr'])}")
    print(f"  ASR latency : median {median_ms:.0f}ms / p95 {p95_ms:.0f}ms")
    print(f"  failures    : {failures}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = Path(f"/tmp/stage2a_mic_verify_{stamp}.json")
    out_path.write_text(json.dumps({
        "timestamp": stamp,
        "room": args.room,
        "asr_latency_ms": {"median": median_ms, "p95": p95_ms},
        "failures": failures,
        "results": results,
    }, indent=2))
    print(f"\nReport: {out_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
