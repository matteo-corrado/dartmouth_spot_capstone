#!/usr/bin/env python3
"""XVF3800 probe: live DoA, clap calibration, LED write-rate limits.

Usage:
  spot-env/bin/python scripts/xvf_probe.py live    # deg/speech at 10 Hz (Ctrl-C to stop)
  spot-env/bin/python scripts/xvf_probe.py clap    # mic_yaw_offset_deg / mic_yaw_sign
  spot-env/bin/python scripts/xvf_probe.py leds    # cycle effects, measure set_ring writes/s

Runtime writes only; never saves config or reboots the device.
"""
import argparse
import math
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.voice_control.mic_array import MicArray  # noqa: E402

CLAP_SECONDS = 3.0
SIGN_TOLERANCE_DEG = 45.0


def circ_dist(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def circ_median_deg(degs):
    """Median of angles after unwrapping around their circular mean."""
    if not degs:
        return None
    mean = math.degrees(math.atan2(
        sum(math.sin(math.radians(d)) for d in degs),
        sum(math.cos(math.radians(d)) for d in degs)))
    offsets = [((d - mean + 180) % 360) - 180 for d in degs]
    return (mean + statistics.median(offsets)) % 360


def cmd_live(mic):
    try:
        while True:
            deg, speech = mic.read_doa()
            print(f"deg={deg:3d}  speech={int(speech)}", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


def cmd_clap(mic):
    mic.start_poller(hz=20.0)
    readings = {}
    for pos in ("FRONT", "LEFT", "BACK", "RIGHT"):
        input(f"clap at {pos} of Spot (press Enter, then clap repeatedly for {CLAP_SECONDS:.0f} s) ")
        t0 = time.monotonic()
        time.sleep(CLAP_SECONDS)
        samples = mic.samples_between(t0, time.monotonic())
        degs = [d for _, d, speech in samples if speech]
        med = circ_median_deg(degs)
        readings[pos] = med
        shown = "none (no speech-flagged samples)" if med is None else f"{med:.1f} deg"
        print(f"  {pos}: {shown} from {len(degs)}/{len(samples)} samples")
    if mic.dead:
        print("device died during capture; results unreliable")
    front, left = readings["FRONT"], readings["LEFT"]
    if front is None or left is None:
        print("need FRONT and LEFT readings; rerun")
        return 1
    sign = 1 if circ_dist(left, (front + 90) % 360) <= SIGN_TOLERANCE_DEG else -1
    print(f"\nmic_yaw_offset_deg = {front:.1f}")
    print(f"mic_yaw_sign = {sign}")
    return 0


def cmd_leds(mic):
    for effect in (1, 3, 4, 5):
        print(f"LED effect {effect}")
        mic.set_led_effect(effect)
        time.sleep(2.0)
    mic.set_led_effect(5)
    colors = [0xFF0000, 0x00FF00, 0x0000FF]
    t0 = time.monotonic()
    for i in range(50):
        c = colors[i % len(colors)]
        mic.set_ring([c] * 12)
    dt = time.monotonic() - t0
    if mic.dead:
        print("device died during set_ring timing")
        return 1
    print(f"50 set_ring writes in {dt:.3f} s -> {50 / dt:.1f} writes/s")
    mic.set_led_effect(4)  # back to DoA mode
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("live", "clap", "leds"))
    args = ap.parse_args()
    mic = MicArray.open()
    if mic is None:
        print("XVF3800 (2886:001a) not found or pyusb missing")
        return 1
    try:
        return {"live": cmd_live, "clap": cmd_clap, "leds": cmd_leds}[args.mode](mic) or 0
    finally:
        mic.close()


if __name__ == "__main__":
    sys.exit(main())
