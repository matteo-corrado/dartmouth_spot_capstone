#!/usr/bin/env python3
"""XVF3800 probe: live DoA, clap calibration, LED write-rate limits.

Usage:
  spot-env/bin/python scripts/xvf_probe.py live    # deg/speech at 10 Hz (Ctrl-C to stop)
  spot-env/bin/python scripts/xvf_probe.py clap    # speak at FRONT/LEFT/BACK/RIGHT -> mic_yaw_offset_deg / mic_yaw_sign
  spot-env/bin/python scripts/xvf_probe.py leds    # index calibration, then cycle effects, measure set_ring writes/s
  spot-env/bin/python scripts/xvf_probe.py index   # LED index/orientation -> ring_offset_leds / ring_reverse

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

from src.voice_control.mic_array import MicArray, XvfError  # noqa: E402

CLAP_SECONDS = 6.0
CLAP_SETTLE_S = 2.0
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
    except XvfError as e:
        print(f"device error: {e}")
        return 1


def capture(mic, pos):
    """Median deg at one position: speech-flagged samples, else all samples (WARNING).

    The first CLAP_SETTLE_S are discarded: the XVF beam takes a moment to
    re-converge on a new talker (longer in echoey rooms)."""
    input(f"stand at Spot's {pos}, press Enter, then keep talking (\"hey spot, hey spot...\") "
          f"for {CLAP_SECONDS:.0f} s; claps optional ")
    t0 = time.monotonic()
    time.sleep(CLAP_SECONDS)
    samples = mic.samples_between(t0 + CLAP_SETTLE_S, time.monotonic())
    flagged = [d for _, d, speech in samples if speech]
    print(f"  {pos}: {len(flagged)}/{len(samples)} samples speech-flagged "
          f"(first {CLAP_SETTLE_S:.0f} s ignored)")
    if flagged:
        m = circ_median_deg(flagged)
        within = sum(circ_dist(d, m) <= 30 for d in flagged) / len(flagged)
        print(f"  {pos}: {within:.0%} of speech samples within 30 deg of the median"
              + ("  <-- unstable, consider rerun" if within < 0.6 else ""))
    if flagged:
        med = circ_median_deg(flagged)
    else:
        med = circ_median_deg([d for _, d, _ in samples])
        if med is not None:
            print(f"  WARNING: no speech-flagged samples at {pos}; using median of ALL samples")
    print(f"  {pos}: " + ("no samples" if med is None else f"{med:.1f} deg"))
    return med


def near(a, b):
    return a is not None and circ_dist(a, b) <= SIGN_TOLERANCE_DEG


def cmd_clap(mic):
    mic.start_poller(hz=20.0)
    r = {pos: capture(mic, pos) for pos in ("FRONT", "LEFT", "BACK", "RIGHT")}
    if mic.dead:
        print("device died during capture; results unreliable")
        return 1
    front, left, back, right = r["FRONT"], r["LEFT"], r["BACK"], r["RIGHT"]
    if front is None or left is None:
        print("need FRONT and LEFT readings; rerun")
        return 1
    if back is not None:
        print(f"sanity: BACK is {circ_dist(back, (front + 180) % 360):.1f} deg from front+180")
    plus = near(left, (front + 90) % 360) and (right is None or near(right, (front - 90) % 360))
    minus = near(left, (front - 90) % 360) and (right is None or near(right, (front + 90) % 360))
    print(f"\nmic_yaw_offset_deg = {front:.1f}")
    if plus == minus:
        print("mic_yaw_sign: ambiguous — rerun")
        return 1
    sign = 1 if plus else -1
    print("\npaste into config/tuning.yaml:")
    print(f"mic_yaw_offset_deg: {front:.1f}")
    print(f"mic_yaw_sign: {sign}")
    return 0


def clock_to_deg(hours):
    """Clock position (12 = Spot's front, 3 = right) -> degrees clockwise from front."""
    return (float(hours) * 30.0) % 360


def derive_ring_mapping(answers, front_mic_deg, doa_cw):
    """answers: {led_index: clock_hours} for LEDs 0/3/6/9. doa_cw: True if mic DoA angle
    increases clockwise seen from above. Returns (front_led, led_cw, offset, reverse)."""
    pos = {k: clock_to_deg(h) for k, h in answers.items()}
    steps = []
    for k in pos:
        nxt = (k + 3) % 12
        if nxt in pos:
            d = ((pos[nxt] - pos[k] + 180) % 360) - 180  # +90 = clockwise
            steps.append(d)
    led_cw = sum(steps) > 0
    direction = 1 if led_cw else -1
    p0 = pos[0]
    front_led = round(((-p0) * direction) / 30.0) % 12
    reverse = led_cw != doa_cw
    sign = -1 if reverse else 1
    offset = (front_led - round(front_mic_deg / 30.0) * sign) % 12
    return front_led, led_cw, offset, reverse


def ask_clock(k):  # noqa: ARG001
    while True:
        raw = input("Which clock position relative to Spot's FRONT "
                    "(12 = front, 3 = Spot's right side, 9 = Spot's left)? ").strip()
        try:
            v = float(raw)
        except ValueError:
            print("  enter a number 1-12 (halves ok, e.g. 12.5)")
            continue
        if 0 <= v < 13:
            return v or 12.0
        print("  enter a number 1-12 (halves ok, e.g. 12.5)")


def cmd_index(mic):
    try:
        mic.set_led_effect(5)
        answers = {}
        for k in (0, 3, 6, 9):
            mic.set_ring([0x00FF00 if i == k else 0 for i in range(12)])
            print(f"LED {k} is lit (bright green, all others off)")
            answers[k] = ask_clock(k)
        mic.set_ring([0] * 12)
        mic.start_poller(hz=20.0)
        front = capture(mic, "FRONT (talk from Spot's front)")
        right = capture(mic, "RIGHT (Spot's 3 o'clock; sets DoA direction)")
        if mic.dead or front is None:
            print("no usable DoA capture for the front; rerun")
            return 1
        if right is None:
            print("WARNING: no RIGHT capture; assuming DoA increases counter-clockwise")
            doa_cw = False
        else:
            cw_err = circ_dist(right, (front + 90) % 360)
            ccw_err = circ_dist(right, (front - 90) % 360)
            doa_cw = cw_err < ccw_err
            if min(cw_err, ccw_err) > SIGN_TOLERANCE_DEG:
                print("WARNING: RIGHT reading is not ~90 deg from FRONT; direction guess is weak")
        front_led, led_cw, offset, reverse = derive_ring_mapping(answers, front, doa_cw)
        print(f"\nLED nearest Spot's front: {front_led}")
        print(f"LED index increases {'clockwise' if led_cw else 'counter-clockwise'} seen from above")
        print(f"mic-frame front angle: {front:.1f} deg; DoA increases "
              f"{'clockwise' if doa_cw else 'counter-clockwise'}")
        print("\npaste into config/tuning.yaml:")
        print(f"ring_offset_leds: {offset}")
        print(f"ring_reverse: {'true' if reverse else 'false'}")
        return 0
    finally:
        mic.set_led_effect(4)


def cmd_leds(mic):
    try:
        rc = cmd_index(mic)
        if rc:
            print("index calibration incomplete; continuing with effects test")
        for effect in (1, 3, 4, 5):
            print(f"LED effect {effect}")
            mic.set_led_effect(effect)
            time.sleep(2.0)
        mic.set_led_effect(5)
        colors = [0xFF0000, 0x00FF00, 0x0000FF]
        t0 = time.monotonic()
        for i in range(50):
            mic.set_ring([colors[i % len(colors)]] * 12)
        dt = time.monotonic() - t0
        if mic.dead:
            print("device died during set_ring timing")
            return 1
        print(f"50 set_ring writes in {dt:.3f} s -> {50 / dt:.1f} writes/s")
        return 0
    finally:
        mic.set_led_effect(4)  # back to DoA mode on every exit path, incl. Ctrl-C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("live", "clap", "leds", "index"))
    args = ap.parse_args()
    mic = MicArray.open()
    if mic is None:
        print("XVF3800 (2886:001a) not found or pyusb missing")
        return 1
    try:
        return {"live": cmd_live, "clap": cmd_clap, "leds": cmd_leds, "index": cmd_index}[args.mode](mic) or 0
    finally:
        mic.close()


if __name__ == "__main__":
    sys.exit(main())
