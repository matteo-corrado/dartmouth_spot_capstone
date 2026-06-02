"""Stage 2E.2 feasibility spike — measure max gripper command rate on Spot.

Deploys the arm to the ready pose, then modulates gripper_open_fraction by a
1 Hz sine wave for a few seconds at each target command rate. Prints the
ACHIEVED rate and jitter so we know the practical ceiling before building the
mouth animator (see docs/superpowers/plans/2026-06-02-stage2e2-gripper-mouth.md
Task 0).

Connects via the repo's non-interactive ``spot_session`` helper (auth + lease +
power-on + stand from env creds). Sits + powers off on exit.

SAFETY: this DEPLOYS THE ARM and moves the gripper. Clear the area around the
arm and keep the e-stop in hand before running.

    python scripts/spike_gripper_rate.py
    python scripts/spike_gripper_rate.py --rates 5 10 20 --secs 5
"""
import argparse
import math
import time

from bosdyn.client.robot_command import RobotCommandBuilder

from src.session import spot_session, deploy_arm_safe, stow_arm


def run_rate(cmd_client, hz: float, secs: float) -> dict:
    """Send sine-modulated gripper_open_fraction at target ``hz`` for ``secs``.

    Returns achieved rate + inter-command gap stats (ms)."""
    period = 1.0 / hz
    n = int(secs * hz)
    start = time.monotonic()
    sent = 0
    gaps = []
    last = start
    for i in range(n):
        target = start + i * period
        now = time.monotonic()
        if target > now:
            time.sleep(target - now)
        frac = 0.5 * (1.0 + math.sin(2 * math.pi * 1.0 * (time.monotonic() - start)))
        cmd_client.robot_command(
            RobotCommandBuilder.claw_gripper_open_fraction_command(frac))
        sent += 1
        t = time.monotonic()
        gaps.append(t - last)
        last = t
    elapsed = time.monotonic() - start
    gaps = gaps[1:] if len(gaps) > 1 else gaps  # drop first (includes setup)
    mean_gap = sum(gaps) / len(gaps) if gaps else 0.0
    jitter = (max(gaps) - min(gaps)) if gaps else 0.0
    return {
        "target_hz": hz,
        "achieved_hz": round(sent / elapsed, 2),
        "mean_gap_ms": round(mean_gap * 1000, 2),
        "jitter_ms": round(jitter * 1000, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rates", type=float, nargs="+", default=[5, 10, 20, 30],
                    help="target command rates in Hz")
    ap.add_argument("--secs", type=float, default=8.0,
                    help="seconds to hold each rate")
    ap.add_argument("--hostname", default=None,
                    help="override Spot IP (default: BOSDYN_ROBOT_IP env)")
    args = ap.parse_args()

    session_kwargs = {"stand_on_enter": True, "sit_on_exit": True}
    if args.hostname:
        session_kwargs["hostname"] = args.hostname

    print("=" * 60)
    print("STAGE 2E.2 GRIPPER RATE SPIKE — arm WILL deploy. E-stop ready.")
    print("=" * 60)

    with spot_session(**session_kwargs) as session:
        robot = session["robot"]
        cmd_client = session["cmd"]
        if not robot.has_arm():
            print("ABORT: Spot has no arm installed — mouth animation is impossible.")
            return

        print("[spike] Deploying arm to ready pose...")
        deploy_arm_safe(cmd_client, timeout_sec=6.0)

        results = []
        for hz in args.rates:
            print(f"[spike] Running {hz} Hz for {args.secs}s...")
            r = run_rate(cmd_client, hz, args.secs)
            print(f"[spike] RESULT {r}")
            results.append(r)

        print("[spike] Closing gripper + stowing arm...")
        stow_arm(cmd_client, timeout_sec=6.0)

        print("\n" + "=" * 60)
        print("SPIKE SUMMARY")
        for r in results:
            verdict = "smooth-OK" if r["achieved_hz"] >= 10 else "TOO-SLOW"
            print(f"  target {r['target_hz']:>4} Hz -> achieved {r['achieved_hz']:>6} Hz "
                  f"| jitter {r['jitter_ms']:>6} ms | {verdict}")
        best = max(results, key=lambda x: x["achieved_hz"])
        print(f"\n  MOUTH_FPS candidate (cap 20): {min(20, int(best['achieved_hz']))}")
        print("  If best achieved < ~10 Hz: switch to syllable-trigger fork (design §6.3).")
        print("=" * 60)


if __name__ == "__main__":
    main()
