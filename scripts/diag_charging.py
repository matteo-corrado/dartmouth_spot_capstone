#!/usr/bin/env python3
"""Charging diagnostics — stream Spot battery + shore-power state over time.

Polls RobotState at a fixed interval and prints one line per sample with SOC,
charge current, voltage, temperature, shore-power state, and any battery /
charger / power faults. Tracks SOC delta and extrapolates a fill-time estimate.

Usage:
    ./spot-env/bin/python scripts/diag_charging.py
    ./spot-env/bin/python scripts/diag_charging.py --interval 30 --duration 1800
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.session import quick_robot
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_id import RobotIdClient
from bosdyn.api.robot_state_pb2 import BatteryState, PowerState


def _bs_name(s: int) -> str:
    return BatteryState.Status.Name(s)


def _shore_name(s: int) -> str:
    try:
        return PowerState.ShorePowerState.Name(s)
    except Exception:
        return f"<{s}>"


def _motor_name(s: int) -> str:
    return PowerState.MotorPowerState.Name(s)


def _bat_faults(rs) -> list[str]:
    out = []
    for f in rs.system_fault_state.faults:
        attrs = list(f.attributes)
        if "battery" in attrs or "charger" in attrs or "power" in attrs:
            out.append(f"{f.name}({f.severity}):{f.error_message!r}")
    return out


class _Tee:
    """Write to stdout AND a logfile. Line-buffered."""
    def __init__(self, path: Path):
        self.path = path
        self.f = open(path, "w", buffering=1)

    def write(self, s: str) -> int:
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.f.write(s)
        return len(s)

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.f.flush()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=float, default=10.0, help="seconds between samples")
    ap.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = forever)")
    ap.add_argument("--no-log", action="store_true", help="disable auto-log to logs/charging/")
    ap.add_argument("--label", type=str, default="", help="optional label appended to log filename")
    args = ap.parse_args()

    if not args.no_log:
        log_dir = PROJECT_ROOT / "logs" / "charging"
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        suffix = f"_{args.label}" if args.label else ""
        log_path = log_dir / f"diag_{ts}{suffix}.log"
        sys.stdout = _Tee(log_path)
        print(f"# logging to: {log_path}")

    robot = quick_robot("charge-diag")
    state_client = robot.ensure_client(RobotStateClient.default_service_name)

    try:
        rid = robot.ensure_client(RobotIdClient.default_service_name).get_id()
        sv = rid.software_release.version
        print(f"# robot SN: {rid.serial_number!r}  nickname: {rid.nickname!r}  SW: {sv.major_version}.{sv.minor_version}.{sv.patch_level}")
    except Exception as exc:
        print(f"# robot id: unavailable ({exc})")

    try:
        rs0 = state_client.get_robot_state()
        if rs0.battery_states:
            print(f"# battery SN: {rs0.battery_states[0].identifier!r}")
    except Exception:
        pass

    start = time.time()
    first_soc: float | None = None
    first_t: float | None = None
    prev_soc: float | None = None

    hdr = "elapsed  SOC%   d%  A      V      T°C   comm%  bat_status         shore                  motor       faults"
    print(hdr)
    print("-" * len(hdr))

    while True:
        rs = state_client.get_robot_state()
        ps = rs.power_state
        if not rs.battery_states:
            print("(no battery_states — battery missing?)")
            time.sleep(args.interval)
            continue
        b = rs.battery_states[0]
        soc = b.charge_percentage.value
        amps = b.current.value
        volts = b.voltage.value
        temps = list(b.temperatures)
        tavg = sum(temps) / len(temps) if temps else 0.0
        comm_loss = b.communications_loss_percent.value if b.HasField("communications_loss_percent") else float("nan")
        shore = _shore_name(ps.shore_power_state)
        motor = _motor_name(ps.motor_power_state)
        faults = _bat_faults(rs)

        now = time.time()
        if first_soc is None:
            first_soc, first_t = soc, now
        dsoc = soc - (prev_soc if prev_soc is not None else soc)
        prev_soc = soc

        elapsed = now - start
        comm_str = f"{comm_loss:5.1f}" if comm_loss == comm_loss else "  nan"
        print(
            f"{elapsed:6.0f}s  {soc:4.1f}  {dsoc:+4.1f}  {amps:+5.2f}  {volts:5.2f}  "
            f"{tavg:4.1f}  {comm_str}  {_bs_name(b.status):17s}  {shore:22s} {motor:11s} {';'.join(faults) or '-'}"
        )

        # ETA to 100% based on total session rate (more stable than per-sample)
        total_dsoc = soc - first_soc
        total_dt = now - first_t
        if total_dt > 60 and total_dsoc > 0.5:
            rate_pct_per_min = total_dsoc / (total_dt / 60.0)
            remain_pct = 100.0 - soc
            eta_min = remain_pct / rate_pct_per_min if rate_pct_per_min > 0 else 0
            print(f"        rate={rate_pct_per_min:.2f}%/min  ETA-to-100%={eta_min:.0f} min")

        if args.duration and (now - start) >= args.duration:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
