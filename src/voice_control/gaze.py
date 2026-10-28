"""Pure math for pointing the gripper camera at the locked speaker (Task 19).

Bearings are body-frame degrees: 0 = straight ahead, +90 = robot's left (+y).
The dispatcher's gaze_at_bearing() (wired in a later lane) feeds gaze_target()
into RobotCommandBuilder.arm_gaze_command(x, y, z, "body", ...).
"""
import math

MAX_GAZE_DEG = 120.0


def _wrap(deg: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return ((deg + 180.0) % 360.0) - 180.0


def gaze_target(body_deg: float, dist_m=2.0, height_m=0.9) -> tuple[float, float, float] | None:
    """Body-frame point dist_m away along body_deg at height_m; None if behind (>120 deg)."""
    d = _wrap(body_deg)
    if abs(d) > MAX_GAZE_DEG:
        return None
    r = math.radians(d)
    return (dist_m * math.cos(r), dist_m * math.sin(r), height_m)


def should_regaze(prev_deg, new_deg, min_change=15.0) -> bool:
    """True when the bearing moved by more than min_change degrees (wrap-aware)."""
    return abs(_wrap(new_deg - prev_deg)) > min_change
