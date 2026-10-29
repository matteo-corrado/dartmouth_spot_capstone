"""Helpers for Spot's built-in ``WorldObjectClient``.

Used by the ``go_to_object`` dispatch path to resolve "fiducial N" / "the
dock" descriptions into odom-frame approach poses WITHOUT going through
YOLO. The LLM only ever generates ``go_to_object(description=...)``; the
smart routing happens here.

Why this is separate from ``visual_nav.py``: the world object service is
Spot's native detector (sub-centimeter accurate for AprilTags) and should
be preferred over YOLO whenever the description maps to one. Keeping it
in its own module avoids tangling it with the YOLO-World fallback.

All functions are read-only: they query state via
``list_world_objects`` and return a :class:`NavigationTarget` (or None),
leaving motion commands to the caller.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass

from bosdyn.api import world_object_pb2
from bosdyn.client.exceptions import Error as BosdynError
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    get_a_tform_b,
    get_odom_tform_body,
)
from bosdyn.client.world_object import WorldObjectClient

logger = logging.getLogger(__name__)


# Regex patterns that mean "the user is asking for fiducial N". Anchored
# on word boundaries so "fiducialize" or "tag-along" don't match. Digits
# are captured as group 1.
_FIDUCIAL_PATTERNS = [
    re.compile(r"\bfiducial[\s_-]*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bapril[\s_-]?tag[\s_-]*(\d+)\b", re.IGNORECASE),
    re.compile(r"\btag[\s_-]*(\d+)\b", re.IGNORECASE),
    re.compile(r"\bmarker[\s_-]*(\d+)\b", re.IGNORECASE),
]

# Substrings (lowercased, .strip()ed) that mean "the dock". These are
# matched as plain substrings rather than word-boundary regex because
# users say things like "the charging dock" or "my charger".
_DOCK_KEYWORDS = ("dock", "charger", "charging station", "charging dock")


@dataclass
class NavigationTarget:
    """An SE2 approach pose to hand off to the existing SE2 trajectory path.

    Attributes:
        x, y: target position in ``frame_name`` coordinates (meters).
        yaw: target yaw in ``frame_name`` coordinates (radians).
        frame_name: The frame the coords live in. Today we always use
            ``ODOM_FRAME_NAME`` so the same ``synchro_se2_trajectory_command``
            path that ``walk_to`` uses can consume this directly.
        source: Human-readable origin string for logging, e.g.
            ``"fiducial_5"`` or ``"dock_42"``.
    """

    x: float
    y: float
    yaw: float
    frame_name: str
    source: str


# ---------------------------------------------------------------------------
# Description parsers — cheap, pure-python, no SDK calls
# ---------------------------------------------------------------------------


def parse_fiducial_tag_id(description: str) -> int | None:
    """Return the fiducial tag id mentioned in ``description``, else ``None``.

    Matches "fiducial 5", "apriltag 5", "april tag 5", "tag 5", "marker 5"
    (case-insensitive, optional separator). Returns ``None`` if no
    pattern matches.
    """
    for pattern in _FIDUCIAL_PATTERNS:
        match = pattern.search(description)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                continue
    return None


def is_dock_reference(description: str) -> bool:
    """Return True if ``description`` asks for a dock / charging station."""
    lower = description.strip().lower()
    return any(kw in lower for kw in _DOCK_KEYWORDS)


# ---------------------------------------------------------------------------
# World-object lookups
# ---------------------------------------------------------------------------


def _approach_pose_from_spot_toward_target(
    robot,
    target_x_odom: float,
    target_y_odom: float,
    standoff_m: float,
) -> tuple[float, float, float] | None:
    """Compute an SE2 approach pose ``standoff_m`` from the target, facing it.

    The approach point sits on the line from Spot's current body origin
    to ``(target_x_odom, target_y_odom)``, ``standoff_m`` back from the
    target. Yaw points toward the target. All coordinates are in the
    ``odom`` frame so the result can be sent straight to
    ``synchro_se2_trajectory_command``.

    Returns ``None`` if Spot's current pose cannot be read (robot state
    RPC failure, disconnected session, etc.).
    """
    try:
        state_client = robot.ensure_client("robot-state")
        robot_state = state_client.get_robot_state()
    except BosdynError as exc:
        logger.warning("get_robot_state failed while computing approach: %s", exc)
        return None
    except Exception as exc:  # defensive — catch ensure_client failures too
        logger.warning("robot-state client unavailable: %s", exc)
        return None

    try:
        odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )
    except Exception as exc:
        logger.warning("get_odom_tform_body failed: %s", exc)
        return None

    spot_x = odom_tform_body.x
    spot_y = odom_tform_body.y

    dx = target_x_odom - spot_x
    dy = target_y_odom - spot_y
    dist = math.hypot(dx, dy)
    if dist < 1e-3:
        # We are already on top of the target — no meaningful approach
        # direction. Stand in place, face forward.
        return (target_x_odom, target_y_odom, odom_tform_body.rot.to_yaw())

    if dist <= standoff_m:
        # Already inside the standoff distance — don't back up, just face
        # the target where we stand.
        return (spot_x, spot_y, math.atan2(dy, dx))

    # Approach point: standoff_m back from the target along the Spot→target
    # line. Equivalently: start at Spot, walk (dist - standoff_m) toward the
    # target.
    ratio = (dist - standoff_m) / dist
    approach_x = spot_x + ratio * dx
    approach_y = spot_y + ratio * dy
    yaw = math.atan2(dy, dx)  # face toward the target
    return (approach_x, approach_y, yaw)


def find_fiducial_target(
    robot,
    tag_id: int,
    standoff_m: float = 1.0,
) -> NavigationTarget | None:
    """Look up fiducial ``tag_id`` and return an odom-frame approach pose.

    Queries Spot's native AprilTag detector via
    ``WorldObjectClient.list_world_objects([WORLD_OBJECT_APRILTAG])``.
    Prefers the filtered (ICP-refined) pose over the raw detection;
    returns ``None`` if neither pose has ``STATUS_OK``, the fiducial
    isn't visible, or a frame-transform lookup fails.

    The returned pose sits ``standoff_m`` from the fiducial along the
    line from Spot's current position to the fiducial, facing the
    fiducial. We deliberately ignore the fiducial's own orientation —
    "walk to the tag" is the user's intent, not "line up perpendicular
    to the tag face".
    """
    try:
        client = robot.ensure_client(WorldObjectClient.default_service_name)
        resp = client.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG]
        )
    except BosdynError as exc:
        logger.warning("list_world_objects(APRILTAG) failed: %s", exc)
        return None
    except Exception as exc:
        logger.warning("world-object client unavailable: %s", exc)
        return None

    ok_status = world_object_pb2.AprilTagProperties.STATUS_OK

    for obj in resp.world_objects:
        props = obj.apriltag_properties
        if props.tag_id != tag_id:
            continue

        # Prefer the filtered (ICP-stabilized) pose; fall back to raw.
        if props.fiducial_filtered_pose_status == ok_status:
            frame_name = props.frame_name_fiducial_filtered
        elif props.fiducial_pose_status == ok_status:
            frame_name = props.frame_name_fiducial
        else:
            logger.info(
                "Fiducial %d visible but pose status not OK "
                "(raw=%s filtered=%s) — skipping",
                tag_id,
                props.fiducial_pose_status,
                props.fiducial_filtered_pose_status,
            )
            return None

        try:
            odom_tform_fid = get_a_tform_b(
                obj.transforms_snapshot, ODOM_FRAME_NAME, frame_name
            )
        except Exception as exc:
            logger.warning(
                "get_a_tform_b(odom, %s) failed: %s", frame_name, exc
            )
            return None
        if odom_tform_fid is None:
            logger.warning(
                "No transform chain from %s to %s in fiducial %d snapshot",
                ODOM_FRAME_NAME,
                frame_name,
                tag_id,
            )
            return None

        approach = _approach_pose_from_spot_toward_target(
            robot,
            target_x_odom=odom_tform_fid.x,
            target_y_odom=odom_tform_fid.y,
            standoff_m=standoff_m,
        )
        if approach is None:
            return None
        ax, ay, ayaw = approach
        return NavigationTarget(
            x=ax,
            y=ay,
            yaw=ayaw,
            frame_name=ODOM_FRAME_NAME,
            source=f"fiducial_{tag_id}",
        )

    logger.info("Fiducial %d is not in the current world-object list", tag_id)
    return None


def find_dock_target(
    robot,
    standoff_m: float = 1.0,
) -> NavigationTarget | None:
    """Find a visible dock and return an odom-frame approach pose.

    On base Spot there is typically at most one visible dock at a time.
    If multiple are reported we pick the closest one by Euclidean
    distance in the odom XY plane. Docks flagged ``unavailable`` are
    skipped.
    """
    try:
        client = robot.ensure_client(WorldObjectClient.default_service_name)
        resp = client.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_DOCK]
        )
    except BosdynError as exc:
        logger.warning("list_world_objects(DOCK) failed: %s", exc)
        return None
    except Exception as exc:
        logger.warning("world-object client unavailable: %s", exc)
        return None

    if not resp.world_objects:
        logger.info("No dock in current world-object list")
        return None

    # Resolve each dock's position in odom frame, skip unavailable ones.
    candidates: list[tuple[float, float, int]] = []
    for obj in resp.world_objects:
        props = obj.dock_properties
        if props.unavailable:
            logger.info("Dock id=%s flagged unavailable, skipping", props.dock_id)
            continue
        try:
            odom_tform_dock = get_a_tform_b(
                obj.transforms_snapshot,
                ODOM_FRAME_NAME,
                props.frame_name_dock,
            )
        except Exception as exc:
            logger.warning(
                "get_a_tform_b(odom, %s) failed: %s",
                props.frame_name_dock,
                exc,
            )
            continue
        if odom_tform_dock is None:
            continue
        candidates.append((odom_tform_dock.x, odom_tform_dock.y, props.dock_id))

    if not candidates:
        return None

    # Pick the closest dock to Spot (or the only one, commonly).
    try:
        state_client = robot.ensure_client("robot-state")
        robot_state = state_client.get_robot_state()
        odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot
        )
        spot_xy = (odom_tform_body.x, odom_tform_body.y)
    except Exception as exc:
        logger.warning("Could not read spot pose, picking first dock: %s", exc)
        spot_xy = None

    if spot_xy is not None:
        candidates.sort(
            key=lambda c: math.hypot(c[0] - spot_xy[0], c[1] - spot_xy[1])
        )
    target_x, target_y, dock_id = candidates[0]

    approach = _approach_pose_from_spot_toward_target(
        robot,
        target_x_odom=target_x,
        target_y_odom=target_y,
        standoff_m=standoff_m,
    )
    if approach is None:
        return None
    ax, ay, ayaw = approach
    return NavigationTarget(
        x=ax,
        y=ay,
        yaw=ayaw,
        frame_name=ODOM_FRAME_NAME,
        source=f"dock_{dock_id}",
    )


def find_navigation_target(
    robot,
    description: str,
    standoff_m: float = 1.0,
) -> NavigationTarget | None:
    """Resolve a user description to a WorldObject-backed approach pose.

    This is the entry point the ``go_to_object`` dispatch handler calls
    BEFORE falling through to YOLO. Returns a :class:`NavigationTarget`
    if ``description`` matches a known fiducial or dock on this robot,
    else ``None`` (meaning: not a world-object reference, try YOLO).

    The order of checks is fiducial-first (since "fiducial 5" is more
    specific and unambiguous than "the dock"), dock-second. Anything
    else returns None.
    """
    tag_id = parse_fiducial_tag_id(description)
    if tag_id is not None:
        return find_fiducial_target(robot, tag_id, standoff_m=standoff_m)
    if is_dock_reference(description):
        return find_dock_target(robot, standoff_m=standoff_m)
    return None
