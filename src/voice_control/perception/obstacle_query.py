"""Body-frame obstacle queries on top of decoded local grids.

This module answers the question "is there something near me?" by
consulting the grids exposed by :mod:`local_grid_helpers` and
translating their cells into body-frame ``(distance, bearing)`` pairs.

Two grid types are currently understood:

* ``obstacle_distance`` — Spot's built-in local obstacle-distance field.
  The semantics documented by Boston Dynamics say this is a signed
  distance-to-nearest-obstacle field: a cell value of ``d`` means "the
  nearest obstacle is ``d`` meters away from that cell", and a value at
  or below zero means the cell itself is inside an obstacle.  On base
  (non-EAP-2) hardware we have not empirically verified this — see the
  uncertainty notes in the module docstring at the bottom.

* ``no_step`` — a binary/uint8 "unsafe to step here" map.  Cells above a
  threshold are flagged as unsafe.

If none of the preferred grids are available the query returns ``None``;
callers should treat that as "no obstacle information" rather than "no
obstacle".

Frame convention: all returned bearings are in the robot's ``body``
frame with ``0`` pointing forward (+X), ``+pi/2`` pointing to the
robot's left (+Y), and ``-pi/2`` pointing right.  We use
``bosdyn.client.frame_helpers.get_a_tform_b`` to map the grid's native
frame (typically ``gpe``) into body coordinates for the sweep.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b

from .local_grid_helpers import DecodedGrid, fetch_grids

logger = logging.getLogger(__name__)


# Default order of grid preference.  ``obstacle_distance`` is the richest
# signal, ``no_step`` is the common fallback, ``terrain_valid`` is listed
# for compatibility but currently treated as an unknown-format fall-
# through (returning ``None`` from that branch lets the scan continue to
# the next grid name in the preference list).
_DEFAULT_GRID_PREFERENCE: list[str] = [
    "obstacle_distance",
    "no_step",
    "terrain_valid",
]

# Threshold above which an ``unsigned 8-bit`` no_step cell is considered
# unsafe.  BD documents no_step as 0 = safe, 255 = unsafe but leaves the
# in-between behavior unspecified; 128 is the midpoint and matches the
# behavior of the existing graph_nav autowalk code.
_NO_STEP_UNSAFE_THRESHOLD = 128

# Threshold below which an ``obstacle_distance`` cell is considered part
# of an obstacle.  BD documents the grid as a signed distance field, so
# the obstacle surface is at ``value == 0``.  We return the nearest cell
# whose value falls below this threshold (``<= 0``) as the "nearest
# obstacle".  See the module docstring for the uncertainty caveat.
_OBSTACLE_DISTANCE_THRESHOLD = 0.0


@dataclass
class ObstacleHit:
    """A single nearest-obstacle result in the robot's body frame.

    Attributes:
        distance_m: Euclidean distance from the body origin to the hit
            cell, in meters.
        bearing_rad: Bearing in the body frame; ``0`` is forward,
            ``+pi/2`` is to the robot's left, ``-pi/2`` to the right.
        grid_name: The decoded grid name that produced the hit, for
            debugging and logging.
        cell_xy_in_grid_frame: ``(x, y)`` of the hit cell center in the
            grid's own frame (typically ``gpe``).  Useful when debugging
            frame-transform bugs.
    """

    distance_m: float
    bearing_rad: float
    grid_name: str
    cell_xy_in_grid_frame: tuple[float, float]


def _robot_origin_in_grid_frame(grid: DecodedGrid) -> tuple[float, float, float] | None:
    """Compute the body-origin (``(0,0,0)_body``) expressed in grid coords.

    ``get_a_tform_b(snap, BODY, grid.frame)`` returns ``body_tform_grid``
    — a pose whose ``transform_point`` call maps a grid-frame point into
    the body frame.  We need the opposite direction (grid coords of the
    body origin), so we invert and transform ``(0, 0, 0)``.
    """

    snap = grid.transforms_snapshot
    if snap is None:
        return None
    try:
        body_tform_grid = get_a_tform_b(snap, BODY_FRAME_NAME, grid.frame)
    except Exception as exc:
        logger.warning(
            "get_a_tform_b(BODY, %s) failed for grid %r: %s",
            grid.frame,
            grid.name,
            exc,
        )
        return None
    if body_tform_grid is None:
        logger.warning(
            "No transform between %s and grid frame %s for grid %r",
            BODY_FRAME_NAME,
            grid.frame,
            grid.name,
        )
        return None
    grid_tform_body = body_tform_grid.inverse()
    return grid_tform_body.transform_point(0.0, 0.0, 0.0)


def _cell_grid_positions(grid: DecodedGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xs, ys)`` cell-center coordinates in the grid's frame.

    The LocalGrid buffer is row-major with rows varying over Y and
    columns over X, so cell ``array[iy, ix]`` has center
    ``(ix + 0.5, iy + 0.5) * cell_size`` in the grid's frame (measured
    from the grid's origin corner).

    Shapes: both returned arrays are ``(num_cells_y, num_cells_x)``.
    """

    n_y, n_x = grid.array.shape
    cs = grid.cell_size_m
    xs = (np.arange(n_x, dtype=np.float64) + 0.5) * cs
    ys = (np.arange(n_y, dtype=np.float64) + 0.5) * cs
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return xx, yy


def _scan_obstacle_distance(
    grid: DecodedGrid,
    max_radius_m: float,
) -> ObstacleHit | None:
    """Find the nearest ``obstacle_distance`` cell flagged as obstacle.

    The implemented semantics (see module docstring for uncertainty):
    scan every cell within ``max_radius_m`` of the robot body origin
    (measured in the grid frame using the ``gpe``→``body`` transform)
    whose *grid value* is ``<= _OBSTACLE_DISTANCE_THRESHOLD`` and return
    the one closest to the body origin.  This treats cells with
    non-positive distance-to-obstacle as "inside an obstacle" and finds
    the closest such cell, which matches the BD documented signed-
    distance-field interpretation.
    """

    origin_grid = _robot_origin_in_grid_frame(grid)
    if origin_grid is None:
        return None
    gx, gy, _gz = origin_grid

    xx, yy = _cell_grid_positions(grid)
    dx = xx - gx
    dy = yy - gy
    # ``sqrt(dx**2 + dy**2)`` is cell-center distance from the robot in
    # the grid's horizontal plane; altitude differences are ignored
    # because the grid is a 2.5D slice.
    dist_to_cell = np.sqrt(dx * dx + dy * dy)

    values = grid.array  # NaN for unknown cells already
    # An "obstacle" cell is one whose stored distance-to-obstacle is at
    # or below zero.  ``np.nan`` comparisons yield ``False``, so unknown
    # cells are automatically excluded from the mask.
    obstacle_mask = values <= _OBSTACLE_DISTANCE_THRESHOLD
    in_range = dist_to_cell <= max_radius_m
    candidate = obstacle_mask & in_range

    if not np.any(candidate):
        return None

    # Replace non-candidate distances with +inf so ``argmin`` picks the
    # nearest candidate.
    masked = np.where(candidate, dist_to_cell, np.inf)
    flat_idx = int(np.argmin(masked))
    iy, ix = divmod(flat_idx, values.shape[1])
    hit_distance = float(dist_to_cell[iy, ix])
    # Bearing in body frame: cell position relative to robot origin is
    # ``(dx, dy)`` in the grid frame.  But we want bearing in BODY
    # coordinates, so transform the relative vector via
    # ``body_tform_grid`` applied to the cell position and then subtract
    # the body origin (which is (0,0,0) in body).
    hit_x_grid = float(xx[iy, ix])
    hit_y_grid = float(yy[iy, ix])
    bearing = _bearing_to_grid_point(grid, hit_x_grid, hit_y_grid)
    if bearing is None:
        return None
    return ObstacleHit(
        distance_m=hit_distance,
        bearing_rad=bearing,
        grid_name=grid.name,
        cell_xy_in_grid_frame=(hit_x_grid, hit_y_grid),
    )


def _scan_no_step(
    grid: DecodedGrid,
    max_radius_m: float,
) -> ObstacleHit | None:
    """Find the nearest ``no_step`` cell flagged as unsafe within range."""

    origin_grid = _robot_origin_in_grid_frame(grid)
    if origin_grid is None:
        return None
    gx, gy, _gz = origin_grid

    xx, yy = _cell_grid_positions(grid)
    dx = xx - gx
    dy = yy - gy
    dist_to_cell = np.sqrt(dx * dx + dy * dy)

    values = grid.array  # already float64 with NaN for unknowns
    unsafe_mask = values >= _NO_STEP_UNSAFE_THRESHOLD
    in_range = dist_to_cell <= max_radius_m
    candidate = unsafe_mask & in_range
    if not np.any(candidate):
        return None

    masked = np.where(candidate, dist_to_cell, np.inf)
    flat_idx = int(np.argmin(masked))
    iy, ix = divmod(flat_idx, values.shape[1])
    hit_distance = float(dist_to_cell[iy, ix])
    hit_x_grid = float(xx[iy, ix])
    hit_y_grid = float(yy[iy, ix])
    bearing = _bearing_to_grid_point(grid, hit_x_grid, hit_y_grid)
    if bearing is None:
        return None
    return ObstacleHit(
        distance_m=hit_distance,
        bearing_rad=bearing,
        grid_name=grid.name,
        cell_xy_in_grid_frame=(hit_x_grid, hit_y_grid),
    )


def _bearing_to_grid_point(
    grid: DecodedGrid,
    x_grid: float,
    y_grid: float,
) -> float | None:
    """Return ``atan2(y_body, x_body)`` for a grid-frame point.

    The point is transformed into body frame via ``body_tform_grid``
    (which is what ``get_a_tform_b`` returns when you ask for ``A=BODY``,
    ``B=grid.frame``).  Altitude is discarded — we only care about the
    horizontal bearing.
    """

    snap = grid.transforms_snapshot
    if snap is None:
        return None
    try:
        body_tform_grid = get_a_tform_b(snap, BODY_FRAME_NAME, grid.frame)
    except Exception as exc:
        logger.warning("bearing transform failed for grid %r: %s", grid.name, exc)
        return None
    if body_tform_grid is None:
        return None
    bx, by, _bz = body_tform_grid.transform_point(x_grid, y_grid, 0.0)
    return math.atan2(by, bx)


def nearest_obstacle_in_body_frame(
    robot,
    max_radius_m: float = 3.0,
    grid_preference: list[str] | None = None,
) -> ObstacleHit | None:
    """Find the nearest obstacle within ``max_radius_m`` of the robot.

    Args:
        robot: A connected ``bosdyn.client.robot.Robot`` instance.
        max_radius_m: Only cells within this Euclidean distance (in the
            grid's horizontal plane, relative to the body origin) are
            considered.  Defaults to 3 m.
        grid_preference: Ordered list of grid type names to try.  The
            first grid that (a) is exposed by this Spot AND (b) has a
            handler in this module AND (c) contains an obstacle inside
            ``max_radius_m`` wins.  Defaults to ``["obstacle_distance",
            "no_step", "terrain_valid"]``.

    Behavior per grid:

    * ``obstacle_distance``: BD documents this as a signed distance-to-
      nearest-obstacle field in meters.  We scan all cells within
      ``max_radius_m`` of the body origin whose stored value is at or
      below ``0.0`` and return the one closest to the robot — that is,
      the nearest cell that lies *inside* an obstacle.  Unknown cells
      (NaN) are automatically excluded.  If no such cell exists, the
      grid is considered "all clear" and we fall through to the next
      grid in ``grid_preference``.
    * ``no_step``: a uint8 "unsafe to step" map.  We return the nearest
      cell whose value is ``>= 128`` (midpoint between the documented
      ``0 = safe`` and ``255 = unsafe`` endpoints).
    * any other grid name: currently unhandled, logged at ``debug`` and
      skipped (the scan falls through to the next preferred grid).

    Returns:
        :class:`ObstacleHit` if an obstacle is found, or ``None`` if no
        usable grid is available or no obstacle falls within the radius.

    Note:
        There is genuine uncertainty about whether ``obstacle_distance``
        on base Spot is a signed distance field, a raw positive distance,
        or something else entirely.  The implementation above assumes
        the signed-distance-field interpretation from the BD docs; if
        this Spot's grid turns out to contain only positive values, the
        scan will never find a candidate and we'll always fall through
        to ``no_step``.  That's a safe failure mode — just less
        informative.
    """

    if grid_preference is None:
        grid_preference = _DEFAULT_GRID_PREFERENCE

    grids = fetch_grids(robot, names=list(grid_preference))
    if not grids:
        return None

    for name in grid_preference:
        grid = grids.get(name)
        if grid is None:
            continue
        if name == "obstacle_distance":
            hit = _scan_obstacle_distance(grid, max_radius_m)
        elif name == "no_step":
            hit = _scan_no_step(grid, max_radius_m)
        else:
            logger.debug("No obstacle-query handler for grid %r, skipping", name)
            continue
        if hit is not None:
            return hit
    return None
