"""Thin wrapper around ``RayCastClient`` for body-frame ray queries.

The BD ``RayCastClient`` takes tuples for origin/direction and constructs
the ``Vec3`` protos internally, so this module mostly exists to:

* cache the client on the module keyed by ``robot._name``,
* default the frame and intersection type list to something sensible,
* convert the returned ``RaycastResponse`` into a simple list of
  :class:`RayHit` dataclasses with human-readable type names,
* and swallow ``RpcError`` / service-unavailable failures into an empty
  list with a warning, so the caller can treat "service missing" and
  "nothing hit" identically if it wants to.

Note on ``TYPE_VOXEL_MAP``: on a base (non-EAP-2) Spot, voxel-map
raycasts may return no hits.  Callers should not assume voxel-map is
available; instead, they should request all types they might care about
and inspect which ones actually came back.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from bosdyn.api import ray_cast_pb2
from bosdyn.client.exceptions import Error as BosdynError
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from bosdyn.client.ray_cast import RayCastClient

logger = logging.getLogger(__name__)


# Default intersection types: ask the robot about all four documented
# surfaces.  TYPE_VOXEL_MAP may return nothing on base Spot; TYPE_HAND_
# DEPTH requires the gripper camera to be active.
_DEFAULT_TYPES: list[int] = [
    ray_cast_pb2.RayIntersection.TYPE_GROUND_PLANE,
    ray_cast_pb2.RayIntersection.TYPE_TERRAIN_MAP,
    ray_cast_pb2.RayIntersection.TYPE_VOXEL_MAP,
    ray_cast_pb2.RayIntersection.TYPE_HAND_DEPTH,
]


# Module-level client cache keyed by ``robot._name`` — same pattern as
# in :mod:`local_grid_helpers`.  Keeps us from paying the
# ``ensure_client`` lookup on every query inside a tight loop (e.g. the
# 5-10 Hz collision pre-check described in the plan).
_client_cache: dict[str, RayCastClient] = {}


@dataclass
class RayHit:
    """A single ray intersection returned by the raycast service.

    Attributes:
        type_name: Human-readable intersection type, e.g.
            ``"TYPE_TERRAIN_MAP"``.  Mirrors
            ``bosdyn.api.ray_cast_pb2.RayIntersection.Type.Name``.
        distance_m: Distance from the ray origin to the hit point, in
            meters.
        hit_position: ``(x, y, z)`` of the hit in the same frame as the
            ray request.  Note: the BD proto calls this
            ``hit_position_in_hit_frame``, and the "hit frame" is
            provided as ``hit_frame_name`` on the response — but in
            practice it is the same frame the ray was issued in unless
            the server chose otherwise.
    """

    type_name: str
    distance_m: float
    hit_position: tuple[float, float, float]


def _get_client(robot) -> RayCastClient:
    """Return a cached :class:`RayCastClient` for ``robot``.

    Raises:
        Whatever ``robot.ensure_client`` raises on first creation — the
        callers below catch ``BosdynError`` to turn that into a warning.
    """

    key = getattr(robot, "_name", None) or id(robot)
    client = _client_cache.get(key)
    if client is None:
        client = robot.ensure_client(RayCastClient.default_service_name)
        _client_cache[key] = client
    return client


def cast_ray(
    robot,
    direction_xyz: tuple[float, float, float],
    *,
    origin_xyz: tuple[float, float, float] = (0.0, 0.0, 0.0),
    frame_name: str | None = None,
    types: list[int] | None = None,
    min_distance_m: float = 0.0,
) -> list[RayHit]:
    """Send a single raycast and return all hits as a list.

    Args:
        robot: A connected ``bosdyn.client.robot.Robot`` instance.
        direction_xyz: Direction vector of the ray in ``frame_name``
            coordinates.  Not required to be unit length — the server
            normalizes internally.
        origin_xyz: Ray origin in ``frame_name`` coordinates.  Defaults
            to ``(0, 0, 0)``, which for ``frame_name == BODY_FRAME_NAME``
            is the robot body origin.
        frame_name: The frame the ray is issued in.  Defaults to
            ``BODY_FRAME_NAME``.
        types: List of ``ray_cast_pb2.RayIntersection.Type`` int values
            to intersect against.  Defaults to all four documented
            types.  Pass an explicit list to opt out of (e.g.)
            ``TYPE_VOXEL_MAP`` on a base Spot where it may be empty.
        min_distance_m: Minimum distance behind the origin at which an
            intersection is accepted (per the BD API — "how far behind
            the ray an intersection can occur").  Defaults to 0.

    Returns:
        List of :class:`RayHit` instances, one per returned
        intersection.  An empty list means either the service was
        unavailable, the call failed, or no intersections were found —
        the caller can treat all three the same if it wants to, or
        inspect logs for which case applied.
    """

    if frame_name is None:
        frame_name = BODY_FRAME_NAME
    if types is None:
        types = list(_DEFAULT_TYPES)

    try:
        client = _get_client(robot)
    except BosdynError as exc:
        logger.warning("RayCast client unavailable: %s", exc)
        return []

    try:
        response = client.raycast(
            ray_origin=origin_xyz,
            ray_direction=direction_xyz,
            raycast_types=types,
            min_distance=min_distance_m,
            frame_name=frame_name,
        )
    except BosdynError as exc:
        logger.warning("RayCast RPC failed: %s", exc)
        return []

    hits: list[RayHit] = []
    for h in response.hits:
        try:
            type_name = ray_cast_pb2.RayIntersection.Type.Name(h.type)
        except ValueError:
            # Server returned an enum value this SDK version doesn't
            # know about — keep the numeric form so the caller can still
            # see something useful.
            type_name = f"TYPE_UNKNOWN({h.type})"
        hp = h.hit_position_in_hit_frame
        hits.append(
            RayHit(
                type_name=type_name,
                distance_m=float(h.distance_meters),
                hit_position=(float(hp.x), float(hp.y), float(hp.z)),
            )
        )
    return hits


def forward_ray_from_body(
    robot,
    max_distance_m: float = 5.0,
) -> list[RayHit]:
    """Convenience: cast a ray straight forward from the body origin.

    ``max_distance_m`` is not currently forwarded to the RPC — the BD
    ``raycast`` call does not accept a maximum distance, only a
    minimum.  The argument is retained as a hint for callers that want
    to post-filter hits (e.g. ``[h for h in hits if h.distance_m <= 5]``)
    and as a forward-compatible knob if a later SDK adds ``max_distance``
    support.
    """

    _ = max_distance_m  # retained for future filtering, see docstring
    return cast_ray(
        robot,
        direction_xyz=(1.0, 0.0, 0.0),
        origin_xyz=(0.0, 0.0, 0.0),
        frame_name=BODY_FRAME_NAME,
    )
