"""Cached camera intrinsics from Spot's image service.

Per-source pinhole intrinsics fetched once via ``list_image_sources()``
at first use and stored in a module-level cache keyed by robot name.
The probe at ``docs/project/perception_probe_results.md`` confirmed
every image source on this Spot populates the ``pinhole`` field of the
``camera_models`` oneof, so we only handle that branch — Kannala-Brandt
and pinhole_brown_conrady sources are skipped with a debug log.

The cache exists for two consumers:

* ``visual_nav._bearing_in_body_frame`` — needs the ``ImageSource`` proto
  to call ``bosdyn.client.image.pixel_to_camera_space``.
* ``visual_nav.navigate_to_object`` — needs the ``PinholeModel`` proto
  to populate ``WalkToObjectInImage.camera_model`` for the SDK
  manipulation API path.

Both consumers can hit the cache repeatedly without paying the
``list_image_sources`` RPC each time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from bosdyn.api import image_pb2
from bosdyn.client.exceptions import Error as BosdynError
from bosdyn.client.image import ImageClient

logger = logging.getLogger(__name__)


@dataclass
class Intrinsics:
    """Pinhole camera intrinsics for one image source.

    Attributes:
        source_name: The image source name, e.g. ``"frontleft_fisheye_image"``.
        fx, fy: Focal length in pixels.
        cx, cy: Principal point in pixels.
        cols, rows: Native sensor resolution.
        image_source_proto: The full ``ImageSource`` proto, retained so
            consumers can pass it to ``pixel_to_camera_space`` or copy
            its ``pinhole`` field into a ``WalkToObjectInImage`` request
            without re-fetching.
    """

    source_name: str
    fx: float
    fy: float
    cx: float
    cy: float
    cols: int
    rows: int
    image_source_proto: image_pb2.ImageSource


# Module-level caches keyed by robot name. ``_sources_cache`` keeps the
# raw list returned by ``list_image_sources()`` so consumers that need
# the proto for a non-pinhole source can still get it; ``_intrinsics_cache``
# stores the unpacked dataclass per pinhole source.
_sources_cache: dict[str, list[image_pb2.ImageSource]] = {}
_intrinsics_cache: dict[str, dict[str, Intrinsics]] = {}


def _robot_key(robot) -> str:
    return getattr(robot, "_name", None) or str(id(robot))


def _ensure_loaded(robot) -> None:
    """Populate the cache for ``robot`` if not already done.

    Failures are logged once and the cache is set to empty so we don't
    retry the failing RPC on every call.
    """

    key = _robot_key(robot)
    if key in _intrinsics_cache:
        return

    try:
        client = robot.ensure_client(ImageClient.default_service_name)
        sources = list(client.list_image_sources())
    except BosdynError as exc:
        logger.warning("list_image_sources failed for %s: %s", key, exc)
        _sources_cache[key] = []
        _intrinsics_cache[key] = {}
        return
    except Exception as exc:  # defensive — image client construction errors
        logger.warning("ImageClient unavailable for %s: %s", key, exc)
        _sources_cache[key] = []
        _intrinsics_cache[key] = {}
        return

    _sources_cache[key] = sources
    intrinsics_map: dict[str, Intrinsics] = {}
    for src in sources:
        which = src.WhichOneof("camera_models")
        if which != "pinhole":
            logger.debug(
                "Skipping non-pinhole source %s (model=%s)", src.name, which
            )
            continue
        i = src.pinhole.intrinsics
        intrinsics_map[src.name] = Intrinsics(
            source_name=src.name,
            fx=float(i.focal_length.x),
            fy=float(i.focal_length.y),
            cx=float(i.principal_point.x),
            cy=float(i.principal_point.y),
            cols=int(src.cols),
            rows=int(src.rows),
            image_source_proto=src,
        )
    _intrinsics_cache[key] = intrinsics_map
    logger.info(
        "Cached intrinsics for %s: %d pinhole sources of %d total",
        key,
        len(intrinsics_map),
        len(sources),
    )


def get_intrinsics(robot, source_name: str) -> Optional[Intrinsics]:
    """Return cached pinhole intrinsics for ``source_name`` on ``robot``.

    Returns ``None`` if the source isn't pinhole, isn't found, or the
    initial ``list_image_sources`` RPC failed. First call per robot
    fetches from the SDK; subsequent calls hit the cache.
    """

    _ensure_loaded(robot)
    return _intrinsics_cache.get(_robot_key(robot), {}).get(source_name)


def get_image_source_proto(
    robot, source_name: str
) -> Optional[image_pb2.ImageSource]:
    """Return the cached ``ImageSource`` proto for ``source_name``.

    Useful when a consumer needs the full proto regardless of camera
    model (e.g. ``pixel_to_camera_space`` accepts any pinhole-shaped
    source). Returns ``None`` if the source isn't in the cache.
    """

    _ensure_loaded(robot)
    for src in _sources_cache.get(_robot_key(robot), []):
        if src.name == source_name:
            return src
    return None


def invalidate(robot) -> None:
    """Drop the cache for ``robot``. Useful after a firmware change."""

    key = _robot_key(robot)
    _sources_cache.pop(key, None)
    _intrinsics_cache.pop(key, None)
