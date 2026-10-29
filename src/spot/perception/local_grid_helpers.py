"""Fetch + decode Spot ``LocalGrid`` responses into numpy arrays.

The helpers here sit on top of :class:`bosdyn.client.local_grid.LocalGridClient`
and turn its ``LocalGridResponse`` protos into :class:`DecodedGrid` instances
that consumers can treat as ordinary 2D numpy arrays.  Responsibilities:

* discover the grid type names this Spot exposes (cached per-robot),
* dispatch both ``ENCODING_RAW`` and ``ENCODING_RLE`` cell buffers,
* translate every documented ``cell_format`` to a numpy dtype,
* apply ``cell_value_scale`` / ``cell_value_offset`` to recover physical
  units,
* decode the ``unknown_cells`` bitfield into a boolean mask and set those
  cells to ``NaN`` in the returned array so callers can use
  ``np.nanmean`` and friends without worrying about masking.

The module never touches the robot beyond ``get_local_grid_types`` and
``get_local_grids`` RPCs.  Any decode-time failure is logged via
``logger.warning`` and the offending grid is silently skipped — callers
should treat a missing key in the returned dict as "not available".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from bosdyn.api import local_grid_pb2
from bosdyn.client.exceptions import Error as BosdynError
from bosdyn.client.local_grid import LocalGridClient

logger = logging.getLogger(__name__)


# Mapping from LocalGrid.CellFormat enum values to numpy dtypes.  Anything
# not in this table is treated as an unsupported cell format and skipped
# with a warning at decode time.
_CELL_FORMAT_TO_DTYPE: dict[int, np.dtype] = {
    local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT32: np.dtype(np.float32),
    local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT64: np.dtype(np.float64),
    local_grid_pb2.LocalGrid.CELL_FORMAT_INT8: np.dtype(np.int8),
    local_grid_pb2.LocalGrid.CELL_FORMAT_UINT8: np.dtype(np.uint8),
    local_grid_pb2.LocalGrid.CELL_FORMAT_INT16: np.dtype(np.int16),
    local_grid_pb2.LocalGrid.CELL_FORMAT_UINT16: np.dtype(np.uint16),
}


# Module-level caches keyed by ``robot._name``.  Re-using ``ensure_client``
# directly would work, but keeping the LocalGridClient handle and the
# discovered grid-type list together in one place gives the discovery RPC
# a natural home and lets callers invalidate everything at once by simply
# dropping the robot key.
_client_cache: dict[str, LocalGridClient] = {}
_grid_names_cache: dict[str, list[str]] = {}


@dataclass
class DecodedGrid:
    """A single decoded :class:`bosdyn.api.local_grid_pb2.LocalGrid`.

    Attributes:
        name: The grid type name, e.g. ``"obstacle_distance"``.
        array: 2D numpy array shaped ``(num_cells_y, num_cells_x)`` in
            ``float64`` with ``cell_value_scale`` / ``cell_value_offset``
            already applied.  Cells marked unknown by the
            ``unknown_cells`` bitfield are set to ``NaN``.
        cell_size_m: Edge length of a single cell in meters.
        frame: The frame name in which the grid is expressed (typically
            ``"gpe"``).  Matches ``LocalGrid.frame_name_local_grid_data``.
        extent_m: ``(width_m, height_m)`` total covered area in meters,
            i.e. ``(num_cells_x * cell_size, num_cells_y * cell_size)``.
        transforms_snapshot: The opaque ``FrameTreeSnapshot`` proto carried
            with the grid — consumers pass this to ``get_a_tform_b`` to
            reach body/vision frames.
        unknown_mask: Boolean mask of shape ``(num_cells_y, num_cells_x)``
            with ``True`` where the original cell was flagged unknown.
            ``None`` if the response carried no ``unknown_cells`` bytes.
        acquisition_time_seconds: Capture timestamp as a single float
            (``seconds + nanos / 1e9``).
    """

    name: str
    array: np.ndarray
    cell_size_m: float
    frame: str
    extent_m: tuple[float, float]
    transforms_snapshot: Any
    unknown_mask: np.ndarray | None
    acquisition_time_seconds: float


def _get_client(robot) -> LocalGridClient:
    """Return a cached :class:`LocalGridClient` for ``robot``.

    Caching by ``robot._name`` keeps us from paying the
    ``ensure_client`` lookup on every fetch and gives the grid-names
    cache a key to share.
    """

    key = getattr(robot, "_name", None) or id(robot)
    client = _client_cache.get(key)
    if client is None:
        client = robot.ensure_client(LocalGridClient.default_service_name)
        _client_cache[key] = client
    return client


def list_grid_type_names(robot) -> list[str]:
    """Return the list of grid type names this robot exposes (cached).

    The first call issues a ``GetLocalGridTypes`` RPC; subsequent calls
    return the cached list.  If the RPC fails we log a warning and return
    an empty list (the caller will see "no grids available" and can
    decide how to degrade).
    """

    key = getattr(robot, "_name", None) or id(robot)
    cached = _grid_names_cache.get(key)
    if cached is not None:
        return cached

    try:
        client = _get_client(robot)
        type_protos = client.get_local_grid_types()
    except BosdynError as exc:
        logger.warning("GetLocalGridTypes failed: %s", exc)
        _grid_names_cache[key] = []
        return []

    # ``get_local_grid_types`` returns a RepeatedComposite of
    # ``LocalGridType`` protos, each with a single ``name`` field.
    names: list[str] = [t.name for t in type_protos if t.name]
    _grid_names_cache[key] = names
    return names


def _decode_cells_raw(data: bytes, dtype: np.dtype, n_cells: int) -> np.ndarray | None:
    """Decode a ``ENCODING_RAW`` payload into a flat numpy array.

    Returns ``None`` on any mismatch so the caller can log and skip.
    """

    itemsize = dtype.itemsize
    if len(data) != itemsize * n_cells:
        logger.warning(
            "RAW grid payload length %d does not match %d cells * %d bytes",
            len(data),
            n_cells,
            itemsize,
        )
        return None
    # ``frombuffer`` aliases the bytes — copy so callers can safely mutate
    # (we set NaNs on the array in-place below).
    return np.frombuffer(data, dtype=dtype).copy()


def _decode_cells_rle(
    data: bytes,
    rle_counts,
    dtype: np.dtype,
    n_cells: int,
) -> np.ndarray | None:
    """Decode a ``ENCODING_RLE`` payload into a flat numpy array.

    ``grid.data`` holds the unique values in the matched dtype and
    ``grid.rle_counts`` holds a parallel int32 array of run lengths; each
    value in ``data`` corresponds to exactly one count in ``rle_counts``.
    ``sum(rle_counts)`` must equal ``n_cells``.
    """

    itemsize = dtype.itemsize
    counts = np.asarray(list(rle_counts), dtype=np.int64)
    if len(data) != itemsize * counts.size:
        logger.warning(
            "RLE grid payload has %d bytes for %d values (%d each); mismatch",
            len(data),
            counts.size,
            itemsize,
        )
        return None
    values = np.frombuffer(data, dtype=dtype)
    total = int(counts.sum())
    if total != n_cells:
        logger.warning(
            "RLE grid expanded length %d does not match %d expected cells",
            total,
            n_cells,
        )
        return None
    # ``np.repeat`` expands ``values`` according to ``counts`` — e.g.
    # values=[5, 2, 9], counts=[3, 1, 2] -> [5, 5, 5, 2, 9, 9].
    return np.repeat(values, counts)


def _decode_unknown_mask(unknown_bytes: bytes, n_cells: int) -> np.ndarray | None:
    """Decode the ``unknown_cells`` field into a boolean flat array.

    Two on-the-wire formats are supported:

    * **One byte per cell** (empirically observed on this Spot, firmware
      5.x): ``len(unknown_bytes) == n_cells`` and each byte is 0 (known)
      or 1 (unknown).
    * **Bitfield, LSB-first** (per the BD documentation):
      ``len(unknown_bytes) == ceil(n_cells / 8)`` and cell ``i`` is
      unknown iff bit ``i % 8`` of byte ``i // 8`` is set.

    The format is detected from the byte length. Anything else is
    logged as a warning and treated as "no unknown info".

    Returns ``None`` if the response carried no bytes at all (meaning
    the server has no opinion about unknown cells for this grid).
    """

    if not unknown_bytes or len(unknown_bytes) == 0:
        return None

    raw = np.frombuffer(unknown_bytes, dtype=np.uint8)

    # One-byte-per-cell format (empirically observed on this Spot — see
    # docs/project/perception_probe_results.md). Each byte holds 0 or 1.
    if len(raw) == n_cells:
        return raw.astype(bool)

    # Bitfield format per BD docs.
    bitfield_bytes = (n_cells + 7) // 8
    if len(raw) == bitfield_bytes:
        bits = np.unpackbits(raw, bitorder="little")
        return bits[:n_cells].astype(bool)

    logger.warning(
        "unknown_cells has %d bytes, expected %d (1 byte/cell) or %d "
        "(bitfield) for %d cells — treating as no unknown info",
        len(raw),
        n_cells,
        bitfield_bytes,
        n_cells,
    )
    return None


def _decode_grid(grid) -> DecodedGrid | None:
    """Convert a single ``LocalGrid`` proto into a :class:`DecodedGrid`.

    Returns ``None`` (with a warning) if the grid has an unsupported
    cell format, encoding, or a malformed payload.
    """

    name = grid.local_grid_type_name
    extent = grid.extent
    n_x = int(extent.num_cells_x)
    n_y = int(extent.num_cells_y)
    n_cells = n_x * n_y
    if n_cells <= 0:
        logger.warning("Grid %r has zero cells (%dx%d), skipping", name, n_x, n_y)
        return None

    dtype = _CELL_FORMAT_TO_DTYPE.get(grid.cell_format)
    if dtype is None:
        logger.warning(
            "Grid %r uses unsupported cell_format %s, skipping",
            name,
            grid.cell_format,
        )
        return None

    if grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RAW:
        flat = _decode_cells_raw(grid.data, dtype, n_cells)
    elif grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RLE:
        flat = _decode_cells_rle(grid.data, grid.rle_counts, dtype, n_cells)
    else:
        logger.warning(
            "Grid %r uses unsupported encoding %s, skipping",
            name,
            grid.encoding,
        )
        return None

    if flat is None:
        return None

    # Shape: rows are Y, columns are X.  BD documents the buffer as
    # row-major with X varying fastest, so ``reshape((n_y, n_x))`` yields
    # ``array[y, x]`` indexing — which is what ``np.meshgrid(..., indexing='xy')``
    # consumers expect.
    try:
        raw_2d = flat.reshape((n_y, n_x))
    except ValueError as exc:
        logger.warning("Grid %r reshape(%d, %d) failed: %s", name, n_y, n_x, exc)
        return None

    # Apply scale + offset to get physical values in float64.  The
    # default scale is 1.0 (proto default is 0.0, which would zero every
    # cell — treat that as "unset" per the BD convention).
    scale = grid.cell_value_scale if grid.cell_value_scale else 1.0
    offset = grid.cell_value_offset
    physical = raw_2d.astype(np.float64) * scale + offset

    unknown_mask_flat = _decode_unknown_mask(grid.unknown_cells, n_cells)
    if unknown_mask_flat is not None:
        unknown_mask = unknown_mask_flat.reshape((n_y, n_x))
        # NaN-mask unknown cells in-place so callers can ignore them with
        # ``np.nanmean`` / ``np.nanmin`` without having to re-apply the
        # mask themselves.
        physical[unknown_mask] = np.nan
    else:
        unknown_mask = None

    cell_size = float(extent.cell_size)
    acq = grid.acquisition_time
    acq_seconds = float(acq.seconds) + float(acq.nanos) / 1e9

    return DecodedGrid(
        name=name,
        array=physical,
        cell_size_m=cell_size,
        frame=grid.frame_name_local_grid_data,
        extent_m=(n_x * cell_size, n_y * cell_size),
        transforms_snapshot=grid.transforms_snapshot,
        unknown_mask=unknown_mask,
        acquisition_time_seconds=acq_seconds,
    )


def fetch_grids(
    robot,
    names: list[str] | None = None,
) -> dict[str, DecodedGrid]:
    """Fetch and decode local grids by name.

    Args:
        robot: A connected ``bosdyn.client.robot.Robot`` instance.
        names: Optional list of grid type names to fetch.  If ``None``,
            we discover all available grid types via
            :func:`list_grid_type_names` (which is cached) and fetch
            every one.

    Returns:
        A dict mapping ``name`` -> :class:`DecodedGrid` for grids that
        were successfully fetched and decoded.  Grids with a non-OK
        response status, an unsupported cell format, or a malformed
        payload are silently skipped (but logged via ``logger.warning``).
        On RPC failure the dict is empty.
    """

    if names is None:
        names = list_grid_type_names(robot)
    if not names:
        return {}

    try:
        client = _get_client(robot)
        responses = client.get_local_grids(list(names))
    except BosdynError as exc:
        logger.warning("GetLocalGrids(%s) failed: %s", names, exc)
        return {}

    decoded: dict[str, DecodedGrid] = {}
    for response in responses:
        if response.status != local_grid_pb2.LocalGridResponse.STATUS_OK:
            logger.warning(
                "Grid %r returned status %s, skipping",
                response.local_grid_type_name,
                response.status,
            )
            continue
        result = _decode_grid(response.local_grid)
        if result is not None:
            decoded[result.name] = result
    return decoded
