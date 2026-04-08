"""Map name/path resolution and last-used bookkeeping for GraphNav maps.

This module is the single source of truth for:
  - resolving a user-provided map argument (bare name, full path, or None) to
    an absolute Path on disk
  - tracking which map was most recently deployed (via maps/.last_used)
  - listing the maps available under the project's maps/ directory

It is consumed by:
  - scripts/wakespot.py            (orchestrator: resolves --map at startup)
  - src/voice_control/spot_dispatch.py  (voice load_map handler)
  - src/location_manager.py       (reads current map for namespacing)

Stub: function signatures only. Real implementation lands in W1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def resolve_map_path(name_or_path: Optional[str], project_root: Path) -> Path:
    """Resolve a map argument to an absolute directory path.

    Accepts:
      - None or "" or "latest": pick the last-used map (from maps/.last_used);
        fall back to the map whose `graph` file has the newest mtime.
      - A bare map name like "jackson": resolves to <project_root>/maps/jackson.
      - An absolute or relative path: returned as-is after validation.

    Raises FileNotFoundError if the resolved path doesn't exist or doesn't
    contain a valid `graph` file.
    """
    raise NotImplementedError("stub — implemented in W1")


def list_available_maps(project_root: Path) -> list[str]:
    """Return sorted names of all maps under <project_root>/maps/.

    A directory counts as a map only if it contains a `graph` file.
    """
    raise NotImplementedError("stub — implemented in W1")


def write_last_used(project_root: Path, map_name: str) -> None:
    """Record `map_name` as the most recently deployed map.

    Writes <project_root>/maps/.last_used containing exactly the map name
    (no trailing newline). Used by both wakespot.py (after startup upload) and
    the voice load_map handler (after a runtime switch).
    """
    raise NotImplementedError("stub — implemented in W1")


def read_last_used(project_root: Path) -> Optional[str]:
    """Return the contents of <project_root>/maps/.last_used, or None if absent."""
    raise NotImplementedError("stub — implemented in W1")
