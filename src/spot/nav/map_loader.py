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
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

# Filename inside <project_root>/maps/ that records the most recently deployed
# map. Single source of truth across wakespot, voice load_map, and the
# location manager's namespace key.
LAST_USED_FILENAME = ".last_used"

# Filename that must be present inside a map directory for it to be considered
# a valid GraphNav map.
GRAPH_FILENAME = "graph"


def _maps_dir(project_root: PathLike) -> Path:
    return Path(project_root) / "maps"


def _is_path_like(name_or_path: str) -> bool:
    """Decide whether a user-supplied string should be treated as a filesystem
    path (vs. a bare map name).

    A string is path-like if it contains a path separator, starts with `.`,
    `~`, or is absolute.
    """
    if not name_or_path:
        return False
    if os.sep in name_or_path or "/" in name_or_path:
        return True
    if name_or_path.startswith(("~", ".")):
        return True
    if os.path.isabs(name_or_path):
        return True
    return False


def _validate_map_dir(path: Path) -> Path:
    """Return `path` resolved to an absolute Path if it is a valid map dir.

    Raises FileNotFoundError otherwise.
    """
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(
            f"Map directory not found: {resolved}"
        )
    if not (resolved / GRAPH_FILENAME).is_file():
        raise FileNotFoundError(
            f"Map directory missing '{GRAPH_FILENAME}' file: {resolved}"
        )
    return resolved


def _format_available(project_root: Path) -> str:
    available = list_available_maps(project_root)
    if not available:
        return "(no maps found under maps/)"
    return ", ".join(available)


def resolve_map_path(
    name_or_path: Optional[str], project_root: PathLike
) -> Path:
    """Resolve a map argument to an absolute directory path.

    Accepts:
      - None or "" or "latest": pick the last-used map (from maps/.last_used);
        fall back to the map whose `graph` file has the newest mtime.
      - A bare map name like "jackson": resolves to <project_root>/maps/jackson.
      - An absolute or relative path: returned as-is after validation.

    Raises FileNotFoundError if the resolved path doesn't exist or doesn't
    contain a valid `graph` file.
    """
    project_root = Path(project_root)
    maps_dir = _maps_dir(project_root)

    # Case 1: empty / None / "latest" -> pick last-used or newest by mtime
    if name_or_path is None or name_or_path == "" or name_or_path == "latest":
        last = read_last_used(project_root) if name_or_path != "latest" else None
        if last:
            candidate = maps_dir / last
            if (candidate / GRAPH_FILENAME).is_file():
                return _validate_map_dir(candidate)
            # .last_used names a map that no longer exists -> fall through to
            # the mtime-based fallback below.

        # Fall back: pick the map directory whose `graph` file has the newest
        # mtime.
        if not maps_dir.is_dir():
            raise FileNotFoundError(
                f"No maps/ directory at {maps_dir}. Available maps: "
                f"{_format_available(project_root)}"
            )

        candidates = []
        for child in maps_dir.iterdir():
            if child.name.startswith("."):
                continue
            if not child.is_dir():
                continue
            graph = child / GRAPH_FILENAME
            if graph.is_file():
                candidates.append((graph.stat().st_mtime, child))
        if not candidates:
            raise FileNotFoundError(
                f"No maps with a '{GRAPH_FILENAME}' file found under {maps_dir}. "
                f"Available maps: {_format_available(project_root)}"
            )
        candidates.sort(key=lambda t: t[0], reverse=True)
        return _validate_map_dir(candidates[0][1])

    # Case 2: looks like a path -> validate as-is
    if _is_path_like(name_or_path):
        try:
            return _validate_map_dir(Path(name_or_path))
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{e}. Available maps under {maps_dir}: "
                f"{_format_available(project_root)}"
            ) from None

    # Case 3: bare name -> <project_root>/maps/<name>
    candidate = maps_dir / name_or_path
    try:
        return _validate_map_dir(candidate)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Map '{name_or_path}' not found at {candidate}. "
            f"Available maps: {_format_available(project_root)}"
        ) from None


def list_available_maps(project_root: PathLike) -> list[str]:
    """Return sorted names of all maps under <project_root>/maps/.

    A directory counts as a map only if it contains a `graph` file. Dotfiles
    (such as `.last_used`) are skipped.
    """
    project_root = Path(project_root)
    maps_dir = _maps_dir(project_root)
    if not maps_dir.is_dir():
        return []

    names: list[str] = []
    for child in maps_dir.iterdir():
        if child.name.startswith("."):
            continue
        if not child.is_dir():
            continue
        if (child / GRAPH_FILENAME).is_file():
            names.append(child.name)
    names.sort()
    return names


def write_last_used(project_root: PathLike, map_name: str) -> None:
    """Record `map_name` as the most recently deployed map.

    Writes <project_root>/maps/.last_used containing exactly the map name
    (no trailing newline). Atomic: writes to a sibling .tmp file then renames,
    so a crash mid-write cannot leave the marker corrupted.
    """
    project_root = Path(project_root)
    maps_dir = _maps_dir(project_root)
    maps_dir.mkdir(parents=True, exist_ok=True)

    target = maps_dir / LAST_USED_FILENAME
    tmp = maps_dir / (LAST_USED_FILENAME + ".tmp")
    tmp.write_text(map_name)
    os.replace(tmp, target)


def read_last_used(project_root: PathLike) -> Optional[str]:
    """Return the contents of <project_root>/maps/.last_used, or None if absent.

    Returns None for missing files, empty files, or any read error.
    """
    project_root = Path(project_root)
    target = _maps_dir(project_root) / LAST_USED_FILENAME
    if not target.is_file():
        return None
    try:
        contents = target.read_text().strip()
    except OSError:
        return None
    return contents or None
