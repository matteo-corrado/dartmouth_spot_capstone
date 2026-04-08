"""Location manager for saving and loading named waypoints.

Automatically saves current waypoint positions when you say "save location [name]".

The on-disk format is namespaced by map (v2). Legacy flat dicts are migrated
to v2 on first read:

    v1 (legacy):  {"kitchen": "wp-id-abc", "lobby": "wp-id-def"}

    v2:           {
                    "_format": "v2",
                    "maps": {
                      "thayer_map": {"kitchen": "wp-id-abc", ...},
                      "jackson":    {"office": "wp-id-ghi", ...}
                    }
                  }

The "current map" namespace is read from <project_root>/maps/.last_used via
src.map_loader.read_last_used(). If that file is absent, the slice key is
"_legacy" — the legacy flat data lands there on migration so it can still be
recovered after the user runs wakespot for the first time.

The public API (save_location / load_location / load_all_locations /
list_locations) is intentionally unchanged: callers continue to operate on
the *current map's* slice and don't need to know about namespacing.
"""
import json
import os
import pathlib
from typing import Optional, Dict, Any

from src.map_loader import read_last_used

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
LOCATIONS_FILE = PROJECT_ROOT / "locations.json"

# Slice key used when no map has been deployed yet (i.e. maps/.last_used does
# not exist). Migrated legacy locations land under this key on first read.
LEGACY_KEY = "_legacy"

# Format marker recorded inside the v2 file. Presence of "_format" in the
# loaded JSON is what tells us we don't need to migrate.
FORMAT_VERSION = "v2"


def _current_map_key() -> str:
    """Return the slice key for the currently-deployed map.

    Reads <project_root>/maps/.last_used. Falls back to LEGACY_KEY if the
    marker file is absent or empty.
    """
    name = read_last_used(PROJECT_ROOT)
    return name if name else LEGACY_KEY


def _empty_v2() -> Dict[str, Any]:
    return {"_format": FORMAT_VERSION, "maps": {}}


def _atomic_write(data: Dict[str, Any]) -> None:
    """Write `data` to LOCATIONS_FILE atomically."""
    LOCATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = LOCATIONS_FILE.with_suffix(LOCATIONS_FILE.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, LOCATIONS_FILE)


def _load_raw() -> Dict[str, Any]:
    """Load the raw locations file, migrating v1 -> v2 on first read.

    - Missing file -> empty v2 dict (not written to disk).
    - Corrupted / empty file -> empty v2 dict (not written to disk).
    - Legacy flat dict -> wrapped under the current map key, written back to
      disk before returning so the migration runs exactly once.
    - v2 dict -> returned as-is.
    """
    if not LOCATIONS_FILE.exists():
        return _empty_v2()

    try:
        with open(LOCATIONS_FILE, "r") as f:
            loaded = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty_v2()

    if not isinstance(loaded, dict):
        return _empty_v2()

    # Already v2 -> return as-is, but defensively ensure the "maps" sub-dict
    # exists.
    if loaded.get("_format") == FORMAT_VERSION:
        if "maps" not in loaded or not isinstance(loaded.get("maps"), dict):
            loaded["maps"] = {}
        return loaded

    # Legacy flat dict {name: waypoint_id}. Wrap under the current map key
    # and persist the migrated form so subsequent reads skip this branch.
    legacy_slice = {
        k: v for k, v in loaded.items() if isinstance(v, str)
    }
    migrated: Dict[str, Any] = {
        "_format": FORMAT_VERSION,
        "maps": {_current_map_key(): legacy_slice},
    }
    try:
        _atomic_write(migrated)
        print(
            f"[location_manager] Migrated legacy locations.json to v2 "
            f"under map slice '{_current_map_key()}' "
            f"({len(legacy_slice)} location(s))."
        )
    except OSError as e:
        # If we can't write back (e.g. read-only mount), still return the
        # in-memory migrated form so callers see consistent data.
        print(f"[location_manager] WARN: failed to persist migration: {e}")
    return migrated


def _slice_for_current_map(data: Dict[str, Any]) -> Dict[str, str]:
    """Return the dict of {name: waypoint_id} for the current map slice.

    Returns an empty dict if the current map has no slice yet (without
    creating one — that only happens on save).
    """
    maps = data.get("maps", {})
    return maps.get(_current_map_key(), {}) or {}


def save_location(name: str, waypoint_id: str) -> bool:
    """Save a named location with its waypoint ID under the current map slice.

    Args:
        name: Human-readable location name (e.g., "A", "B", "kitchen")
        waypoint_id: GraphNav waypoint ID from current localization

    Returns:
        True if saved successfully
    """
    data = _load_raw()
    maps = data.setdefault("maps", {})
    key = _current_map_key()
    slice_dict = maps.setdefault(key, {})
    slice_dict[name.lower()] = waypoint_id

    _atomic_write(data)

    print(f"\u2713 Saved location '{name}' -> waypoint {waypoint_id} (map: {key})")
    return True


def load_location(name: str) -> Optional[str]:
    """Load waypoint ID for a named location from the current map slice.

    Args:
        name: Location name (e.g., "B", "kitchen")

    Returns:
        Waypoint ID string, or None if not found in the current map.
    """
    data = _load_raw()
    return _slice_for_current_map(data).get(name.lower())


def load_all_locations() -> Dict[str, str]:
    """Load all saved locations for the current map slice.

    Returns:
        Dict mapping location names to waypoint IDs (current map only).
    """
    data = _load_raw()
    # Return a copy so callers can mutate without corrupting the file.
    return dict(_slice_for_current_map(data))


def list_locations() -> Dict[str, str]:
    """List all saved location mappings for the current map slice."""
    return load_all_locations()
