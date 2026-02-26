# location_manager.py

`src/location_manager.py` -- Waypoint name storage (JSON-backed).

Persists human-readable names for GraphNav waypoint IDs. When the user says
"save location kitchen", the current waypoint ID is stored under the key
`"kitchen"`. Later, "go to kitchen" resolves to that waypoint ID.

## Key Functions

### `save_location(name, waypoint_id) -> bool`

Save a named location mapping.

```python
def save_location(name: str, waypoint_id: str) -> bool
```

Names are normalized to lowercase. Creates the file and parent directory
if they do not exist. Overwrites if the name already exists.

### `load_location(name) -> str | None`

Look up a waypoint ID by name.

```python
def load_location(name: str) -> Optional[str]
```

Returns the waypoint ID string, or None if not found. Lookup is
case-insensitive (stored lowercase).

### `load_all_locations() -> dict`

Load the full name-to-waypoint mapping.

```python
def load_all_locations() -> Dict[str, str]
```

Returns an empty dict if the file does not exist or is malformed.

### `list_locations() -> dict`

Alias for `load_all_locations()`. Returns all saved location mappings.

## Storage

File: `locations.json` in the project root.

```json
{
  "kitchen": "quiet-toad-B5X.hMjvdIhVc7t6FuAFNQ==",
  "lab": "dizzy-cat-K2R.pLmQwNjXc9u8GuBGOA==",
  "hallway": "brave-owl-M3T.xYnRzPkZd0v9HvCHPA=="
}
```

Names are normalized to lowercase. Spaces in names passed from the LLM or
regex parser are converted to underscores by the caller (`llm_brain.py`
normalizes to `lowercase_with_underscores`, `intent.py` does the same).

## Usage Example

```python
from src.location_manager import save_location, load_location, list_locations

# Save current position
save_location("kitchen", "quiet-toad-B5X.hMjvdIhVc7t6FuAFNQ==")

# Look up later
wp_id = load_location("kitchen")
if wp_id:
    graph_nav_client.navigate_to(wp_id, ...)

# List all saved locations
all_locs = list_locations()
# {"kitchen": "quiet-toad-...", "lab": "dizzy-cat-..."}
```

## Dependencies

**Project modules:** none (standalone)

**External packages:** `json` (stdlib), `pathlib` (stdlib)
