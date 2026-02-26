# graph_nav_utils.py

`src/graph_nav_utils.py` -- GraphNav map management.

Handles uploading recorded maps to the robot and initializing localization.
Maps consist of a graph file, waypoint snapshots, and edge snapshots stored
in a directory structure.

## Key Functions

### `upload_graph_and_snapshots(robot, upload_path) -> bool`

Upload a complete GraphNav map to the robot.

```python
def upload_graph_and_snapshots(robot, upload_path: str) -> bool
```

**Args:**
- `robot` -- Authenticated `bosdyn.client.Robot` instance.
- `upload_path` -- Directory containing `graph`, `waypoint_snapshots/`,
  and `edge_snapshots/`.

Clears existing map on the robot before uploading. Uploads graph structure
first, then waypoint snapshots, then edge snapshots. Prints progress every
5 snapshots.

### `initialize_localization(robot, use_fiducial) -> bool`

Initialize robot localization to the uploaded map.

```python
def initialize_localization(robot, use_fiducial: bool = True) -> bool
```

Two strategies:
- **Fiducial-based** (`use_fiducial=True`): Robot must see a fiducial
  (AprilTag) from the map. Most accurate.
- **Waypoint-based** (`use_fiducial=False`): Robot must be physically near
  the first waypoint. Uses odometry with +/-20cm / +/-20 deg search.

Skips initialization if already localized.

### `list_waypoints(robot) -> list[str]`

Download the graph and return all waypoint IDs.

```python
def list_waypoints(robot) -> List[str]
```

### `get_graph_waypoint_order(robot) -> list[str]`

Return all waypoint IDs in graph recording order. The graph's waypoints list
preserves the order they were recorded, representing the natural traversal
path.

```python
def get_graph_waypoint_order(robot) -> List[str]
```

### `get_localization_state(robot) -> dict | None`

Get current localization state.

```python
def get_localization_state(robot) -> Optional[dict]
```

**Returns:**
```python
{"waypoint_id": "abc-123", "is_localized": True}
# or
None  # on error
```

## Map Directory Structure

```
maps/building1/
    graph                    # Serialized Graph protobuf
    waypoint_snapshots/
        snapshot_0001        # Serialized WaypointSnapshot protobufs
        snapshot_0002
        ...
    edge_snapshots/
        edge_snapshot_0001   # Serialized EdgeSnapshot protobufs
        ...
```

Maps are recorded using BD's tablet app or the `graph_nav_command_line`
example. The `maps/` directory is in `.gitignore` (maps are device-specific).

## Usage Example

```python
from src.graph_nav_utils import upload_graph_and_snapshots, initialize_localization

# Upload map and localize (typically done in spot_session)
if upload_graph_and_snapshots(robot, "maps/building1"):
    initialize_localization(robot, use_fiducial=True)
```

## Dependencies

**Project modules:** none (standalone)

**External packages:** `bosdyn.client` (graph_nav.GraphNavClient,
frame_helpers), `bosdyn.api` (graph_nav_pb2, map_pb2, nav_pb2)
