# session.py

`src/session.py` -- Spot robot session context manager.

Wraps the full lifecycle of a Spot connection: authentication, time sync,
lease acquisition, power-on, stand, and cleanup on exit. Used by
`spot_dispatch.py` to maintain a persistent robot session.

## Key Function

### `spot_session(...)`

Context manager that yields a session dict with all BD SDK clients.

```python
@contextmanager
def spot_session(
    hostname: str = BOSDYN_ROBOT_IP,
    username: str = BOSDYN_CLIENT_USERNAME,
    password: str = BOSDYN_CLIENT_PASSWORD,
    stand_on_enter: bool = True,
    sit_on_exit: bool = True,
    upload_map: bool = False,
    map_path: str = None,
    auto_localize: bool = False
)
```

**Yields:**
```python
{
    "robot": robot,              # bosdyn.client.Robot
    "lease_client": lease_client,
    "cmd": cmd_client,           # RobotCommandClient
    "power": power_client,       # PowerClient
    "state": state_client,       # RobotStateClient
}
```

## Lifecycle

### On enter

1. Create SDK and robot instance.
2. Authenticate with username/password.
3. Wait for time sync.
4. Clear stale keepalive policies (prevents "KeepaliveMotorsOff" errors from
   previous crashed sessions).
5. Force-take lease (overrides stale claims).
6. Start lease keepalive.
7. If `stand_on_enter`:
   - Power on motors (with retry on keepalive/estop errors).
   - Stand via `blocking_stand()`.
8. If `upload_map` and `map_path`:
   - Upload GraphNav map.
   - Optionally auto-localize via fiducial.

### On exit

1. If `sit_on_exit`: `blocking_sit()`.
2. Power off motors (graceful, not cut immediately).
3. Shutdown lease keepalive (returns lease).

## Configuration

Credentials are loaded from `src/config.py` which reads environment variables:

| Variable | Description |
|----------|-------------|
| `BOSDYN_ROBOT_IP` | Robot hostname or IP |
| `BOSDYN_CLIENT_USERNAME` | Authentication username |
| `BOSDYN_CLIENT_PASSWORD` | Authentication password |

## Usage Example

```python
from src.session import spot_session

# Full lifecycle (stand on enter, sit + power off on exit)
with spot_session() as session:
    cmd = session["cmd"]
    state = session["state"]
    robot_state = state.get_robot_state()
    print(f"Battery: {robot_state.power_state.locomotion_charge_percentage.value}%")

# Keep robot standing after context exits (for voice control)
with spot_session(sit_on_exit=False) as session:
    pass  # Robot stays standing

# With map upload and auto-localization
with spot_session(upload_map=True, map_path="maps/building1",
                  auto_localize=True) as session:
    pass
```

## Dependencies

**Project modules:** `src.config` (credentials), `src.graph_nav_utils`
(optional, for map upload)

**External packages:** `bosdyn.client` (create_standard_sdk, LeaseClient,
RobotStateClient, PowerClient, TimeSyncClient, RobotCommandClient,
KeepaliveClient)
