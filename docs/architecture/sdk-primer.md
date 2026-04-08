# Boston Dynamics SDK Primer

This page explains the core concepts of the Boston Dynamics (BD) Python SDK that underpin the entire voice control system. If you have never worked with the Spot SDK before, read this before diving into the module reference.

## SDK Package Structure

The project uses BD SDK v5.0.1.1. The main packages are:

```
bosdyn-client          # Core SDK — robot connection, commands, leases
bosdyn-mission         # Mission/Autowalk API
bosdyn-choreography-client  # Choreography API (available but not used)
```

All SDK code starts with `from bosdyn.client import ...`.

## Connecting to Spot

Every interaction with Spot follows the same pattern:

```python
from bosdyn.client import create_standard_sdk

# 1. Create an SDK instance (one per application)
sdk = create_standard_sdk("my_app_name")

# 2. Create a robot object (does not connect yet)
robot = sdk.create_robot("192.168.80.3")

# 3. Authenticate (uses credentials from the admin console)
robot.authenticate("username", "password")

# 4. Sync clocks (required before any commands)
robot.time_sync.wait_for_sync()
```

The `robot` object is your handle to the physical robot. From it, you create **service clients** to interact with specific subsystems.

## Service Clients

Each Spot subsystem exposes a gRPC service. You get a client for each one:

```python
from bosdyn.client.robot_command import RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.lease import LeaseClient
from bosdyn.client.power import PowerClient
from bosdyn.client.image import ImageClient
from bosdyn.client.graph_nav import GraphNavClient

cmd_client   = robot.ensure_client(RobotCommandClient.default_service_name)
state_client = robot.ensure_client(RobotStateClient.default_service_name)
lease_client = robot.ensure_client(LeaseClient.default_service_name)
# ... etc
```

`ensure_client()` creates the client once and caches it — safe to call multiple times.

### Key Clients Used in This Project

| Client | Purpose | Used in |
|--------|---------|---------|
| `RobotCommandClient` | Send movement commands (walk, turn, stand, sit) | `spot_dispatch.py` |
| `RobotStateClient` | Read battery, pose, motor state, faults | `spot_dispatch.py` |
| `LeaseClient` | Acquire control of the robot | `session.py` |
| `PowerClient` | Power motors on/off | `session.py` |
| `ImageClient` | Capture camera images | `spot_dispatch.py` |
| `GraphNavClient` | Autonomous navigation to waypoints | `spot_dispatch.py` |
| `EstopClient` | Software emergency stop | `estop_run.py`, `web_panel.py` |

## Leases

Only one client can control Spot at a time. **Leases** enforce this.

```python
lease_client = robot.ensure_client(LeaseClient.default_service_name)

# Acquire the lease (will fail if another client holds it)
lease = lease_client.acquire()

# Or forcefully take it (overrides the tablet or other clients)
lease = lease_client.take()
```

The project uses `lease_client.take()` to always override any existing claim (e.g., from the tablet). This is deliberate — the voice pipeline should always be able to take control.

### Lease Keepalive

Leases expire if not refreshed. `LeaseKeepAlive` handles this automatically:

```python
from bosdyn.client.lease import LeaseKeepAlive

keepalive = LeaseKeepAlive(lease_client, must_acquire=False, return_at_exit=True)
# Refreshes the lease in the background
# return_at_exit=True releases the lease when the keepalive is shut down
```

!!! warning "Lease contention"
    If someone picks up the Spot tablet and takes control, the tablet acquires the lease and the Jetson loses it. The voice pipeline will re-take the lease on the next command (since `spot_dispatch.py` maintains a persistent session), but any in-progress navigation will be interrupted.

## Time Sync

Spot requires all command timestamps to be in the robot's clock. The SDK handles this transparently, but you must sync clocks before sending commands:

```python
robot.time_sync.wait_for_sync()
```

This is done once during session setup. If time sync fails (e.g., network latency spike), you'll see `TimeSyncError` — just retry.

## E-Stop (Emergency Stop)

Spot will not move unless at least one E-Stop endpoint is active. This is a safety requirement — if all E-Stop endpoints disconnect, the robot stops immediately.

```python
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive

estop_client = robot.ensure_client(EstopClient.default_service_name)
endpoint = EstopEndpoint(estop_client, name="my_estop", estop_timeout=3.0)
endpoint.force_simple_setup()

# Start keepalive (sends heartbeats every ~1s)
keepalive = EstopKeepAlive(endpoint)
keepalive.allow()  # Tell the robot it's OK to move
```

If the keepalive stops sending heartbeats (process crash, network loss), the robot E-Stops within `estop_timeout` seconds (3s in this project).

**In this project**, E-Stop is managed by either:

- `scripts/estop_run.py` — terminal-based, Ctrl+C to stop
- `scripts/web_panel.py` — browser-based, claim/stop/release buttons

The voice control pipeline (`run_voice_control.py`) does **not** run its own E-Stop — it assumes one is already running.

## Sending Movement Commands

All movement goes through `RobotCommandBuilder` and `RobotCommandClient`:

```python
from bosdyn.client.robot_command import RobotCommandBuilder, blocking_stand

# Stand up
blocking_stand(cmd_client, timeout_sec=10)

# Sit down
from bosdyn.client.robot_command import blocking_sit
blocking_sit(cmd_client, timeout_sec=10)

# Walk forward 1 meter
from bosdyn.client.robot_command import RobotCommandBuilder
cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.5, v_y=0.0, v_rot=0.0)
cmd_client.robot_command(command=cmd, end_time_secs=time.time() + 2.0)

# Turn 90 degrees
import math
cmd = RobotCommandBuilder.synchro_velocity_command(v_x=0.0, v_y=0.0, v_rot=math.radians(60))
cmd_client.robot_command(command=cmd, end_time_secs=time.time() + 1.5)
```

### Velocity Commands

`synchro_velocity_command(v_x, v_y, v_rot)` is the workhorse:

| Parameter | Unit | Description |
|-----------|------|-------------|
| `v_x` | m/s | Forward (+) / backward (-) |
| `v_y` | m/s | Left (+) / right (-) strafe |
| `v_rot` | rad/s | Counter-clockwise (+) / clockwise (-) rotation |

Commands require an `end_time_secs` — the robot stops after this time if no new command is sent. This is a safety feature.

### Body Height

```python
from bosdyn.geometry import EulerZXY

cmd = RobotCommandBuilder.synchro_stand_command(
    body_height=0.1  # meters above default stance (-0.2 to +0.2 range)
)
cmd_client.robot_command(command=cmd)
```

## Reading Robot State

```python
state = state_client.get_robot_state()

# Battery
battery = state.battery_states[0]
charge = battery.charge_percentage.value  # 0-100
runtime = battery.estimated_runtime.seconds  # seconds remaining

# Motor power
is_powered = (state.power_state.motor_power_state ==
              state.power_state.STATE_ON)

# Faults
for fault in state.system_fault_state.faults:
    print(fault.name, fault.severity)
```

## Capturing Images

Spot has 5 fisheye cameras plus depth sensors:

```python
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2

image_client = robot.ensure_client(ImageClient.default_service_name)

# Request a JPEG from the front-left camera
responses = image_client.get_image_from_sources([
    image_pb2.ImageRequest(
        image_source_name="frontleft_fisheye_image",
        image_format=image_pb2.Image.FORMAT_JPEG,  # 1
        quality_percent=75,
    )
])

jpeg_bytes = responses[0].shot.image.data
```

### Camera Source Names

| Source | Position |
|--------|----------|
| `frontleft_fisheye_image` | Front-left (primary for VLM) |
| `frontright_fisheye_image` | Front-right |
| `left_fisheye_image` | Left side |
| `right_fisheye_image` | Right side |
| `back_fisheye_image` | Rear |

!!! note "Fisheye orientation"
    Fisheye cameras are mounted sideways on Spot. Images must be rotated 90 degrees clockwise for correct display. When mapping pixel coordinates back to the original frame (e.g., for depth lookup), un-rotate: `orig_x = rot_y`, `orig_y = H_orig - 1 - rot_x`.

## GraphNav (Autonomous Navigation)

GraphNav lets Spot navigate recorded maps autonomously:

```python
from bosdyn.client.graph_nav import GraphNavClient

graph_nav = robot.ensure_client(GraphNavClient.default_service_name)

# Navigate to a waypoint by ID
nav_response = graph_nav.navigate_to(
    waypoint_id="waypoint_abc123...",
    cmd_duration=30.0,  # timeout
)

# Check status
from bosdyn.api.graph_nav import nav_pb2
status = nav_response.status
if status == nav_pb2.NavigateToResponse.STATUS_OK:
    print("Navigation started")
elif status == nav_pb2.NavigateToResponse.STATUS_NO_PATH:
    print("No path found")
```

### Navigation Feedback Loop

Navigation is asynchronous. You poll for completion:

```python
while True:
    feedback = graph_nav.navigation_feedback()
    status = feedback.status

    if status == nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
        print("Arrived!")
        break
    elif status == nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
        print("Stuck — retrying with backtrack")
        break
    elif status == nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_LOST:
        print("Lost — need to re-localize")
        break

    time.sleep(0.5)
```

This pattern is exactly what `spot_dispatch.py` implements in `_navigate_waypoint_sequence()`, with re-localization and backtrack-on-stuck recovery.

## Keepalive Policies

BD SDK 5.0+ introduces keepalive policies that automatically stop the robot if a client disconnects unexpectedly. These can become "stale" if a previous session crashed without cleaning up.

```python
from bosdyn.client.keepalive import KeepaliveClient, remove_all_policies

keepalive_client = robot.ensure_client(KeepaliveClient.default_service_name)
remove_all_policies(keepalive_client)
```

The project clears stale policies on every session start (`session.py`) to avoid `KeepaliveMotorsOff` errors.

## Common Errors and What They Mean

| Error | Cause | Fix |
|-------|-------|-----|
| `ResourceAlreadyClaimedError` | Another client holds the lease | Use `lease_client.take()` instead of `acquire()` |
| `KeepaliveMotorsOffError` | Stale keepalive policy blocking power-on | Clear policies with `remove_all_policies()` |
| `EstoppedError` | Robot is E-Stopped | Release E-Stop from tablet or web panel |
| `TimeSyncError` | Clock sync failed | Retry `robot.time_sync.wait_for_sync()` |
| `NotLocalizedError` | GraphNav doesn't know robot's position | Re-localize with fiducial or known waypoint |
| `RetryableRpcError` | Transient network issue | Retry the request |

## Further Reading

- [Boston Dynamics SDK Documentation](https://dev.bostondynamics.com/)
- [Python SDK Examples on GitHub](https://github.com/boston-dynamics/spot-sdk)
- [API Reference (protobuf)](https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference)
