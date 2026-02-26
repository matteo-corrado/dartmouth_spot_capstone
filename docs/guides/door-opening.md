# Door Opening

Spot can open push-bar doors using its arm and the AutoPush command from the BD SDK.

!!! warning "Status: On Ice"
    Door opening is implemented but needs further testing with the physical robot. The `--vlm --depth 0.5` configuration has not been fully validated.

---

## How It Works

1. **Calibrate** — Use `calibrate_door.py` to identify the push point on the door from the front camera
2. **Configure** — Save door parameters to `door_config.json`
3. **Execute** — Voice command "open the door" triggers the AutoPush sequence

The system uses the BD SDK's `AutoPushCommand`, which takes:
- A 3D push point (from camera + depth estimation)
- The hinge side (left or right)

---

## Calibration

### Step 1: Position Spot facing the door

Drive Spot to stand ~1.5m in front of the door, facing it directly.

### Step 2: Run the calibration script

```bash
python scripts/calibrate_door.py
```

This captures the front camera image (fisheye, rotated 90 CW for display) and lets you click on the push bar location. The pixel coordinates and hinge side are saved to `door_config.json`.

### Step 3: Verify the config

```json
{
  "camera_source": "frontleft_fisheye_image",
  "handle_pixel_x_rotated": 140.0,
  "handle_pixel_y_rotated": 165.0,
  "hinge_side": "left"
}
```

---

## Testing

### Standalone test (no voice control)

```bash
python scripts/test_door_standalone.py
```

Options:
```bash
python scripts/test_door_standalone.py --vlm          # Use VLM to find push point
python scripts/test_door_standalone.py --depth 0.5    # Override depth estimate (meters)
python scripts/test_door_standalone.py --vlm --depth 0.5  # Recommended next test
```

### Via voice command

```bash
python scripts/run_voice_control.py
```

Say: "Hey Spot, open the door"

---

## Fisheye Coordinate System

Spot's fisheye cameras are mounted sideways. For door calibration:

- **Display**: Images rotated 90 CW for upright viewing
- **API calls**: Coordinates must be un-rotated back to the original frame
- **Formula**: `orig_x = rot_y`, `orig_y = H_orig - 1 - rot_x` (use `raw_rows`, not `raw_cols`)

---

## Door Service (AutoWalk Integration)

For AutoWalk missions, the door service runs as a Remote Mission Service:

```
src/door_service/door_mission_service.py
```

This is a gRPC service that Spot calls at configured door waypoints during mission playback. See the POST_FLASH_MIGRATION guide for setup details.

---

## Supported Door Types

| Type | Method | Status |
|------|--------|--------|
| Push-bar doors | AutoPushCommand | Implemented, needs testing |
| Handle doors | Grasp + push/pull | Implemented in door_mission_service, not tested |
| Automatic doors | Button press | Implemented in door_mission_service, not tested |
