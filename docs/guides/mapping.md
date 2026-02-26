# Mapping Guide

How to record, download, and manage GraphNav maps for Spot navigation.

## Overview

Spot uses Boston Dynamics' GraphNav for autonomous navigation. The workflow is:

1. Record a map on the tablet (walk Spot through the space)
2. Download the map to the Jetson
3. Upload the map to Spot's GraphNav memory
4. Localize the robot on the map
5. Use voice commands to navigate

---

## Step 1: Record a Map (Tablet)

1. Open **Spot Controller** app on the tablet
2. Tap **Menu** > **Autowalk** > **Record a Mission**
3. Drive Spot slowly through all areas you want mapped
4. At key locations, tap **"+"** to add a waypoint and give it a short name:
    - Use lowercase, no spaces: `kitchen`, `lab_door`, `lobby_entrance`
    - Avoid generic names: `waypoint_1`, `spot`, `here`
5. Return to the starting point to close the loop
6. Tap **Stop Recording** > **Save**

!!! tip "Tips for Better Maps"
    - Drive slowly for better visual data
    - Create loops (return to start) for alternative routes
    - Place AprilTag fiducials on walls for auto-localization
    - Record in good lighting
    - Mark stairs/ramps during recording

---

## Step 2: Download Map to Jetson

```bash
cd ~/dartmouth_spot_capstone
source spot-env/bin/activate

python scripts/download_map_from_spot.py --output maps/my_map
```

This downloads the graph, waypoint snapshots, and edge snapshots. It also auto-extracts waypoint names to `locations.json`.

---

## Step 3: Upload Map to GraphNav

```bash
python scripts/setup_map.py --map-path maps/my_map
```

This uploads the map to Spot's active GraphNav memory. The tablet recording saves to disk, but GraphNav memory is separate — this step is required.

---

## Step 4: Localize the Robot

The robot needs to know where it is on the map before navigating.

### Option A: Fiducial-based (automatic)

Place Spot where it can see an AprilTag from the map. Run:

```bash
python scripts/setup_map.py --map-path maps/my_map
```

The `setup_map.py` script attempts fiducial localization by default.

### Option B: Waypoint-based

Drive Spot to a known waypoint, then:

```bash
python scripts/setup_map.py --map-path maps/my_map --waypoint-init
```

### Option C: Tablet

1. In **Autowalk**, open your map
2. Drag the robot icon to Spot's current position
3. Rotate to match orientation
4. Tap **Set Location**

---

## Step 5: Navigate with Voice

```bash
python scripts/run_voice_control.py
```

Say: "Hey Spot, go to kitchen"

---

## Managing Locations

### List all waypoints

```bash
python scripts/map_waypoints.py --list
```

### Add a location via voice

While voice control is running, drive Spot to a new position and say:

> "Save this location as front desk"

This saves the current waypoint to `locations.json`.

### Edit locations manually

Edit `locations.json` in the project root:

```json
{
  "kitchen": "waypoint_abc123...",
  "lab": "waypoint_def456...",
  "new_location": "waypoint_ghi789..."
}
```

Find waypoint IDs with `python scripts/map_waypoints.py --list`.

---

## Map Directory Structure

```
maps/
  my_map/
    graph                  # Map topology
    waypoint_snapshots/    # Visual data at each waypoint
    edge_snapshots/        # Visual data along paths
```

Maps are gitignored (robot-specific). Each Jetson downloads its own map from Spot.

---

## Daily Usage (Map Already Exists)

```bash
# 1. Localize robot (tablet or script)
# 2. Start voice control
python scripts/run_voice_control.py
# 3. Give navigation commands
```

No need to re-upload the map unless it has changed.
