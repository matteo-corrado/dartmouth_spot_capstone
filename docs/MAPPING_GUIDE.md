# Streamlined Spot Mapping & Voice Control Guide

## Quick Start (5 Steps, 15 minutes)

### Prerequisites
- Spot is powered on and connected to WiFi
- Tablet is connected to Spot
- Jetson is connected to same network as Spot

---

## **Step 1: Record Map with Tablet (5 min)**

### Open Autowalk
1. Open **Spot Controller** app on tablet
2. Tap **☰ menu** → **"Autowalk"**
3. Tap **"Record a Mission"** or **"Record"**

### Record the Map
1. **Drive Spot around** using joystick
   - Go slowly and smoothly
   - Cover all areas you need
   - Make a complete loop back to start

2. **Add waypoints** at key locations:
   - Stop at each important spot
   - Tap **"+"** or screen to add waypoint
   - **Give it a short name**: `office`, `kitchen`, `desk`, `lobby`
   - Keep names **lowercase, no spaces** (use underscores if needed)

3. **Finish recording**:
   - Return to starting point
   - Tap **"Stop Recording"**
   - Name your map: `lab_map` or similar
   - Tap **"Save"**

✅ **Map is now saved on Spot**

---

## **Step 2: Download Map to Jetson (1 min)**

On your Jetson:

```bash
cd ~/dartmouth_spot_capstone
source spot-env/bin/activate

# Download map from Spot (auto-generates locations.json)
python scripts/download_map_from_spot.py --output maps/my_lab_map
```

This will:
- ✓ Download the map from Spot
- ✓ Extract waypoint names
- ✓ Auto-create `locations.json` with your named waypoints
- ✓ Save everything to `maps/my_lab_map/`

---

## **Step 3: Upload Map Back to Spot (1 min)**

```bash
# Upload the downloaded map to Spot's memory
python scripts/setup_map.py --map-path maps/my_lab_map
```

**Note:** This step is necessary because:
- GraphNav needs the map in its active memory
- The tablet recording saves to disk, but GraphNav memory is separate

---

## **Step 4: Localize Robot (30 sec)**

### Option A: Using Tablet (Easiest)
1. In **Autowalk**, open your map
2. **Manually drag the robot icon** to where Spot currently is
3. **Rotate the icon** to match Spot's orientation
4. Tap **"Set Location"** or **"Localize"**

### Option B: Using Fiducial (Automatic)
1. Place Spot where it can see a fiducial (AprilTag) from the map
2. It should auto-localize
3. Check tablet - robot icon should be solid/green

### Option C: Drive to Known Location (Simple)
1. Drive Spot to one of your named waypoints
2. In Autowalk, tap that waypoint on the map
3. Select **"Localize Here"**

✅ **Robot knows where it is on the map**

---

## **Step 5: Test Voice Control (1 min)**

```bash
cd ~/dartmouth_spot_capstone
source spot-env/bin/activate

# Start the integrated voice control (server + client)
python scripts/run_voice_control.py
```

**Or manually (2 terminals):**

```bash
# Terminal 1: Server
python src/voice_control/server.py

# Terminal 2: Client (adjust energy threshold as needed)
python src/voice_control/client_mic.py --device 24 --energy-mult 5.0
```

### Try Commands:
- **"stand"** - Robot stands up
- **"sit"** - Robot sits down
- **"turn left"** - Rotates 90° left
- **"turn right 45 degrees"** - Rotates 45° right
- **"go to office"** - Navigates to office waypoint
- **"go to kitchen"** - Navigates to kitchen waypoint

---

## **Troubleshooting**

### "Location not found"
- **Cause:** Waypoint name not in `locations.json`
- **Fix:** Check `locations.json` for exact spelling
- Or re-download map: `python scripts/download_map_from_spot.py`

### "No waypoints in map"
- **Cause:** Map not uploaded to GraphNav
- **Fix:** `python scripts/setup_map.py --map-path maps/YOUR_MAP`

### "Robot is stuck" / Navigation fails
- **Cause:** Robot not localized or path blocked
- **Fix:**
  1. Localize using tablet (drag robot icon)
  2. Clear obstacles
  3. Try command again

### Whisper transcribes wrong words
- **Cause:** Background noise or VAD too sensitive
- **Fix:** Use higher energy threshold
  ```bash
  python client_mic.py --device 24 --energy-mult 10.0  # Stricter
  ```

### "work" transcribed as "WEC"
- **Fix:** Added vocabulary hints in server.py (already done)
- Or say "work area" instead of just "work"

---

## **Quick Reference**

### One-Time Setup (when creating new map)
```bash
# 1. Record map on tablet (see Step 1)

# 2. Download map
python scripts/download_map_from_spot.py --output maps/my_map

# 3. Upload to GraphNav
python scripts/setup_map.py --map-path maps/my_map

# 4. Localize (use tablet - see Step 4)

# 5. Done! Use voice control anytime
```

### Daily Usage (map already exists)
```bash
# 1. Make sure E-Stop is released (tablet)

# 2. Localize robot (tablet - drag icon to current position)

# 3. Start voice control
python scripts/run_voice_control.py

# 4. Give voice commands!
```

---

## **Adding New Locations to Existing Map**

### Option 1: Re-record entire map
Follow Step 1 again with all locations

### Option 2: Add location via voice while navigating
```bash
# 1. Drive or navigate Spot to the new location
# 2. Say: "remember this location as front door"
# 3. It will save to locations.json automatically
```

### Option 3: Manual edit (if you know waypoint ID)
Edit `locations.json`:
```json
{
  "office": "waypoint_abc123",
  "kitchen": "waypoint_def456",
  "new_location": "waypoint_ghi789"  ← Add this
}
```

Find waypoint IDs:
```bash
python scripts/map_waypoints.py --list
```

---

## **Map Directory Structure**

After running the scripts, you'll have:

```
dartmouth_spot_capstone/
├── maps/
│   └── my_lab_map/              ← Your downloaded map
│       ├── graph                 ← Map structure
│       ├── waypoint_snapshot_*   ← Waypoint data
│       └── edge_snapshot_*       ← Path connections
├── locations.json               ← Auto-generated waypoint names
└── scripts/
    ├── download_map_from_spot.py  ← Downloads map from Spot
    └── setup_map.py               ← Uploads map to GraphNav
```

---

## **Tips for Better Maps**

1. **Name waypoints clearly** when recording:
   - Use: `kitchen`, `office`, `lobby_entrance`
   - Avoid: `Waypoint 1`, `spot`, `here`

2. **Drive slowly** when recording:
   - Gives better visual data
   - Creates cleaner paths

3. **Create loops**:
   - Always return to start
   - Creates alternative routes

4. **Add fiducials** (AprilTags):
   - Stick them on walls at key locations
   - Makes auto-localization work
   - Robot can find itself without manual help

5. **Record in good lighting**:
   - Cameras need light to see
   - Avoid very dark areas

6. **Mark uneven terrain**:
   - Stairs, ramps, rough terrain
   - Use tablet to mark these during recording

---

## **Advanced: Updating Locations Without Re-recording**

If you just need to rename waypoints without re-recording:

```bash
# 1. List all current waypoints
python scripts/map_waypoints.py --list

# 2. Copy a waypoint ID
# Example output: waypoint_abc123def456...

# 3. Edit locations.json manually:
{
  "new_name": "waypoint_abc123def456..."
}

# 4. Test with voice control:
# "go to new_name"
```

---

## **Summary: Minimal Steps**

**First time:**
1. Record map on tablet with named waypoints
2. `python scripts/download_map_from_spot.py`
3. `python scripts/setup_map.py --map-path maps/downloaded_map`
4. Localize on tablet
5. `python scripts/run_voice_control.py`

**Every time after:**
1. Localize on tablet (drag robot icon)
2. `python scripts/run_voice_control.py`
3. Give voice commands!

---

That's it! The workflow is now streamlined.
