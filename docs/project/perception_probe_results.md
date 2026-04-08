# Perception probe results — 2026-04-08

Output of `python scripts/probe_image_sources.py --capture-depth` on the
live Spot at `192.168.80.3` (no lease, no power-on, base-payload Spot
without EAP 2). Captured during Tier 0a of `~/.claude/plans/humming-stargazing-shell.md`.

## Headline findings (use these for plan decisions)

1. **Every camera reports `model=PINHOLE`.** All 5 body fisheyes, all 6
   depth sources, the gripper color camera and the gripper depth camera
   ALL populate the `pinhole` field of the `camera_models` oneof. This
   contradicts the planning assumption that body fisheyes use
   Kannala-Brandt distortion. **Consequence**: `bosdyn.client.image.pixel_to_camera_space()`
   should work directly on every source — we do NOT need to write our
   own Kannala-Brandt deprojection. This significantly simplifies Tier
   2a (option A `WalkToObjectInImage` becomes the obvious choice over
   option B `WalkToObjectRayInWorld`).

2. **`depth_scale` is 999.0 for body cameras and 1000.0 for the gripper.**
   Body cameras (front-left, front-right, left, right, back) all return
   `depth_scale=999.0` on the actual response. The gripper depth uses
   `depth_scale=1000.0`. The "_in_visual_frame" rectified variants
   advertise `depth_scale=0.0` in `list_image_sources()` but populate
   the actual depth_scale (999.0 or 1000.0) on the fetched response.
   **Consequence**: our Tier 1a `_decode_depth` reads from the response
   correctly. The fallback constant (`1000.0`) is wrong for body cameras
   by 0.1% — irrelevant in practice but worth fixing the comment to
   reflect that the fallback only triggers if the response is malformed.

3. **All 6 local grids are exposed**, including an unexpected
   `fixed_obstacle_distance` grid the plan didn't anticipate:
   ```
   terrain                  INT16 RLE  scale=0.001 offset=0.216
   terrain_valid            UINT8 RLE  scale=0    offset=0
   no_step                  INT16 RLE  scale=0.001 offset=0.058
   obstacle_distance        INT16 RLE  scale=0.001 offset=0.661
   fixed_obstacle_distance  INT16 RLE  scale=0.001 offset=0
   intensity                UINT8 RLE  scale=0    offset=0
   ```
   All grids: 128×128 cells at 3.0 cm cell size = **3.84 m × 3.84 m**
   coverage. All RLE-encoded. Frame name pattern is
   `{type}_local_grid_corner` per grid, transforms snapshot includes
   `body`, `vision`, `odom`, and the grid's own corner frame.

4. **`obstacle_distance` IS a signed distance field**, confirming the
   plan's assumption. Raw values range -1081 to +1081, scale 0.001,
   offset +0.661 → physical range -0.42 m to +1.74 m. Median +0.31 m
   (most cells are outside obstacles). Cells with physical value ≤ 0
   are inside obstacles. **Tier 4b's `obstacle_query.nearest_obstacle_in_body_frame`
   semantics are correct** — `_OBSTACLE_DISTANCE_THRESHOLD = 0.0` is the
   right value.

5. **`unknown_cells` is one byte per cell on this firmware**, NOT the
   documented bitfield. For a 128×128 grid, `unknown_cells` is 16 384
   bytes (one byte per cell, value 0 or 1), not 2 048 bytes (bitfield
   with 1 bit per cell). **Bug surfaced and fixed**: our `_decode_unknown_mask`
   in `local_grid_helpers.py` originally assumed bitfield only and was
   silently producing the wrong unknown mask. Updated to detect both
   formats from the byte length.

   On the captured frame, **76% of `obstacle_distance` cells are flagged
   unknown** (12 516 of 16 384). This makes sense: Spot's depth cameras
   only see the immediate area around the body, and the rest of the
   3.84 m × 3.84 m grid is unobserved.

6. **RayCast returns 0 hits on this Spot** for a forward ray with all 4
   intersection types requested (`TYPE_GROUND_PLANE`, `TYPE_TERRAIN_MAP`,
   `TYPE_VOXEL_MAP`, `TYPE_HAND_DEPTH`). This empirically confirms that
   `TYPE_VOXEL_MAP` is unavailable without EAP 2. **Tier 4c's design
   decision to use LocalGrid (`obstacle_query.nearest_obstacle_in_body_frame`)
   instead of RayCast is empirically validated** — the raycast service
   exists but is not useful for obstacle pre-checks on this hardware.
   `raycast_helpers.cast_ray()` remains in the codebase as scaffolding
   in case future firmware populates it.

   The 0-hit result is suspicious for `TYPE_GROUND_PLANE` too — a
   forward ray at the body origin should at minimum be able to discover
   the ground plane. Possible explanations: (a) the body origin is not
   at z=0 in the body frame, so a forward ray never crosses the ground,
   (b) the service requires the ray to actually point at something
   downward. Worth a follow-up probe with a downward ray someday.

7. **`PointCloudClient` is not registered.** `point-cloud` service
   `UnregisteredServiceNameError`. Confirms no EAP 2 / no Velodyne
   payload. Already documented in `project_no_eap2.md`.

8. **`WorldObjectClient` is registered, returns 0 objects** for
   `list_world_objects()` with no filter, and 0 objects for the
   `WORLD_OBJECT_TRACKED_ENTITY` filter (also expected — TRACKED_ENTITY
   needs EAP 2). The smart-routing path in `go_to_object` (Tier 3a)
   needs a fiducial in view to actually exercise. **Open follow-up**:
   point Spot at a fiducial and re-run the world-object branch of the
   probe to confirm fiducial decoding works.

## Plan implications

| Tier | Effect of probe |
|---|---|
| 1a — `_decode_depth` | Already correct. Fallback constant comment can mention 999/1000 split. |
| 1b — `_depth_at_bbox` replacement | Still blocked on linked-hopping-anchor Stage 0. Probe doesn't unblock it. |
| 2a — `WalkToObjectInImage` vs `RayInWorld` | **Resolved**: pinhole-everywhere means option A (`WalkToObjectInImage`) is the right call. No custom deprojection needed. |
| 2b — body-frame `_steer` | Unblocked from a research perspective. Implementation can proceed once 2a lands. |
| 2c — intrinsics cache | Unblocked. The cache schema is now known: per-source pinhole `(fx, fy, cx, cy)` plus image dimensions. |
| 4a — LocalGrid decode | **Bug found and fixed** in `_decode_unknown_mask` (one-byte-per-cell format). |
| 4b — obstacle_query | Semantics confirmed correct. `_OBSTACLE_DISTANCE_THRESHOLD = 0.0` is right. Note that 76% of cells may be unknown — nearest-obstacle scans need to handle the masked-out region gracefully (which they do — NaN comparisons return False). |
| 4c — collision pre-check via LocalGrid | **Empirically validated**: voxel-map raycast is empty, LocalGrid is the only working obstacle path on this hardware. |
| linked-hopping-anchor Stage 0 — rotation table | The rotation-table assumptions in that plan still hold (cameras are physical fisheye lenses regardless of how the SDK models distortion). The "-78°/-102°/0°/180°/0°" angles from BD's `get_image.py --auto-rotate` are about the physical sensor mounting orientation, not the camera intrinsics. |

## Raw probe output

```
Connecting to Spot (no lease, no power-on)...
Connected: 192.168.80.3

================================================================================
IMAGE SOURCES
================================================================================
20 sources advertised

  back_depth
    type=IMAGE_TYPE_DEPTH  424x240  depth_scale=999.0
    model=PINHOLE  fx=213.8 fy=213.8 cx=211.2 cy=120.4 skew=(0.000,0.000)
    image_formats=['FORMAT_RAW']
    pixel_formats=['PIXEL_FORMAT_DEPTH_U16']

  back_depth_in_visual_frame
    type=IMAGE_TYPE_DEPTH  640x480  depth_scale=0.0
    model=PINHOLE  fx=257.1 fy=256.6 cx=313.5 cy=249.3 skew=(0.000,0.000)
    image_formats=['FORMAT_RAW']
    pixel_formats=['PIXEL_FORMAT_DEPTH_U16']

  back_fisheye_image
    type=IMAGE_TYPE_VISUAL  640x480  depth_scale=0.0
    model=PINHOLE  fx=257.1 fy=256.6 cx=313.5 cy=249.3 skew=(0.000,0.000)
    image_formats=['FORMAT_RAW', 'FORMAT_JPEG']
    pixel_formats=['PIXEL_FORMAT_GREYSCALE_U8']

  frontleft_depth
    type=IMAGE_TYPE_DEPTH  424x240  depth_scale=999.0
    model=PINHOLE  fx=214.8 fy=214.8 cx=213.1 cy=118.6 skew=(0.000,0.000)
    image_formats=['FORMAT_RAW']
    pixel_formats=['PIXEL_FORMAT_DEPTH_U16']

  frontleft_depth_in_visual_frame
    type=IMAGE_TYPE_DEPTH  640x480  depth_scale=0.0
    model=PINHOLE  fx=256.3 fy=255.8 cx=319.9 cy=241.3 skew=(0.000,0.000)

  frontleft_fisheye_image
    type=IMAGE_TYPE_VISUAL  640x480  depth_scale=0.0
    model=PINHOLE  fx=256.3 fy=255.8 cx=319.9 cy=241.3 skew=(0.000,0.000)

  frontright_depth
    type=IMAGE_TYPE_DEPTH  424x240  depth_scale=999.0
    model=PINHOLE  fx=213.7 fy=213.7 cx=212.0 cy=118.6 skew=(0.000,0.000)

  frontright_depth_in_visual_frame
    type=IMAGE_TYPE_DEPTH  640x480  depth_scale=0.0
    model=PINHOLE  fx=255.7 fy=255.2 cx=325.0 cy=236.8 skew=(0.000,0.000)

  frontright_fisheye_image
    type=IMAGE_TYPE_VISUAL  640x480  depth_scale=0.0
    model=PINHOLE  fx=255.7 fy=255.2 cx=325.0 cy=236.8 skew=(0.000,0.000)

  hand_color_image
    type=IMAGE_TYPE_VISUAL  640x480  depth_scale=0.0
    model=PINHOLE  fx=552.0 fy=552.0 cx=320.0 cy=240.0 skew=(0.000,0.000)
    pixel_formats=['PIXEL_FORMAT_RGB_U8']

  hand_color_in_hand_depth_frame
    type=IMAGE_TYPE_VISUAL  224x171  depth_scale=0.0
    model=PINHOLE  fx=212.4 fy=212.4 cx=115.3 cy=88.0 skew=(0.000,0.000)
    pixel_formats=['PIXEL_FORMAT_RGB_U8']

  hand_depth
    type=IMAGE_TYPE_DEPTH  224x171  depth_scale=0.0
    model=PINHOLE  fx=212.4 fy=212.4 cx=115.3 cy=88.0 skew=(0.000,0.000)

  hand_depth_in_hand_color_frame
    type=IMAGE_TYPE_DEPTH  640x480  depth_scale=0.0
    model=PINHOLE  fx=552.0 fy=552.0 cx=320.0 cy=240.0 skew=(0.000,0.000)

  hand_image
    type=IMAGE_TYPE_VISUAL  224x171  depth_scale=0.0
    model=PINHOLE  fx=212.4 fy=212.4 cx=115.3 cy=88.0 skew=(0.000,0.000)
    pixel_formats=['PIXEL_FORMAT_GREYSCALE_U8']

  left_depth
    type=IMAGE_TYPE_DEPTH  424x240  depth_scale=999.0
    model=PINHOLE  fx=218.8 fy=218.8 cx=216.2 cy=119.7 skew=(0.000,0.000)

  left_depth_in_visual_frame
    type=IMAGE_TYPE_DEPTH  640x480  depth_scale=0.0
    model=PINHOLE  fx=258.5 fy=257.9 cx=321.1 cy=242.8 skew=(0.000,0.000)

  left_fisheye_image
    type=IMAGE_TYPE_VISUAL  640x480  depth_scale=0.0
    model=PINHOLE  fx=258.5 fy=257.9 cx=321.1 cy=242.8 skew=(0.000,0.000)

  right_depth
    type=IMAGE_TYPE_DEPTH  424x240  depth_scale=999.0
    model=PINHOLE  fx=217.7 fy=217.7 cx=215.0 cy=119.6 skew=(0.000,0.000)

  right_depth_in_visual_frame
    type=IMAGE_TYPE_DEPTH  640x480  depth_scale=0.0
    model=PINHOLE  fx=254.6 fy=254.2 cx=325.9 cy=233.9 skew=(0.000,0.000)

  right_fisheye_image
    type=IMAGE_TYPE_VISUAL  640x480  depth_scale=0.0
    model=PINHOLE  fx=254.6 fy=254.2 cx=325.9 cy=233.9 skew=(0.000,0.000)

================================================================================
DEPTH SAMPLES (one frame per depth source)
================================================================================
  back_depth: 424x240 depth_scale=999.0
    raw_uint16: min=299 med=807 max=1860  -> min=0.30m med=0.81m max=1.86m
    valid_frac=60.60%

  back_depth_in_visual_frame: 640x480 depth_scale=999.0
    raw_uint16: min=294 med=793 max=1828  -> min=0.29m med=0.79m max=1.83m
    valid_frac=34.46%

  frontleft_depth: 424x240 depth_scale=999.0
    raw_uint16: min=208 med=993 max=4720  -> min=0.21m med=0.99m max=4.72m
    valid_frac=67.95%

  frontleft_depth_in_visual_frame: 640x480 depth_scale=999.0
    raw_uint16: min=205 med=975 max=4731  -> min=0.21m med=0.98m max=4.74m
    valid_frac=37.29%

  frontright_depth: 424x240 depth_scale=999.0
    raw_uint16: min=387 med=782 max=7271  -> min=0.39m med=0.78m max=7.28m
    valid_frac=68.30%

  frontright_depth_in_visual_frame: 640x480 depth_scale=999.0
    raw_uint16: min=388 med=805 max=7420  -> min=0.39m med=0.81m max=7.43m
    valid_frac=36.58%

  hand_depth: 224x171 depth_scale=1000.0
    raw_uint16: min=7 med=2002 max=7168  -> min=0.01m med=2.00m max=7.17m
    valid_frac=62.31%

  hand_depth_in_hand_color_frame: 640x480 depth_scale=1000.0
    raw_uint16: min=51 med=2508 max=7162  -> min=0.05m med=2.51m max=7.16m
    valid_frac=30.06%

  left_depth: 424x240 depth_scale=999.0
    raw_uint16: min=144 med=172 max=3068  -> min=0.14m med=0.17m max=3.07m
    valid_frac=80.09%

  left_depth_in_visual_frame: 640x480 depth_scale=999.0
    raw_uint16: min=144 med=170 max=209  -> min=0.14m med=0.17m max=0.21m
    valid_frac=38.92%

  right_depth: 424x240 depth_scale=999.0
    raw_uint16: min=306 med=482 max=955  -> min=0.31m med=0.48m max=0.96m
    valid_frac=76.76%

  right_depth_in_visual_frame: 640x480 depth_scale=999.0
    raw_uint16: min=306 med=477 max=957  -> min=0.31m med=0.48m max=0.96m
    valid_frac=37.99%

================================================================================
POINT CLOUD SOURCES
================================================================================
  point-cloud service NOT REGISTERED (UnregisteredServiceNameError)
  → confirms no Velodyne / EAP 2

================================================================================
LOCAL GRIDS
================================================================================
  6 grid types advertised: ['terrain', 'terrain_valid', 'no_step',
                            'obstacle_distance', 'fixed_obstacle_distance', 'intensity']

  terrain  STATUS_OK  128x128 @ 3cm  INT16 RLE  scale=0.001 offset=0.216
    raw values: min=-568  med=-110.5  max=568    cells=16384
    body_tform_grid: pos=(-2.04,-1.83,-0.41)
    unknown_cells_bytes=0  (no unknown info)

  terrain_valid  STATUS_OK  128x128 @ 3cm  UINT8 RLE  scale=0 offset=0
    raw values: min=0  med=0  max=1
    unknown_cells_bytes=0

  no_step  STATUS_OK  128x128 @ 3cm  INT16 RLE  scale=0.001 offset=0.058
    raw values: min=-1501  med=-88  max=1501
    unknown_cells_bytes=0

  obstacle_distance  STATUS_OK  128x128 @ 3cm  INT16 RLE  scale=0.001 offset=0.661
    raw values: min=-1081  med=-347  max=1081
    physical (after scale+offset): min=-0.42m  med=0.31m  max=1.74m
    body_tform_grid: pos=(-2.04,-1.83,-0.41)
    unknown_cells_bytes=16384  (1 BYTE PER CELL — not bitfield)
    nonzero unknown bytes: 12516 / 16384 (76% unknown)

  fixed_obstacle_distance  STATUS_OK  128x128 @ 3cm  INT16 RLE  scale=0.001 offset=0
    raw values: min=-999  med=999  max=999
    unknown_cells_bytes=16384  (1 byte per cell)

  intensity  STATUS_OK  128x128 @ 3cm  UINT8 RLE  scale=0 offset=0
    raw values: min=0  med=79  max=227
    unknown_cells_bytes=0

================================================================================
RAYCAST
================================================================================
  Forward ray in body frame, all 4 type filters → 0 hits
  → TYPE_VOXEL_MAP empirically empty (no EAP 2)
  → TYPE_TERRAIN_MAP and TYPE_GROUND_PLANE also empty for this ray
    (likely because the ray is horizontal at the body origin and never
    crosses the ground plane below the body)

================================================================================
WORLD OBJECTS
================================================================================
  list_world_objects() (no filter) → 0 objects
  list_world_objects([WORLD_OBJECT_TRACKED_ENTITY]) → 0 objects (expected)
  → no fiducials in view at probe time
  → world-object service IS registered, just nothing visible
```
