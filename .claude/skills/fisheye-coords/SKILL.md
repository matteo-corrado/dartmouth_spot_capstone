---
name: fisheye-coords
description: Reference for Spot's fisheye camera coordinate transforms. Use whenever code rotates a fisheye image for display, or maps a pixel coordinate from a rotated image back into the original camera frame for a Boston Dynamics SDK API call (image grasping, world object queries, depth probes).
---

# Spot fisheye coordinates

Spot's front fisheye cameras (`frontleft_fisheye_image`, `frontright_fisheye_image`) are mounted **sideways** on the robot. The pixel data the SDK returns is in the sensor's native orientation, which is rotated 90° from how a human reads it.

## The two transforms

**For display** (sensor → screen): rotate the image **90° clockwise**.

**For API calls** (screen pixel → sensor pixel): un-rotate so the BD SDK gets the coordinate in the orientation it expects.

```
orig_x = rot_y
orig_y = H_orig - 1 - rot_x
```

Where:
- `(rot_x, rot_y)` is the pixel coordinate the user clicked / YOLO produced **on the rotated (display) image**.
- `(orig_x, orig_y)` is the coordinate to send to BD APIs in the **sensor (original) frame**.
- `H_orig` is `image.shot.image.rows` from the BD `image_pb2` payload — that is, **`raw_rows`, NOT `raw_cols`**. Using `raw_cols` is the most common bug and produces an off-by-aspect-ratio target that looks plausible at first.

## Why it's easy to get wrong

After rotating 90° CW, the *displayed* image has dimensions `(W_disp, H_disp) = (H_orig, W_orig)`. It's tempting to write `H_orig - 1 - rot_x` as `W_disp - 1 - rot_x`, but if you compute `W_disp` from the rotated image's `.shape[1]` you are using the wrong number when feeding it back into a sensor-frame API. **Always derive from the original sensor dimensions** (`image_response.shot.image.rows`), not from the rotated array.

## Image format enums

The BD SDK uses small integer enums on `image.format`:

- `FORMAT_JPEG = 1`
- `FORMAT_RAW = 2`

Code that branches on format must use these constants (or the `image_pb2.Image.Format` enum), not magic strings.

## When to apply this skill

Apply this skill any time code:

1. Calls a BD API that takes a pixel coordinate — `ManipulationApiRequest`, `WalkToObjectInImage`, image-based grasping, `WorldObjectClient` queries that reference image coordinates.
2. Receives a click / YOLO bbox / AprilTag corner from a rotated display and forwards it into the SDK.
3. Rotates a fisheye image for any reason — confirm the rotation direction matches the assumed convention (90° CW for display).

## Self-test

Before relying on the transform in a new code path, run this sanity check:

```python
# Round-trip a known sensor-frame point through the rotation and back.
H_orig, W_orig = 480, 640         # whatever the actual fisheye reports
orig_x, orig_y = 100, 200          # a known point in the sensor frame
# Rotate 90 CW: (x, y) -> (H_orig - 1 - y, x)
rot_x = H_orig - 1 - orig_y
rot_y = orig_x
# Un-rotate using the documented formula
back_x = rot_y
back_y = H_orig - 1 - rot_x
assert (back_x, back_y) == (orig_x, orig_y), f"got {(back_x, back_y)}"
```

If a new code path can't pass this round-trip, it has the rotation direction or the `raw_rows` / `raw_cols` swap wrong.
