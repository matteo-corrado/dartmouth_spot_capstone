#!/usr/bin/env python3
"""Probe Spot's image and point-cloud services.

Read-only diagnostic. Lists every image source and point cloud source the
robot exposes, along with the camera-model intrinsics and depth_scale needed
to deproject pixels into 3D rays. Used to ground the perception overhaul in
empirical reality rather than docs/agent guesses.

Does NOT take a lease, does NOT power on motors.

Usage:
  python scripts/probe_image_sources.py
  python scripts/probe_image_sources.py --capture-depth   # also fetch one
                                                            # depth frame per
                                                            # depth source and
                                                            # report stats
"""
import argparse
import sys

import numpy as np

# Make `import src...` work when running from the repo root
sys.path.insert(0, ".")

from bosdyn.api import image_pb2
from bosdyn.api import local_grid_pb2
from bosdyn.api import world_object_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.local_grid import LocalGridClient
from bosdyn.client.point_cloud import PointCloudClient
from bosdyn.client.ray_cast import RayCastClient
from bosdyn.client.world_object import WorldObjectClient

from src.session import quick_robot


def fmt_intrinsics(src):
    """Return a one-line summary of which camera_model is populated and key intrinsics."""
    model = src.WhichOneof("camera_models")
    if model is None:
        return "model=NONE (no intrinsics)"
    if model == "pinhole":
        i = src.pinhole.intrinsics
        return (f"model=PINHOLE  fx={i.focal_length.x:.1f} fy={i.focal_length.y:.1f} "
                f"cx={i.principal_point.x:.1f} cy={i.principal_point.y:.1f} "
                f"skew=({i.skew.x:.3f},{i.skew.y:.3f})")
    if model == "pinhole_brown_conrady":
        i = src.pinhole_brown_conrady.intrinsics
        d = src.pinhole_brown_conrady
        return (f"model=PINHOLE_BROWN_CONRADY  fx={i.focal_length.x:.1f} fy={i.focal_length.y:.1f} "
                f"cx={i.principal_point.x:.1f} cy={i.principal_point.y:.1f} "
                f"k1={d.k1:.4f} k2={d.k2:.4f} k3={d.k3:.4f} p1={d.p1:.4f} p2={d.p2:.4f}")
    if model == "kannala_brandt":
        i = src.kannala_brandt.intrinsics
        d = src.kannala_brandt
        return (f"model=KANNALA_BRANDT  fx={i.focal_length.x:.1f} fy={i.focal_length.y:.1f} "
                f"cx={i.principal_point.x:.1f} cy={i.principal_point.y:.1f} "
                f"k1={d.k1:.4f} k2={d.k2:.4f} k3={d.k3:.4f} k4={d.k4:.4f}")
    return f"model={model} (unrecognised)"


def probe_images(image_client, capture_depth=False):
    print("=" * 80)
    print("IMAGE SOURCES")
    print("=" * 80)
    sources = image_client.list_image_sources()
    print(f"{len(sources)} sources advertised\n")

    depth_sources = []
    for s in sorted(sources, key=lambda x: x.name):
        type_name = image_pb2.ImageSource.ImageType.Name(s.image_type)
        print(f"  {s.name}")
        print(f"    type={type_name}  {s.cols}x{s.rows}  depth_scale={s.depth_scale}")
        print(f"    {fmt_intrinsics(s)}")
        formats_attr = getattr(s, "image_formats", None)
        if formats_attr:
            fmts = [image_pb2.Image.Format.Name(f) for f in formats_attr]
            print(f"    image_formats={fmts}")
        pix_attr = getattr(s, "pixel_formats", None)
        if pix_attr:
            pix = [image_pb2.Image.PixelFormat.Name(p) for p in pix_attr]
            print(f"    pixel_formats={pix}")
        print()
        if s.image_type == image_pb2.ImageSource.IMAGE_TYPE_DEPTH:
            depth_sources.append(s.name)

    print(f"Depth sources: {depth_sources}\n")

    if capture_depth and depth_sources:
        print("=" * 80)
        print("DEPTH SAMPLES (one frame per depth source)")
        print("=" * 80)
        for name in depth_sources:
            try:
                req = build_image_request(
                    name,
                    pixel_format=image_pb2.Image.PIXEL_FORMAT_DEPTH_U16,
                    image_format=image_pb2.Image.FORMAT_RAW,
                )
                resp = image_client.get_image([req])[0]
                rows = resp.shot.image.rows
                cols = resp.shot.image.cols
                pf = image_pb2.Image.PixelFormat.Name(resp.shot.image.pixel_format)
                fmt = image_pb2.Image.Format.Name(resp.shot.image.format)
                ds = resp.source.depth_scale
                arr = np.frombuffer(resp.shot.image.data, dtype=np.uint16).reshape(rows, cols)
                valid = arr[(arr > 0) & (arr < 65535)]
                if valid.size:
                    raw_min, raw_med, raw_max = int(valid.min()), int(np.median(valid)), int(valid.max())
                    if ds and ds > 0:
                        m_min, m_med, m_max = raw_min / ds, raw_med / ds, raw_max / ds
                        meters = f"  min={m_min:.2f}m  med={m_med:.2f}m  max={m_max:.2f}m"
                    else:
                        meters = "  (depth_scale=0, cannot convert)"
                else:
                    raw_min = raw_med = raw_max = -1
                    meters = "  (no valid pixels)"
                print(f"  {name}: {cols}x{rows} pf={pf} fmt={fmt} depth_scale={ds}")
                print(f"    raw_uint16: min={raw_min} med={raw_med} max={raw_max}{meters}")
                print(f"    valid_frac={valid.size / arr.size:.2%}  total={arr.size}")
                print(f"    sensor_frame='{resp.shot.frame_name_image_sensor}'")
                print()
            except Exception as e:
                print(f"  {name}: FAILED to fetch ({type(e).__name__}: {e})\n")


def probe_point_clouds(robot):
    print("=" * 80)
    print("POINT CLOUD SOURCES")
    print("=" * 80)
    try:
        pc_client = robot.ensure_client(PointCloudClient.default_service_name)
    except Exception as e:
        print(f"  point-cloud service not registered on this robot ({type(e).__name__}: {e})")
        print("  (this is normal — point-cloud service usually only exists with a Velodyne / EAP payload)")
        return
    try:
        sources = pc_client.list_point_cloud_sources()
        if not sources:
            print("  service is registered but advertises 0 sources")
            return
        for s in sources:
            print(f"  {s.name}  frame_name_sensor='{s.frame_name_sensor}'")
    except Exception as e:
        print(f"  list_point_cloud_sources() failed ({type(e).__name__}: {e})")


def _decode_grid_values(grid):
    """Decode a LocalGrid's data bytes into a numpy array of physical values.

    Handles ENCODING_RAW; for ENCODING_RLE we just compute summary stats from
    the run-length-encoded form without expanding (cheaper for diagnostics).

    Returns (physical_array_or_none, raw_min, raw_med, raw_max, valid_count, total_count).
    """
    cell_format = grid.cell_format
    dtype_map = {
        local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT32: np.float32,
        local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT64: np.float64,
        local_grid_pb2.LocalGrid.CELL_FORMAT_INT8: np.int8,
        local_grid_pb2.LocalGrid.CELL_FORMAT_UINT8: np.uint8,
        local_grid_pb2.LocalGrid.CELL_FORMAT_INT16: np.int16,
        local_grid_pb2.LocalGrid.CELL_FORMAT_UINT16: np.uint16,
    }
    dtype = dtype_map.get(cell_format)
    if dtype is None:
        return None, None, None, None, 0, 0

    if grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RAW:
        try:
            arr = np.frombuffer(grid.data, dtype=dtype)
        except Exception:
            return None, None, None, None, 0, 0
        if arr.size == 0:
            return None, None, None, None, 0, 0
        scale = grid.cell_value_scale if grid.cell_value_scale != 0 else 1.0
        offset = grid.cell_value_offset
        physical = arr.astype(np.float64) * scale + offset
        return (
            physical,
            float(arr.min()),
            float(np.median(arr)),
            float(arr.max()),
            int(arr.size),
            int(arr.size),
        )
    elif grid.encoding == local_grid_pb2.LocalGrid.ENCODING_RLE:
        # data + rle_counts pair: each (count, value) run
        try:
            values = np.frombuffer(grid.data, dtype=dtype)
        except Exception:
            return None, None, None, None, 0, 0
        counts = np.array(grid.rle_counts, dtype=np.int64)
        if values.size == 0 or counts.size == 0:
            return None, None, None, None, 0, 0
        total = int(counts.sum())
        return (
            None,  # don't expand for diagnostics
            float(values.min()),
            float(np.median(values)),  # median of unique values, not weighted
            float(values.max()),
            int(values.size),
            total,
        )
    return None, None, None, None, 0, 0


def probe_local_grids(robot):
    print("=" * 80)
    print("LOCAL GRIDS")
    print("=" * 80)
    try:
        lg_client = robot.ensure_client(LocalGridClient.default_service_name)
    except Exception as e:
        print(f"  local-grid-service not registered on this robot ({type(e).__name__}: {e})")
        return
    try:
        type_descriptors = lg_client.get_local_grid_types()
    except Exception as e:
        print(f"  get_local_grid_types() failed ({type(e).__name__}: {e})")
        return

    names = [t.name for t in type_descriptors]
    print(f"  {len(names)} grid types advertised: {names}\n")
    if not names:
        return

    try:
        responses = lg_client.get_local_grids(names)
    except Exception as e:
        print(f"  get_local_grids({names}) failed ({type(e).__name__}: {e})")
        return

    for r in responses:
        status_name = local_grid_pb2.LocalGridResponse.Status.Name(r.status)
        print(f"  {r.local_grid_type_name}  status={status_name}")
        if r.status != local_grid_pb2.LocalGridResponse.STATUS_OK:
            print()
            continue
        g = r.local_grid
        ext = g.extent
        cell_fmt = local_grid_pb2.LocalGrid.CellFormat.Name(g.cell_format)
        encoding = local_grid_pb2.LocalGrid.Encoding.Name(g.encoding)
        print(f"    extent: cell_size={ext.cell_size}m  {ext.num_cells_x}x{ext.num_cells_y}"
              f"  ({ext.num_cells_x * ext.cell_size:.2f}m x {ext.num_cells_y * ext.cell_size:.2f}m)")
        print(f"    cell_format={cell_fmt}  encoding={encoding}")
        print(f"    cell_value_scale={g.cell_value_scale}  cell_value_offset={g.cell_value_offset}")
        print(f"    frame_name_local_grid_data='{g.frame_name_local_grid_data}'")
        print(f"    data_bytes={len(g.data)}  unknown_cells_bytes={len(g.unknown_cells)}"
              f"  rle_counts={len(g.rle_counts)}")

        physical, rmin, rmed, rmax, vcount, tcount = _decode_grid_values(g)
        if rmin is not None:
            print(f"    raw values: min={rmin}  med={rmed}  max={rmax}  cells={tcount}")
            if physical is not None:
                pmin, pmed, pmax = float(physical.min()), float(np.median(physical)), float(physical.max())
                print(f"    physical:   min={pmin:.4f}  med={pmed:.4f}  max={pmax:.4f}")
        else:
            print(f"    (could not decode cell values)")

        # Frame chain check: are the body / vision frames reachable from the grid frame?
        try:
            from bosdyn.client import frame_helpers
            snap = g.transforms_snapshot
            frames = list(snap.child_to_parent_edge_map.keys()) if snap else []
            print(f"    transforms_snapshot frames ({len(frames)}): "
                  f"{frames[:6]}{'...' if len(frames) > 6 else ''}")
            try:
                t = frame_helpers.get_a_tform_b(snap, frame_helpers.BODY_FRAME_NAME, g.frame_name_local_grid_data)
                if t is None:
                    print(f"    body_tform_grid: NONE (no path in tree)")
                else:
                    print(f"    body_tform_grid: pos=({t.x:.2f},{t.y:.2f},{t.z:.2f})")
            except Exception as e:
                print(f"    body_tform_grid: error ({type(e).__name__}: {e})")
        except Exception as e:
            print(f"    (frame check failed: {e})")
        print()


def probe_raycast(robot):
    print("=" * 80)
    print("RAYCAST")
    print("=" * 80)
    try:
        rc_client = robot.ensure_client(RayCastClient.default_service_name)
    except Exception as e:
        print(f"  ray-cast service not registered on this robot ({type(e).__name__}: {e})")
        return

    from bosdyn.api import ray_cast_pb2
    from bosdyn.client import frame_helpers

    # Send a single ray straight forward in body frame, requesting all backing
    # source types. The response tells us which sources are populated on this
    # firmware/payload combo (TYPE_VOXEL_MAP is the load-bearing question).
    # Note: RayCastClient.raycast() takes (x, y, z) tuples — its internal
    # _raycast_request indexes them with [0]/[1]/[2], so passing Vec3 protos
    # directly raises TypeError ("Vec3 object is not subscriptable").
    ray_origin = (0.0, 0.0, 0.0)
    ray_direction = (1.0, 0.0, 0.0)
    raycast_types = [
        ray_cast_pb2.RayIntersection.TYPE_GROUND_PLANE,
        ray_cast_pb2.RayIntersection.TYPE_TERRAIN_MAP,
        ray_cast_pb2.RayIntersection.TYPE_VOXEL_MAP,
        ray_cast_pb2.RayIntersection.TYPE_HAND_DEPTH,
    ]
    print("  Sending forward ray in body frame, requesting all source types...")
    try:
        resp = rc_client.raycast(
            ray_origin,
            ray_direction,
            raycast_types,
            min_distance=0.0,
            frame_name=frame_helpers.BODY_FRAME_NAME,
        )
    except Exception as e:
        print(f"  raycast() failed ({type(e).__name__}: {e})")
        return

    hits = list(resp.hits) if hasattr(resp, "hits") else []
    print(f"  {len(hits)} hit(s):")
    for h in hits:
        type_name = ray_cast_pb2.RayIntersection.Type.Name(h.type)
        pt = h.hit_position_in_hit_frame
        print(f"    type={type_name}  distance={h.distance_meters:.3f}m  "
              f"hit=({pt.x:.3f},{pt.y:.3f},{pt.z:.3f})")
    if not hits:
        print("    (no intersections — empty world model OR ray passed through nothing in any source)")
    print()


def probe_world_objects(robot):
    print("=" * 80)
    print("WORLD OBJECTS")
    print("=" * 80)
    try:
        wo_client = robot.ensure_client(WorldObjectClient.default_service_name)
    except Exception as e:
        print(f"  world-object service not registered on this robot ({type(e).__name__}: {e})")
        return

    print("  Listing all world objects (no type filter)...")
    try:
        resp = wo_client.list_world_objects()
        objects = list(resp.world_objects)
        print(f"  {len(objects)} object(s) found")
        for obj in objects:
            populated_props = []
            for f in obj.DESCRIPTOR.fields:
                if f.name.endswith("_properties") and obj.HasField(f.name):
                    populated_props.append(f.name)
            print(f"    id={obj.id}  name='{obj.name}'  props={populated_props}")
    except Exception as e:
        print(f"  list_world_objects() failed ({type(e).__name__}: {e})")

    # Specifically check for tracked entities (EAP 2 / Moving Object Detection feature).
    print("\n  Trying TRACKED_ENTITY-specific filter (requires EAP 2 — expected empty here)...")
    try:
        resp = wo_client.list_world_objects(
            object_type=[world_object_pb2.WORLD_OBJECT_TRACKED_ENTITY]
        )
        objects = list(resp.world_objects)
        print(f"  {len(objects)} tracked-entity object(s)")
        for obj in objects:
            print(f"    id={obj.id}  name='{obj.name}'")
    except Exception as e:
        print(f"  filter failed ({type(e).__name__}: {e})")
    print()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--capture-depth", action="store_true",
                   help="Fetch one frame per depth source and report uint16 stats + meters")
    p.add_argument("--skip-images", action="store_true",
                   help="Skip the image-source probe (useful when iterating on local_grid / raycast)")
    args = p.parse_args()

    print("Connecting to Spot (no lease, no power-on)...")
    robot = quick_robot("probe-image-sources")
    print(f"Connected: {robot._name}\n")

    if not args.skip_images:
        image_client = robot.ensure_client(ImageClient.default_service_name)
        probe_images(image_client, capture_depth=args.capture_depth)
        probe_point_clouds(robot)
    probe_local_grids(robot)
    probe_raycast(robot)
    probe_world_objects(robot)


if __name__ == "__main__":
    main()
