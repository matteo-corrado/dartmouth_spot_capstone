#!/usr/bin/env python3
"""
Download the current map from Spot and save it locally.
Automatically extracts waypoint names to locations.json.
"""
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.session import spot_session
from src.map_loader import write_last_used
from src.location_manager import save_location
from bosdyn.client.graph_nav import GraphNavClient


def download_map(output_dir="maps/downloaded_map"):
    """Download map from Spot and extract waypoint names."""

    print("=" * 60)
    print("DOWNLOAD MAP FROM SPOT")
    print("=" * 60)
    print()

    # Create output directory structure
    output_path = Path(project_root) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    waypoint_snapshots_dir = output_path / "waypoint_snapshots"
    waypoint_snapshots_dir.mkdir(exist_ok=True)
    edge_snapshots_dir = output_path / "edge_snapshots"
    edge_snapshots_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_path}")
    print()

    # Connect to robot (no need to stand/sit for map download)
    print("[1/3] Connecting to Spot...")
    waypoint_names = {}
    failed_waypoints = []
    failed_edges = []

    with spot_session(stand_on_enter=False, sit_on_exit=False) as session:
        robot = session['robot']
        graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
        print("✓ Connected")
        print()

        # Download graph
        print("[2/3] Downloading map from Spot...")
        graph = graph_nav_client.download_graph()

        if not graph.waypoints:
            print("✗ No map found on robot. Record a map first using the tablet.")
            return

        print(f"✓ Downloaded graph with {len(graph.waypoints)} waypoints and {len(graph.edges)} edges")
        print()

        # Save graph
        graph_file = output_path / "graph"
        with open(graph_file, 'wb') as f:
            f.write(graph.SerializeToString())
        print(f"✓ Saved graph to: {graph_file}")

        # Download waypoint snapshots
        print()
        print(f"[3/3] Downloading waypoint snapshots...")

        for i, waypoint in enumerate(graph.waypoints):
            waypoint_id = waypoint.id
            snapshot_id = waypoint.snapshot_id

            try:
                # Get snapshot using snapshot_id (not waypoint_id)
                snapshot = graph_nav_client.download_waypoint_snapshot(snapshot_id)

                # Save snapshot to waypoint_snapshots subdirectory
                # Use "snapshot_*" naming (not "waypoint_snapshot_*") to match upload script expectations
                snapshot_file = waypoint_snapshots_dir / f"snapshot_{waypoint_id}"
                with open(snapshot_file, 'wb') as f:
                    f.write(snapshot.SerializeToString())

                # Extract waypoint name if available
                # Waypoint annotations contain user-friendly names
                waypoint_name = None
                if waypoint.annotations:
                    # Try to get name from annotations
                    if waypoint.annotations.name:
                        waypoint_name = waypoint.annotations.name.lower().replace(' ', '_')

                if waypoint_name:
                    waypoint_names[waypoint_name] = waypoint_id
                    print(f"  [{i+1}/{len(graph.waypoints)}] {waypoint_name} → {waypoint_id[:8]}...")
                else:
                    print(f"  [{i+1}/{len(graph.waypoints)}] waypoint_{i} → {waypoint_id[:8]}... (no name)")

            except Exception as e:
                failed_waypoints.append((i, waypoint_id[:8], str(e)))
                print(f"  [{i+1}/{len(graph.waypoints)}] ✗ FAILED: {waypoint_id[:8]}... ({e.__class__.__name__})")
                continue

        # Download edge snapshots
        for i, edge in enumerate(graph.edges):
            snapshot_id = None
            try:
                # Use edge.snapshot_id for downloading the snapshot
                snapshot_id = edge.snapshot_id

                snapshot = graph_nav_client.download_edge_snapshot(snapshot_id)
                # Save snapshot to edge_snapshots subdirectory
                snapshot_file = edge_snapshots_dir / f"edge_snapshot_{snapshot_id}"
                with open(snapshot_file, 'wb') as f:
                    f.write(snapshot.SerializeToString())
            except Exception as e:
                # Handle snapshot_id being undefined if error happens early
                snapshot_id_str = snapshot_id[:8] if snapshot_id else f"edge_{i}"
                failed_edges.append((i, snapshot_id_str, str(e)))
                continue

        # Report results
        success_waypoints = len(graph.waypoints) - len(failed_waypoints)
        success_edges = len(graph.edges) - len(failed_edges)

        print(f"✓ Downloaded {success_waypoints}/{len(graph.waypoints)} waypoint snapshots")
        print(f"✓ Downloaded {success_edges}/{len(graph.edges)} edge snapshots")

        if failed_waypoints:
            print(f"\n⚠ Failed waypoints ({len(failed_waypoints)}):")
            for idx, wp_id, error in failed_waypoints[:5]:  # Show first 5
                print(f"    [{idx}] {wp_id}... - {error[:60]}")
            if len(failed_waypoints) > 5:
                print(f"    ... and {len(failed_waypoints) - 5} more")

        if failed_edges:
            print(f"\n⚠ Failed edges ({len(failed_edges)}):")
            for idx, edge_id, error in failed_edges[:5]:
                print(f"    [{idx}] {edge_id}... - {error[:60]}")
            if len(failed_edges) > 5:
                print(f"    ... and {len(failed_edges) - 5} more")

        print()

    # Generate locations.json (v2 namespaced format).
    #
    # The map we just downloaded becomes the "current map" — record that in
    # maps/.last_used BEFORE writing locations so save_location() lands the
    # entries under the right slice. output_dir may be relative ("maps/foo")
    # or absolute, so take the basename either way.
    map_name = os.path.basename(os.path.normpath(str(output_dir)))
    write_last_used(project_root, map_name)
    print(f"✓ Marked '{map_name}' as the current map (maps/.last_used)")

    if waypoint_names:
        # Route through location_manager so the v2 namespacing is honored.
        for name, wp_id in waypoint_names.items():
            save_location(name, wp_id)

        print("✓ Generated locations.json:")
        for name, wp_id in waypoint_names.items():
            print(f"    '{name}' → {wp_id[:8]}...")
        print()
    else:
        print("⚠ No waypoint names found. You can add names manually to locations.json")
        print()

    print("=" * 60)
    print("MAP DOWNLOAD COMPLETE!")
    print("=" * 60)
    print()

    # Check if there were failures that need to be addressed
    if failed_waypoints or failed_edges:
        total_failed = len(failed_waypoints) + len(failed_edges)
        print(f"⚠ WARNING: {total_failed} snapshot(s) failed to download")
        print("   This may affect navigation reliability.")
        print("   Consider re-recording the map on the tablet if issues persist.")
        print()

    print("Next steps:")
    print(f"1. Upload map: python scripts/setup_map.py --map-path {output_dir}")
    print(f"2. Localize robot using tablet")
    print(f"3. Test voice control: python scripts/run_voice_control.py")
    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download map from Spot")
    parser.add_argument("--output", default="maps/downloaded_map",
                       help="Output directory (default: maps/downloaded_map)")
    args = parser.parse_args()

    download_map(args.output)
