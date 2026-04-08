#!/usr/bin/env python3
"""Setup script to upload GraphNav map and initialize localization."""
import sys
import pathlib
import argparse

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.session import spot_session
from src.graph_nav_utils import upload_graph_and_snapshots, initialize_localization, get_localization_state
from src.map_loader import resolve_map_path, write_last_used


def main():
    parser = argparse.ArgumentParser(description="Upload GraphNav map and initialize localization")
    parser.add_argument("--map-path", type=str, default=None,
                        help="Map name (e.g. 'jackson') or path to map directory. "
                             "If omitted, uses maps/.last_used or the most recent map by graph mtime.")
    parser.add_argument("--no-localize", action="store_true",
                        help="Skip localization initialization")
    parser.add_argument("--waypoint-init", action="store_true",
                        help="Use waypoint-based initialization instead of fiducial")
    args = parser.parse_args()

    project_root = pathlib.Path(__file__).resolve().parents[1]

    # Resolve map path via the shared helper:
    #   - bare name like "jackson" -> <project_root>/maps/jackson
    #   - relative or absolute path -> validated as-is
    #   - None -> last-used (or newest by graph mtime)
    try:
        map_path = resolve_map_path(args.map_path, project_root)
    except FileNotFoundError as e:
        print(f"\u2717 {e}")
        return 1
    
    print("=" * 60)
    print("GraphNav Map Setup")
    print("=" * 60)
    print(f"\nMap path: {map_path}")
    print("\n⚠️  IMPORTANT: Make sure E-Stop is running in another terminal!")
    print("   Run: python scripts/estop_run.py")
    print("\nPress Enter to continue...")
    input()
    
    with spot_session(stand_on_enter=True, sit_on_exit=False) as session:
        robot = session["robot"]
        
        # Check if map is already uploaded
        print("\n[Checking current map state...]")
        localization = get_localization_state(robot)
        if localization and localization["is_localized"]:
            print(f"   Robot is already localized to waypoint: {localization['waypoint_id']}")
            response = input("   Upload new map anyway? (y/N): ").strip().lower()
            if response != 'y':
                print("   Skipping map upload")
                return 0
        
        # Upload map
        print("\n[1/2] Uploading map...")
        if not upload_graph_and_snapshots(robot, str(map_path)):
            print("✗ Map upload failed")
            return 1

        # Record this map as the most recently deployed one so wakespot,
        # the voice load_map handler, and location_manager all agree on
        # the current map.
        write_last_used(project_root, map_path.name)
        print(f"\u2713 Recorded '{map_path.name}' as last used (maps/.last_used)")

        # Initialize localization
        if not args.no_localize:
            print("\n[2/2] Initializing localization...")
            if args.waypoint_init:
                print("   Using waypoint-based initialization")
                print("   (Robot must be exactly at the first waypoint)")
                use_fiducial = False
            else:
                print("   Using fiducial-based initialization")
                print("   Make sure robot can see a fiducial from the map!")
                use_fiducial = True
            
            print("   Press Enter when ready...")
            input()
            
            if not initialize_localization(robot, use_fiducial=use_fiducial):
                print("\n⚠️  Localization failed. You may need to:")
                if use_fiducial:
                    print("   1. Ensure robot can see a fiducial from the map")
                    print("   2. Or try waypoint-based init with --waypoint-init")
                else:
                    print("   1. Ensure robot is exactly at the first waypoint")
                    print("   2. Or try fiducial-based init (default)")
                print("   3. Or manually localize using GraphNav tools")
                return 1
        else:
            print("\n[2/2] Skipping localization (use --no-localize to skip)")
        
        print("\n" + "=" * 60)
        print("✓ Map setup complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Use 'save location [name]' to save waypoints at specific locations")
        print("     (e.g., 'save location spot_office' when robot is at that location)")
        print("  2. Use 'go to [name]' to navigate to saved locations")
        print("     (e.g., 'go to spot_office')")
        print("\nTo map waypoints interactively, run:")
        print("  python scripts/map_waypoints.py")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
