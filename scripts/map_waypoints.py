#!/usr/bin/env python3
"""Helper script to list waypoints and create location mappings."""
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.session import spot_session
from src.graph_nav_utils import list_waypoints, get_localization_state
from src.location_manager import save_location, list_locations


def format_waypoint_id(wp_id: str, max_length: int = 50) -> str:
    """Format waypoint ID for display."""
    if len(wp_id) <= max_length:
        return wp_id
    # Show first part and last part
    return f"{wp_id[:20]}...{wp_id[-20:]}"


def get_short_code(wp_id: str) -> str:
    """Generate a short code from waypoint ID."""
    # Extract first 2 chars of first two words
    parts = wp_id.split('-')
    if len(parts) >= 2:
        return (parts[0][:2] + parts[1][:2]).lower()
    elif len(parts) == 1:
        return parts[0][:4].lower()
    else:
        return wp_id[:4].lower()


def main():
    print("=" * 60)
    print("Waypoint Mapper")
    print("=" * 60)
    print("\nThis tool helps you map waypoint IDs to human-readable location names.")
    print("You can either:")
    print("  1. List all waypoints and map them by number")
    print("  2. Use current robot position to save a location")
    print()
    
    with spot_session(stand_on_enter=False, sit_on_exit=False) as session:
        robot = session["robot"]
        
        # Check localization
        localization = get_localization_state(robot)
        if localization and localization["is_localized"]:
            print(f"✓ Robot is localized to waypoint: {format_waypoint_id(localization['waypoint_id'])}")
        else:
            print("⚠️  Robot is not localized. Some features may not work.")
            print("   Run 'python scripts/setup_map.py' first to upload and localize.")
        
        # List existing locations
        existing_locations = list_locations()
        if existing_locations:
            print(f"\nExisting location mappings ({len(existing_locations)}):")
            for name, wp_id in existing_locations.items():
                print(f"  • {name:20s} -> {format_waypoint_id(wp_id)}")
        else:
            print("\nNo location mappings yet.")
        
        # List all waypoints
        print("\n[Loading waypoints from map...]")
        waypoint_ids = list_waypoints(robot)
        
        if not waypoint_ids:
            print("✗ No waypoints found. Make sure map is uploaded.")
            print("   Run 'python scripts/setup_map.py' first.")
            return 1
        
        print(f"\nFound {len(waypoint_ids)} waypoints in map:")
        print("-" * 60)
        for i, wp_id in enumerate(waypoint_ids, 1):
            short_code = get_short_code(wp_id)
            # Check if this waypoint is already mapped
            mapped_to = [name for name, wid in existing_locations.items() if wid == wp_id]
            mapped_str = f" (mapped to: {', '.join(mapped_to)})" if mapped_to else ""
            print(f"  {i:2d}. [{short_code:4s}] {format_waypoint_id(wp_id)}{mapped_str}")
        print("-" * 60)
        
        # Interactive mapping
        print("\nCommands:")
        print("  <number> <name>  - Map waypoint number to location name")
        print("  current <name>   - Save current robot position as location")
        print("  list             - Show all existing mappings")
        print("  q                - Quit")
        
        while True:
            print("\n> ", end="")
            try:
                response = input().strip()
                if not response:
                    continue
                
                if response.lower() == 'q':
                    break
                
                if response.lower() == 'list':
                    locations = list_locations()
                    if locations:
                        print("\nAll location mappings:")
                        for name, wp_id in locations.items():
                            print(f"  • {name:20s} -> {format_waypoint_id(wp_id)}")
                    else:
                        print("No location mappings yet.")
                    continue
                
                parts = response.split(maxsplit=1)
                if len(parts) < 2:
                    print("Format: <number> <location_name> or 'current <location_name>'")
                    continue
                
                if parts[0].lower() == 'current':
                    # Save current position
                    if not localization or not localization["is_localized"]:
                        print("✗ Robot is not localized. Cannot save current position.")
                        continue
                    
                    location_name = parts[1].lower().strip()
                    waypoint_id = localization["waypoint_id"]
                    save_location(location_name, waypoint_id)
                    print(f"✓ Mapped '{location_name}' -> {format_waypoint_id(waypoint_id)}")
                    # Update localization state
                    localization = get_localization_state(robot)
                    existing_locations = list_locations()
                
                else:
                    # Map by waypoint number
                    try:
                        wp_num = int(parts[0]) - 1
                        location_name = parts[1].lower().strip()
                        
                        if 0 <= wp_num < len(waypoint_ids):
                            waypoint_id = waypoint_ids[wp_num]
                            save_location(location_name, waypoint_id)
                            print(f"✓ Mapped '{location_name}' -> {format_waypoint_id(waypoint_id)}")
                            existing_locations = list_locations()
                        else:
                            print(f"✗ Invalid waypoint number (1-{len(waypoint_ids)})")
                    except ValueError:
                        print("✗ Invalid waypoint number")
            
            except KeyboardInterrupt:
                print("\n\nExiting...")
                break
            except Exception as e:
                print(f"✗ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n✓ Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
