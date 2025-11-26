"""Location manager for saving and loading named waypoints.

Automatically saves current waypoint positions when you say "save location [name]".
"""
import json
import pathlib
from typing import Optional, Dict

LOCATIONS_FILE = pathlib.Path(__file__).resolve().parents[1] / "locations.json"

def save_location(name: str, waypoint_id: str) -> bool:
    """Save a named location with its waypoint ID.
    
    Args:
        name: Human-readable location name (e.g., "A", "B", "kitchen")
        waypoint_id: GraphNav waypoint ID from current localization
    
    Returns:
        True if saved successfully
    """
    # Load existing locations
    locations = load_all_locations()
    
    # Add new location
    locations[name.lower()] = waypoint_id
    
    # Save to file
    LOCATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOCATIONS_FILE, 'w') as f:
        json.dump(locations, f, indent=2)
    
    print(f"✓ Saved location '{name}' -> waypoint {waypoint_id}")
    return True

def load_location(name: str) -> Optional[str]:
    """Load waypoint ID for a named location.
    
    Args:
        name: Location name (e.g., "B", "kitchen")
    
    Returns:
        Waypoint ID string, or None if not found
    """
    locations = load_all_locations()
    return locations.get(name.lower())

def load_all_locations() -> Dict[str, str]:
    """Load all saved locations.
    
    Returns:
        Dict mapping location names to waypoint IDs
    """
    if not LOCATIONS_FILE.exists():
        return {}
    try:
        with open(LOCATIONS_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def list_locations() -> Dict[str, str]:
    """List all saved location mappings."""
    return load_all_locations()