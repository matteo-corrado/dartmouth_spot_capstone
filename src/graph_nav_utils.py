"""GraphNav utilities for uploading maps and managing localization."""
import os
import pathlib
from typing import List, Optional
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body


def upload_graph_and_snapshots(robot, upload_path: str) -> bool:
    """Upload a GraphNav map to the robot.
    
    Args:
        robot: Authenticated robot instance
        upload_path: Path to directory containing 'graph', 'waypoint_snapshots', and 'edge_snapshots'
    
    Returns:
        True if successful, False otherwise
    """
    upload_path = pathlib.Path(upload_path)
    
    # Verify directory structure
    graph_file = upload_path / "graph"
    waypoint_dir = upload_path / "waypoint_snapshots"
    edge_dir = upload_path / "edge_snapshots"
    
    if not graph_file.exists():
        print(f"✗ Graph file not found: {graph_file}")
        return False
    if not waypoint_dir.exists() or not waypoint_dir.is_dir():
        print(f"✗ Waypoint snapshots directory not found: {waypoint_dir}")
        return False
    if not edge_dir.exists() or not edge_dir.is_dir():
        print(f"✗ Edge snapshots directory not found: {edge_dir}")
        return False
    
    print(f"[GraphNav] Uploading map from {upload_path}...")
    
    graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
    
    # Read graph file
    try:
        with open(graph_file, 'rb') as f:
            graph_data = f.read()
            graph = map_pb2.Graph()
            graph.ParseFromString(graph_data)
    except Exception as e:
        print(f"✗ Failed to read graph file: {e}")
        return False
    
    print(f"[GraphNav] Graph contains {len(graph.waypoints)} waypoints and {len(graph.edges)} edges")
    
    # Upload graph
    print("[GraphNav] Uploading graph structure...")
    try:
        graph_nav_client.upload_graph(graph=graph)
        print("✓ Graph uploaded")
    except Exception as e:
        print(f"✗ Failed to upload graph: {e}")
        return False
    
    # Upload waypoint snapshots
    print("[GraphNav] Uploading waypoint snapshots...")
    waypoint_snapshots = sorted(waypoint_dir.glob("snapshot_*"))
    if not waypoint_snapshots:
        print("⚠️  No waypoint snapshots found")
    else:
        for i, snapshot_file in enumerate(waypoint_snapshots, 1):
            try:
                with open(snapshot_file, 'rb') as f:
                    snapshot_data = f.read()
                    snapshot = map_pb2.WaypointSnapshot()
                    snapshot.ParseFromString(snapshot_data)
                    graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot=snapshot)
                if i % 5 == 0 or i == len(waypoint_snapshots):
                    print(f"  Progress: {i}/{len(waypoint_snapshots)} snapshots uploaded")
            except Exception as e:
                print(f"✗ Failed to upload snapshot {snapshot_file.name}: {e}")
                return False
        
        print("✓ All waypoint snapshots uploaded")
    
    # Upload edge snapshots
    print("[GraphNav] Uploading edge snapshots...")
    edge_snapshots = sorted(edge_dir.glob("edge_snapshot_*"))
    if not edge_snapshots:
        print("⚠️  No edge snapshots found")
    else:
        for i, snapshot_file in enumerate(edge_snapshots, 1):
            try:
                with open(snapshot_file, 'rb') as f:
                    snapshot_data = f.read()
                    snapshot = map_pb2.EdgeSnapshot()
                    snapshot.ParseFromString(snapshot_data)
                    graph_nav_client.upload_edge_snapshot(edge_snapshot=snapshot)
                if i % 5 == 0 or i == len(edge_snapshots):
                    print(f"  Progress: {i}/{len(edge_snapshots)} snapshots uploaded")
            except Exception as e:
                print(f"✗ Failed to upload edge snapshot {snapshot_file.name}: {e}")
                return False
        
        print("✓ All edge snapshots uploaded")
    
    print("[GraphNav] ✓ Map upload complete!")
    return True


def initialize_localization(robot, use_fiducial: bool = True) -> bool:
    """Initialize robot localization to the map.
    
    Args:
        robot: Authenticated robot instance
        use_fiducial: If True, try to localize using fiducial. Otherwise, use waypoint.
    
    Returns:
        True if localized successfully, False otherwise
    """
    graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
    
    # Check current localization
    try:
        localization = graph_nav_client.get_localization_state()
        if localization.localization.waypoint_id:
            print(f"[GraphNav] Already localized to waypoint: {localization.localization.waypoint_id}")
            return True
    except Exception as e:
        print(f"[GraphNav] Warning: Could not check localization state: {e}")
    
    print("[GraphNav] Not localized. Attempting to initialize...")
    
    try:
        if use_fiducial:
            print("[GraphNav] Attempting fiducial-based localization...")
            print("   (Make sure robot can see a fiducial from the map)")
            # For fiducial init, provide an empty localization as initial guess
            empty_localization = nav_pb2.Localization()
            graph_nav_client.set_localization(
                initial_guess_localization=empty_localization,
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST
            )
        else:
            # Get first waypoint from map
            graph = graph_nav_client.download_graph()
            if not graph.waypoints:
                print("✗ No waypoints in map")
                return False
            first_waypoint_id = graph.waypoints[0].id
            print(f"[GraphNav] Attempting waypoint-based localization to: {first_waypoint_id}")
            print("   (Robot must be exactly at this waypoint)")
            
            # Get current robot state for odometry
            from bosdyn.client.robot_state import RobotStateClient
            robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
            robot_state = robot_state_client.get_robot_state()
            current_odom_tform_body = get_odom_tform_body(
                robot_state.kinematic_state.transforms_snapshot).to_proto()
            
            # Create an initial localization to the specified waypoint
            localization = nav_pb2.Localization()
            localization.waypoint_id = first_waypoint_id
            localization.waypoint_tform_body.rotation.w = 1.0
            
            graph_nav_client.set_localization(
                initial_guess_localization=localization,
                max_distance=0.2,  # Search +/-20cm
                max_yaw=20.0 * 3.14159 / 180.0,  # Search +/-20 degrees
                fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
                ko_tform_body=current_odom_tform_body
            )
        
        # Verify localization
        localization = graph_nav_client.get_localization_state()
        if localization.localization.waypoint_id:
            print(f"✓ Localized to waypoint: {localization.localization.waypoint_id}")
            return True
        else:
            print("✗ Localization failed - robot may not see a fiducial or be at the waypoint")
            return False
    except Exception as e:
        print(f"✗ Localization error: {e}")
        import traceback
        traceback.print_exc()
        return False


def list_waypoints(robot) -> List[str]:
    """List all waypoints in the current map.
    
    Args:
        robot: Authenticated robot instance
    
    Returns:
        List of waypoint IDs
    """
    graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
    try:
        graph = graph_nav_client.download_graph()
        waypoint_ids = [wp.id for wp in graph.waypoints]
        return waypoint_ids
    except Exception as e:
        print(f"✗ Failed to list waypoints: {e}")
        return []


def get_localization_state(robot) -> Optional[dict]:
    """Get current localization state.
    
    Args:
        robot: Authenticated robot instance
    
    Returns:
        Dict with localization info, or None if error
    """
    graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
    try:
        localization = graph_nav_client.get_localization_state()
        return {
            "waypoint_id": localization.localization.waypoint_id,
            "is_localized": bool(localization.localization.waypoint_id),
        }
    except Exception as e:
        print(f"✗ Failed to get localization state: {e}")
        return None
