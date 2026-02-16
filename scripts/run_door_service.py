#!/usr/bin/env python3
"""Run the door-opening Remote Mission Service for Spot AutoWalks.

This starts a gRPC server on the Jetson that Spot calls at configured
waypoints during AutoWalk playback to open doors.

Usage:
    # Get the Jetson's IP as seen by Spot:
    HOST_IP=$(python3 -m bosdyn.client $SPOT_IP self-ip)

    # Start the door service:
    python scripts/run_door_service.py --host-ip $HOST_IP --port 50052

    # Then on the tablet:
    #   1. Record AutoWalk
    #   2. At a door, tap "+" → "Create New Action" → "Remote GRPC"
    #   3. Select "door-opening-service"
    #   4. Configure: door_type=button/handle, door_direction=push/pull, etc.
    #   5. Continue recording
    #   6. On playback, Spot calls this service at the door waypoint

Requires:
    - E-Stop running (scripts/estop_run.py in separate terminal)
    - Robot powered on and standing
"""

import sys
import pathlib
import argparse
import logging
import time

# Add project root to path
project_root = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import grpc
from concurrent import futures

from bosdyn.api.mission import remote_service_pb2_grpc
from bosdyn.client import create_standard_sdk
from bosdyn.client.util import setup_logging
from bosdyn.client.directory_registration import DirectoryRegistrationClient, DirectoryRegistrationKeepAlive

from src.door_service.door_mission_service import DoorOpeningServicer


SERVICE_NAME = "door-opening-service"
SERVICE_TYPE = "bosdyn.api.mission.RemoteMissionService"
SERVICE_AUTHORITY = "door-service"


def main():
    parser = argparse.ArgumentParser(description="Spot Door Opening Remote Mission Service")
    parser.add_argument("--host-ip", required=True,
                        help="IP of this machine as seen by the robot (use: python3 -m bosdyn.client $SPOT_IP self-ip)")
    parser.add_argument("--port", type=int, default=50052,
                        help="Port for the gRPC server (default: 50052)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level=log_level)
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("SPOT DOOR OPENING SERVICE")
    print("=" * 60)
    print(f"\n  Host IP: {args.host_ip}")
    print(f"  Port:    {args.port}")
    print(f"\n  IMPORTANT: Make sure E-Stop is running!")
    print(f"  Run in a separate terminal: python scripts/estop_run.py")
    print()

    # Connect to robot
    import os
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    spot_ip = os.getenv("BOSDYN_ROBOT_IP", "192.168.80.3")
    spot_user = os.getenv("BOSDYN_CLIENT_USERNAME", "SpotDBEC")
    spot_pass = os.getenv("BOSDYN_CLIENT_PASSWORD")

    if not spot_pass:
        print("ERROR: Set BOSDYN_CLIENT_PASSWORD in .env file")
        return 1

    print(f"Connecting to Spot at {spot_ip}...")
    sdk = create_standard_sdk("DoorOpeningService")
    robot = sdk.create_robot(spot_ip)
    robot.authenticate(spot_user, spot_pass)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()
    print("Connected to Spot!")

    # Create gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    servicer = DoorOpeningServicer(robot)
    remote_service_pb2_grpc.add_RemoteMissionServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{args.port}")
    server.start()
    print(f"\ngRPC server listening on port {args.port}")

    # Register with robot's directory service
    print("Registering service with Spot directory...")
    try:
        dir_reg_client = robot.ensure_client(DirectoryRegistrationClient.default_service_name)
        keep_alive = DirectoryRegistrationKeepAlive(
            dir_reg_client,
            logger=logger
        )
        keep_alive.start(
            directory_name=SERVICE_NAME,
            service_type=SERVICE_TYPE,
            authority=SERVICE_AUTHORITY,
            host_ip=args.host_ip,
            port=args.port,
        )
        print(f"Service registered as '{SERVICE_NAME}'")
    except Exception as e:
        print(f"WARNING: Could not register with directory: {e}")
        print("The service is running but may not appear on the tablet.")
        print("You may need to configure it manually.")
        keep_alive = None

    print("\n" + "=" * 60)
    print("DOOR SERVICE READY")
    print("=" * 60)
    print("\nOn the tablet:")
    print("  1. Record AutoWalk → at a door → '+' → 'Create New Action' → 'Remote GRPC'")
    print(f"  2. Select '{SERVICE_NAME}'")
    print("  3. Configure: door_type, door_direction, heights, etc.")
    print("\nPress Ctrl+C to stop...\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down door service...")
        if keep_alive:
            keep_alive.shutdown()
        server.stop(grace=5)
        print("Door service stopped.")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
