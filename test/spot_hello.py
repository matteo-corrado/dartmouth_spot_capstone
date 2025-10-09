import argparse
import sys
import time

from bosdyn.client import create_standard_sdk
from bosdyn.client.util import authenticate
from bosdyn.client.time_sync import TimeSyncClient, TimeSyncEndpoint
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.power import PowerClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder

def parse_args():
    p = argparse.ArgumentParser(description="Minimal Spot connect/stand/sit demo")
    p.add_argument("--hostname", required=True, help="Robot hostname or IP (e.g., 192.168.80.3)")
    p.add_argument("--username", required=True)
    p.add_argument("--password", required=True)
    p.add_argument("--stand-seconds", type=float, default=5.0, help="How long to stand before sitting")
    return p.parse_args()

def main():
    args = parse_args()

    # 1) SDK & robot object
    sdk = create_standard_sdk("my-spot-app")
    robot = sdk.create_robot(args.hostname)

    # 2) Auth
    authenticate(robot)  # will prompt if no creds, otherwise use below
    robot.authenticate(args.username, args.password)

    # 3) Time sync (required for most commands)
    ts_client: TimeSyncClient = robot.ensure_client(TimeSyncClient.default_service_name)
    ts_endpoint = TimeSyncEndpoint(ts_client)
    ts_endpoint.establish_time_sync()

    # 4) Power on requires a lease; keep it alive while we run
    lease_client: LeaseClient = robot.ensure_client(LeaseClient.default_service_name)
    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # 5) Power on
        power_client: PowerClient = robot.ensure_client(PowerClient.default_service_name)
        if not robot.is_powered_on():
            power_client.power_on(timeout_sec=20)
        robot.time_sync.wait_for_sync()  # just to be safe

        # 6) Stand, wait, then sit
        cmd_client: RobotCommandClient = robot.ensure_client(RobotCommandClient.default_service_name)

        print("Standing…")
        blocking_stand(cmd_client, timeout_sec=15)

        time.sleep(max(0.0, args.stand_seconds))

        print("Sitting…")
        sit_cmd = RobotCommandBuilder.synchro_sit_command()
        cmd_client.robot_command(sit_cmd)
        time.sleep(2.0)

        # 7) (Optional) Power off if we powered it on
        print("Powering off…")
        power_client.power_off(cut_immediately=False, timeout_sec=20)

    print("Done.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
