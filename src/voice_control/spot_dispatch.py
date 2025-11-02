from bosdyn.client import create_standard_sdk
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand
from bosdyn.client.lease import LeaseClient
from bosdyn.client.estop import EstopClient
from bosdyn.client.util import authenticate

def connect(robot_hostname, username, password):
    sdk = create_standard_sdk("SpotVoicePOCClient")
    robot = sdk.create_robot(robot_hostname)
    authenticate(robot)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_command = robot.ensure_client(RobotCommandClient.default_service_name)
    estop = robot.ensure_client(EstopClient.default_service_name)
    return robot, lease_client, robot_command, estop

def dispatch_intent(intent, robot_command):
    name = intent["intent"]
    p = intent.get("params", {})
    if name == "estop":
        print("[E-STOP] (hook into EstopClient in production)")
    elif name == "stand":
        print("Issuing stand command...")
        blocking_stand(robot_command, timeout_sec=10)
        print("Robot is now standing.")
    elif name == "sit":
        print("Issuing sit command...")
        robot_command.sit()
        print("Robot is now sitting.")
    elif name == "follow":
        print("Would start follow behavior (implement your tracker here).")
    elif name == "walk_to":
        print("Would send a relative walk goal:", p.get("relative"))
    elif name == "turn":
        print(f"Would rotate {p.get('dir')} {p.get('deg')} deg")
    elif name == "ptz_aim":
        print("Would aim PTZ at target:", p.get("target"))
    elif name == "home":
        print("Would send robot to home position.")
    else:
        print("Unknown intent:", name)
