import os
from dotenv import load_dotenv

load_dotenv()

BOSDYN_ROBOT_IP = os.getenv("BOSDYN_ROBOT_IP", "192.168.80.3")
BOSDYN_CLIENT_USERNAME = os.getenv("BOSDYN_CLIENT_USERNAME", "SpotDBEC")
BOSDYN_CLIENT_PASSWORD = os.getenv("BOSDYN_CLIENT_PASSWORD", "Acetabular10")