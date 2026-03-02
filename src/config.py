import os
from dotenv import load_dotenv

load_dotenv()

BOSDYN_ROBOT_IP = os.getenv("BOSDYN_ROBOT_IP", "")
BOSDYN_CLIENT_USERNAME = os.getenv("BOSDYN_CLIENT_USERNAME", "")
BOSDYN_CLIENT_PASSWORD = os.getenv("BOSDYN_CLIENT_PASSWORD", "")