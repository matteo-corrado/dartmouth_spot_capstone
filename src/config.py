import os
from dotenv import load_dotenv

load_dotenv()

# Security: NO hardcoded defaults - must be set in environment
BOSDYN_ROBOT_IP = os.getenv("BOSDYN_ROBOT_IP")
BOSDYN_CLIENT_USERNAME = os.getenv("BOSDYN_CLIENT_USERNAME")
BOSDYN_CLIENT_PASSWORD = os.getenv("BOSDYN_CLIENT_PASSWORD")

if not all([BOSDYN_ROBOT_IP, BOSDYN_CLIENT_USERNAME, BOSDYN_CLIENT_PASSWORD]):
    raise ValueError(
        "Missing required environment variables. Please set:\n"
        "- BOSDYN_ROBOT_IP\n"
        "- BOSDYN_CLIENT_USERNAME\n"
        "- BOSDYN_CLIENT_PASSWORD\n"
        "in your .env file or shell environment."
    )