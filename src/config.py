"""Environment configuration for Spot robot connection and Dartmouth APIs."""
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Robot ----------
BOSDYN_ROBOT_IP = os.getenv("BOSDYN_ROBOT_IP", "192.168.80.3")
BOSDYN_CLIENT_USERNAME = os.getenv("BOSDYN_CLIENT_USERNAME", "")
BOSDYN_CLIENT_PASSWORD = os.getenv("BOSDYN_CLIENT_PASSWORD", "")

# ---------- Dartmouth Developer API (STT, raw REST) ----------
DARTMOUTH_API_KEY = os.getenv("DARTMOUTH_API_KEY", "")
DARTMOUTH_JWT_URL = os.getenv(
    "DARTMOUTH_JWT_URL", "https://api.dartmouth.edu/api/jwt"
)
DARTMOUTH_STT_URL = os.getenv(
    "DARTMOUTH_STT_URL",
    "https://api.dartmouth.edu/api/ai/speech-recognition",
)

# ---------- Dartmouth Chat API (LLM / VLM via langchain-dartmouth) ----------
DARTMOUTH_CHAT_API_KEY = os.getenv("DARTMOUTH_CHAT_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "meta.llama-3-1-8b-instruct")
VLM_MODEL = os.getenv("VLM_MODEL", "openai.gpt-4.1-mini-2025-04-14")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "150"))
VLM_MAX_TOKENS = int(os.getenv("VLM_MAX_TOKENS", "200"))