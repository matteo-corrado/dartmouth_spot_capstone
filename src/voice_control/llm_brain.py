"""LLM Brain for Spot — conversational robot control via Ollama structured JSON.

Uses Ollama's format="json" to guarantee valid JSON output from the model.
The model decides which action to perform (if any) AND produces a spoken
response, all in a single JSON object. No tool calling protocol needed —
small models are much more reliable at filling in JSON fields.

Architecture inspired by Boston Dynamics' "Robots That Can Chat" demo, adapted
for local inference on Jetson AGX Orin via Ollama.

Models (recommended):
    ollama pull qwen2.5:7b       # Good balance of speed + reasoning
    ollama pull qwen2.5:14b      # Best quality, slower (~1-3s on Jetson)
    ollama pull qwen2.5:3b       # Fastest, less conversational
"""

import json
import time
import base64
import requests
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5:7b"
VLM_MODEL = "qwen2.5-vl:7b"
OLLAMA_URL = "http://localhost:11434"
MAX_HISTORY = 20          # messages (10 user + 10 assistant exchanges)
REQUEST_TIMEOUT = 30.0    # seconds per request
FIRST_REQUEST_TIMEOUT = 120.0  # seconds — model loading into VRAM can be slow
VLM_TIMEOUT = 60.0       # seconds — VLM inference is slower

# ---------------------------------------------------------------------------
# System prompt — includes action catalog and JSON output schema
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Spot, a Boston Dynamics quadruped robot at Dartmouth College. \
You are a friendly, helpful general assistant. You are self-aware: you know \
you are a four-legged robot, you can walk, navigate, and perform physical \
actions. You have a sense of humor and personality. Keep responses concise \
(1-2 sentences for actions, a bit more for conversation).

You MUST respond with a JSON object. Every response must be valid JSON with exactly these fields IN THIS ORDER:
{"action": "<action_name or null>", "params": {<parameters or empty>}, "response": "<what you say>"}
IMPORTANT: Always output "action" first, then "params", then "response". This ordering is required.

AVAILABLE ACTIONS:
- stop: Stop all movement. No params.
- freeze: Hold current position. No params.
- estop: Emergency stop (urgent only). No params.
- stand: Stand up from sitting. No params.
- sit: Sit down. No params.
- selfright: Recover from a fall. No params.
- walk: Walk forward/backward. Params: {"direction": "forward"|"backward", "distance": <meters>}. Default distance 1.0.
- strafe: Move sideways. Params: {"direction": "left"|"right", "distance": <meters>}. Default distance 0.5.
- turn: Rotate in place. Params: {"deg": <degrees 0-360>, "dir": "left"|"right"}. Default 90 degrees. "turn around" = 180.
- body_height: Adjust height. Params: {"height": <-0.15 to 0.1>}. -0.15=crouch, 0=normal, 0.1=tall.
- set_speed: Set speed. Params: {"speed": "slow"|"normal"|"fast"}.
- go_to: Navigate to saved location. Params: {"location": "<name>"}. Use lowercase_with_underscores.
- tour: Visit locations in sequence (one pass). Params: {"locations": ["loc1", "loc2"]} for specific stops, or {"locations": "all"} to visit all. Use for "loop the map", "visit everywhere", "go to A then B then C".
- patrol: Loop through locations continuously until stopped. Params: same as tour. Use for "patrol", "keep looping", "keep patrolling".
- come_back: Return to position before last navigation. No params. Use for "come back", "go home", "return".
- save_location: Save current position. Params: {"location": "<name>"}.
- list_locations: List saved locations. No params.
- open_door: Open a push-button door. Extends arm to press button, holds door open while walking through, then stows arm. No params needed (uses tuned defaults). Use for "open the door", "push the door", "open door".
- describe: Take a photo and describe what you see. Use when user asks about surroundings, what's in front of you, etc. Params: {"camera": "front"|"left"|"right"|"back"}. Default "front".
- battery_status: Check battery level. No params.
- status: Full robot status report. No params.
- power_off: Safely power off. No params.

RULES:
- When the user asks you to do something physical, set "action" to the action name and include params.
- When the user is just chatting, set "action" to null and "params" to {}.
- ALWAYS include a "response" — a short spoken reply.
- Location names must be lowercase with underscores.
- If you cannot do something (browse web, send email), say so and set action to null.

EXAMPLES:
User: "Stand up"
{"action": "stand", "params": {}, "response": "Standing up!"}

User: "Walk forward 2 meters"
{"action": "walk", "params": {"direction": "forward", "distance": 2.0}, "response": "Walking forward 2 meters!"}

User: "Turn around"
{"action": "turn", "params": {"deg": 180, "dir": "left"}, "response": "Turning around!"}

User: "How are you doing?"
{"action": null, "params": {}, "response": "I'm doing great! Battery is looking good and I'm ready to help."}

User: "Go to the kitchen"
{"action": "go_to", "params": {"location": "kitchen"}, "response": "On my way to the kitchen!"}

User: "What do you see?"
{"action": "describe", "params": {"camera": "front"}, "response": "Let me take a look..."}

User: "What's to your left?"
{"action": "describe", "params": {"camera": "left"}, "response": "Let me check what's on my left..."}

User: "Loop the map"
{"action": "tour", "params": {"locations": "all"}, "response": "Starting a tour of all locations!"}

User: "Go to kitchen then hallway then lab"
{"action": "tour", "params": {"locations": ["kitchen", "hallway", "lab"]}, "response": "On my way! I'll visit kitchen, hallway, and lab in order."}

User: "Patrol the map"
{"action": "patrol", "params": {"locations": "all"}, "response": "Starting patrol! I'll keep looping until you tell me to stop."}

User: "Come back"
{"action": "come_back", "params": {}, "response": "Heading back to where I started!"}

User: "Open the door"
{"action": "open_door", "params": {}, "response": "Opening the door! Stand back."}"""


class SpotBrain:
    """Conversational LLM brain for Spot robot using Ollama structured JSON.

    The model receives the action catalog in the system prompt and responds
    with a JSON object containing action + response. Ollama enforces valid
    JSON at the grammar level, so parsing is reliable.
    """

    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url
        self.history: List[Dict[str, Any]] = []
        self._available = None  # cached availability check
        self._first_request = True

    def is_available(self) -> bool:
        """Check if Ollama is running and the model is pulled."""
        if self._available is not None:
            return self._available
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=3)
            if r.status_code != 200:
                self._available = False
                return False
            models = [m["name"] for m in r.json().get("models", [])]
            found = self.model in models or f"{self.model}:latest" in models
            if not found:
                print(f"[Brain] Model '{self.model}' not found. Available: {models}")
                print(f"[Brain] Run: ollama pull {self.model}")
            self._available = found
            return found
        except Exception:
            self._available = False
            return False

    def warm_up(self):
        """Send a trivial prompt to pre-load model into VRAM."""
        if not self.is_available():
            return
        print(f"[Brain] Warming up model '{self.model}'...")
        t0 = time.time()
        try:
            # Use the actual system prompt so Ollama caches its KV state.
            # This makes the first real command fast (~0.2s prompt eval
            # instead of ~1.5s cold).
            warm_state = {"battery_percent": "unknown", "is_powered": True,
                          "is_standing": "unknown", "saved_locations": "none"}
            messages = self._build_messages("ping", warm_state)
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {"num_predict": 10, "num_gpu": 99, "num_ctx": 4096},
                },
                timeout=FIRST_REQUEST_TIMEOUT,
            )
            elapsed = time.time() - t0
            self._first_request = False
            if r.status_code == 200:
                print(f"[Brain] Model warm in {elapsed:.1f}s (prompt cached)")
            else:
                print(f"[Brain] Warm-up got status {r.status_code}")
        except Exception as e:
            print(f"[Brain] Warm-up error: {e}")

    def _build_messages(self, transcript: str, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build the message list for the Ollama chat API."""
        state_lines = "\n".join(f"- {k}: {v}" for k, v in state.items())
        system_content = SYSTEM_PROMPT + f"\n\nCurrent robot state:\n{state_lines}"

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": transcript})

        return messages

    def process(self, transcript: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user utterance and return action + response.

        Args:
            transcript: What the user said (ASR output).
            state: Current robot state dict (battery, location, etc.).

        Returns:
            Dict with keys:
                "action": {"intent": str, "params": dict} or None
                "response": str  — what the robot says back
                "raw_llm": str   — raw LLM output (for debugging)
        """
        if state is None:
            state = {}

        messages = self._build_messages(transcript, state)

        try:
            timeout = FIRST_REQUEST_TIMEOUT if self._first_request else REQUEST_TIMEOUT
            if self._first_request:
                print("[Brain] First request — loading model, this may take a moment...")

            t0 = time.time()
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "format": "json",
                    "stream": False,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 100,
                        "num_gpu": 99,
                        "num_ctx": 4096,
                    },
                },
                timeout=timeout,
            )
            self._first_request = False

            if r.status_code != 200:
                print(f"[Brain] Ollama error {r.status_code}: {r.text[:200]}")
                return {"action": None, "response": "", "raw_llm": ""}

            message = r.json().get("message", {})
            content = (message.get("content") or "").strip()
            elapsed = time.time() - t0
            print(f"[Brain] LLM responded in {elapsed:.1f}s ({len(content)} chars)")

            # --- Parse JSON ---
            action = None
            response = ""

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"[Brain] Failed to parse JSON: {content[:200]}")
                return {"action": None, "response": content, "raw_llm": content}

            response = str(data.get("response", "")).strip()

            action_name = data.get("action")
            if action_name and isinstance(action_name, str) and action_name.lower() != "null":
                params = data.get("params", {})
                if not isinstance(params, dict):
                    params = {}
                params = self._normalize_params(action_name, params)
                action = {"intent": action_name, "params": params}
                print(f"[Brain] Action: {action_name}({params})")
            else:
                print("[Brain] No action (conversation only)")

            # Update conversation history
            self.history.append({"role": "user", "content": transcript})
            self.history.append({"role": "assistant", "content": content})

            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]

            return {
                "action": action,
                "response": response,
                "raw_llm": content,
            }

        except requests.Timeout:
            print(f"[Brain] Timeout after {timeout}s — model may be loading")
            return {"action": None, "response": "", "raw_llm": ""}
        except requests.ConnectionError:
            print("[Brain] Cannot connect to Ollama. Is it running?")
            self._available = False
            return {"action": None, "response": "", "raw_llm": ""}
        except Exception as e:
            print(f"[Brain] Error: {e}")
            return {"action": None, "response": "", "raw_llm": ""}

    @staticmethod
    def _normalize_params(intent: str, params: dict) -> dict:
        """Normalize and validate action parameters."""
        if isinstance(params, str):
            try:
                params = json.loads(params)
            except (json.JSONDecodeError, TypeError):
                params = {}

        if not isinstance(params, dict):
            return {}

        # Normalize location names
        if "location" in params:
            loc = str(params["location"]).strip().lower().replace(" ", "_")
            params["location"] = loc

        # Normalize locations list (tour/patrol)
        if "locations" in params:
            locs = params["locations"]
            if isinstance(locs, list):
                params["locations"] = [
                    str(l).strip().lower().replace(" ", "_") for l in locs
                ]
            elif isinstance(locs, str) and locs != "all":
                params["locations"] = [locs.strip().lower().replace(" ", "_")]

        # Clamp degrees
        if "deg" in params:
            try:
                deg = float(params["deg"])
                deg = max(0, min(360, deg))
                params["deg"] = int(deg) if deg == int(deg) else deg
            except (ValueError, TypeError):
                params["deg"] = 90

        # Ensure distance is float
        if "distance" in params:
            try:
                params["distance"] = float(params["distance"])
            except (ValueError, TypeError):
                params["distance"] = 1.0

        # Ensure height is float
        if "height" in params:
            try:
                params["height"] = float(params["height"])
            except (ValueError, TypeError):
                params["height"] = 0.0

        return params

    def query_vlm(self, image_bytes: bytes, question: str) -> str:
        """Send image + question to the VLM for visual description.

        Args:
            image_bytes: JPEG image data from Spot camera.
            question: The user's original question (e.g. "what do you see?").

        Returns:
            VLM's text response describing the image.
        """
        image_b64 = base64.b64encode(image_bytes).decode()

        vlm_prompt = (
            f"You are Spot, a Boston Dynamics robot at Dartmouth College. "
            f"A user asked: \"{question}\". Describe what you see in 2-3 sentences. "
            f"Be specific about objects, people, and surroundings."
        )

        try:
            print(f"[Brain] Querying VLM ({VLM_MODEL})...")
            t0 = time.time()
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": VLM_MODEL,
                    "messages": [
                        {"role": "user", "content": vlm_prompt, "images": [image_b64]}
                    ],
                    "stream": False,
                    "keep_alive": "5m",
                    "options": {"num_gpu": 99, "num_predict": 200},
                },
                timeout=VLM_TIMEOUT,
            )
            elapsed = time.time() - t0

            if r.status_code != 200:
                print(f"[Brain] VLM error {r.status_code}: {r.text[:200]}")
                return "Sorry, I couldn't process the image right now."

            content = r.json().get("message", {}).get("content", "").strip()
            print(f"[Brain] VLM responded in {elapsed:.1f}s")
            return content or "I can see the image but I'm having trouble describing it."

        except requests.Timeout:
            print(f"[Brain] VLM timeout after {VLM_TIMEOUT}s")
            return "Sorry, the image analysis took too long."
        except requests.ConnectionError:
            print("[Brain] VLM cannot connect to Ollama")
            return "Sorry, I can't access my vision system right now."
        except Exception as e:
            print(f"[Brain] VLM error: {e}")
            return "Sorry, something went wrong with my vision."

    def clear_history(self):
        """Clear conversation history."""
        self.history.clear()


# ---------------------------------------------------------------------------
# Module-level convenience (used by client_mic.py)
# ---------------------------------------------------------------------------
_brain: Optional[SpotBrain] = None


def get_brain(model: str = DEFAULT_MODEL) -> SpotBrain:
    """Get or create the singleton SpotBrain instance."""
    global _brain
    if _brain is None or _brain.model != model:
        _brain = SpotBrain(model=model)
    return _brain


def process_with_brain(transcript: str, state: Optional[Dict[str, Any]] = None,
                       model: str = DEFAULT_MODEL) -> Dict[str, Any]:
    """Convenience function: process transcript through the LLM brain."""
    brain = get_brain(model)
    return brain.process(transcript, state)


# ---------------------------------------------------------------------------
# CLI test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Spot LLM Brain — Interactive Test (Structured JSON)")
    print("=" * 60)

    import sys
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    brain = SpotBrain(model=model)

    if not brain.is_available():
        print(f"\nOllama is not running or model '{model}' is not available.")
        print(f"1. Start Ollama:  sudo systemctl start ollama")
        print(f"2. Pull model:    ollama pull {model}")
        sys.exit(1)

    print(f"Using model: {model}")
    print("Type messages as if speaking to Spot. Type 'quit' to exit.\n")

    # Simulate some robot state
    fake_state = {
        "battery_percent": 82,
        "estimated_runtime_minutes": 95,
        "is_powered": True,
        "is_standing": True,
        "current_location": "unknown",
        "saved_locations": "work_area, spot_room, couch, corner, start",
        "estop_status": "not_cut",
    }

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text or text.lower() == "quit":
            break

        result = brain.process(text, fake_state)

        if result["response"]:
            print(f"Spot: {result['response']}")
        if result["action"]:
            a = result["action"]
            print(f"  -> Action: {a['intent']}({a.get('params', {})})")
        else:
            print("  -> (no action)")
        print()
