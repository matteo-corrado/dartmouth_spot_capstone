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
VLM_MODEL = "qwen2.5vl:7b"
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

You MUST respond with a JSON object with exactly these fields:
{"actions": [<list of actions>], "response": "<what you say>"}

Each action in the list: {"action": "<name>", "params": {<parameters>}}
For a single action, use a list with one item.
For chained commands (do X then Y), list them in execution order.
For conversation only (no physical action), use an empty list.

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
- tour: Visit locations in sequence (one pass). Params: {"locations": ["loc1", "loc2"]} for specific stops, or {"locations": "all"} to visit all. Use for "loop the map", "visit everywhere".
- patrol: Loop through locations continuously until stopped. Params: same as tour. Use for "patrol", "keep looping", "keep patrolling".
- come_back: Return to position before last navigation. No params. Use for "come back", "go home", "return".
- save_location: Save current position. Params: {"location": "<name>"}.
- list_locations: List saved locations. No params.
- open_door: Open a push-bar door. No params needed (uses tuned defaults). Use for "open the door", "push the door".
- go_to_object: Walk toward a visible object using the camera. Params: {"description": "<what to find>"}. Use for "go to the red chair", "find the backpack", "walk to the table". Only for objects you can SEE — use go_to for saved map locations.
- follow_me: Follow the nearest person, maintaining distance. No params. Use for "follow me", "come with me", "tag along".
- describe: Take a photo and describe what you see. Params: {"camera": "front"|"left"|"right"|"back", "query": "<specific object to look for, if any>"}. Default camera "front". Omit query for general "what do you see" questions.
- battery_status: Check battery level. No params.
- status: Full robot status report. No params.
- power_off: Safely power off. No params.

RULES:
- When the user asks you to do something physical, put action(s) in the "actions" list.
- When the user is just chatting, use an empty list: "actions": [].
- For chained requests ("do X then Y"), list multiple actions in order — the robot executes them sequentially.
- For multi-location trips ("go to A then B then C"), prefer a SINGLE tour action with a locations list over chaining multiple go_to actions.
- ALWAYS include a "response" — a short spoken reply.
- Location names must be lowercase with underscores.
- IMPORTANT: Check "saved_locations" in the robot state. If the user says "go to X" and X matches a saved location name, ALWAYS use go_to (map navigation). Only use go_to_object for objects NOT in saved_locations (e.g. "go to the red chair" when "red_chair" is not a saved location).
- IMPORTANT: "Do you see X?", "Can you see X?", "Is there a X?" are OBSERVATION questions — use describe (look with camera), NOT go_to_object. Only use go_to_object when the user explicitly says "go to X", "walk to X", "find X", or "approach X".

EXAMPLES:
User: "Stand up"
{"actions": [{"action": "stand", "params": {}}], "response": "Standing up!"}

User: "Walk forward 2 meters"
{"actions": [{"action": "walk", "params": {"direction": "forward", "distance": 2.0}}], "response": "Walking forward 2 meters!"}

User: "Turn around"
{"actions": [{"action": "turn", "params": {"deg": 180, "dir": "left"}}], "response": "Turning around!"}

User: "How are you doing?"
{"actions": [], "response": "I'm doing great! Battery is looking good and I'm ready to help."}

User: "Go to the kitchen"
{"actions": [{"action": "go_to", "params": {"location": "kitchen"}}], "response": "On my way to the kitchen!"}

User: "What do you see?"
{"actions": [{"action": "describe", "params": {"camera": "front"}}], "response": "Let me take a look..."}

User: "Go to kitchen then hallway then lab"
{"actions": [{"action": "tour", "params": {"locations": ["kitchen", "hallway", "lab"]}}], "response": "On my way! I'll visit kitchen, hallway, and lab in order."}

User: "Go to the conference and come back"
{"actions": [{"action": "go_to", "params": {"location": "conference"}}, {"action": "come_back", "params": {}}], "response": "Going to conference and coming right back!"}

User: "Go to lab, then sit down"
{"actions": [{"action": "go_to", "params": {"location": "lab"}}, {"action": "sit", "params": {}}], "response": "Heading to lab, then I'll sit down!"}

User: "Patrol the map"
{"actions": [{"action": "patrol", "params": {"locations": "all"}}], "response": "Starting patrol! I'll keep looping until you tell me to stop."}

User: "Come back"
{"actions": [{"action": "come_back", "params": {}}], "response": "Heading back to where I started!"}

User: "Open the door"
{"actions": [{"action": "open_door", "params": {}}], "response": "Opening the door! Stand back."}

User: "Do you see the blue chair?"
{"actions": [{"action": "describe", "params": {"camera": "front", "query": "blue chair"}}], "response": "Let me check for the blue chair..."}

User: "Go to the red chair"
{"actions": [{"action": "go_to_object", "params": {"description": "red chair"}}], "response": "Looking for the red chair!"}

User: "Find my backpack"
{"actions": [{"action": "go_to_object", "params": {"description": "backpack"}}], "response": "Searching for your backpack!"}

User: "Follow me"
{"actions": [{"action": "follow_me", "params": {}}], "response": "Following you! I'll stay close."}"""


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
        self._vlm_warmed = False

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

    def warm_up_vlm(self):
        """Pre-load VLM into VRAM by sending a tiny image.

        Called on-demand before first VLM query (not at startup) because
        LLM and VLM share VRAM on Jetson — warming VLM would evict LLM.
        """
        if self._vlm_warmed:
            return
        print(f"[Brain] Warming up VLM '{VLM_MODEL}'...")
        t0 = time.time()
        try:
            # 1x1 white JPEG (smallest valid image)
            tiny_jpeg = base64.b64encode(
                b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01'
                b'\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07'
                b'\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13'
                b'\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c'
                b'(7),01444\x1f\'9=82<.342\xff\xc0\x00\x0b\x08\x00\x01\x00'
                b'\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01\x05\x01'
                b'\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x01'
                b'\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10'
                b'\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00'
                b'\x01}\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07"q'
                b'\x142\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n'
                b'\x16\x17\x18\x19\x1a%&\'()*456789:CDEFGHIJSTUVWXYZcdefghij'
                b'stuvwxyz\x83\x84\x85\x86\x87\x88\x89\x8a\x92\x93\x94\x95'
                b'\x96\x97\x98\x99\x9a\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa'
                b'\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xc2\xc3\xc4\xc5\xc6'
                b'\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xe1'
                b'\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4\xf5'
                b'\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?'
                b'\x00\xfb\xd2\x8a(\x03\xff\xd9'
            ).decode()
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": VLM_MODEL,
                    "messages": [{"role": "user", "content": "hi", "images": [tiny_jpeg]}],
                    "stream": False,
                    "keep_alive": "5m",
                    "options": {"num_gpu": 99, "num_predict": 5},
                },
                timeout=FIRST_REQUEST_TIMEOUT,
            )
            elapsed = time.time() - t0
            self._vlm_warmed = True
            if r.status_code == 200:
                print(f"[Brain] VLM warm in {elapsed:.1f}s")
            else:
                print(f"[Brain] VLM warm-up got status {r.status_code}")
        except Exception as e:
            print(f"[Brain] VLM warm-up error: {e}")

    def _build_messages(self, transcript: str, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build the message list for the Ollama chat API."""
        state_lines = "\n".join(f"- {k}: {v}" for k, v in state.items())
        system_content = SYSTEM_PROMPT + f"\n\nCurrent robot state:\n{state_lines}"

        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": transcript})

        return messages

    def process(self, transcript: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user utterance and return actions + response.

        Args:
            transcript: What the user said (ASR output).
            state: Current robot state dict (battery, location, etc.).

        Returns:
            Dict with keys:
                "actions": list of {"intent": str, "params": dict}
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
                        "num_predict": 150,
                        "num_gpu": 99,
                        "num_ctx": 4096,
                    },
                },
                timeout=timeout,
            )
            self._first_request = False

            if r.status_code != 200:
                print(f"[Brain] Ollama error {r.status_code}: {r.text[:200]}")
                return {"actions": [], "response": "", "raw_llm": ""}

            message = r.json().get("message", {})
            content = (message.get("content") or "").strip()
            elapsed = time.time() - t0
            print(f"[Brain] LLM responded in {elapsed:.1f}s ({len(content)} chars)")

            # --- Parse JSON ---
            actions = []
            response = ""

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                print(f"[Brain] Failed to parse JSON: {content[:200]}")
                return {"actions": [], "response": content, "raw_llm": content}

            response = str(data.get("response", "")).strip()

            # Support new "actions" list format
            raw_actions = data.get("actions")
            if isinstance(raw_actions, list):
                for item in raw_actions:
                    if isinstance(item, dict):
                        name = item.get("action")
                        if name and isinstance(name, str):
                            params = item.get("params", {})
                            if not isinstance(params, dict):
                                params = {}
                            params = self._normalize_params(name, params)
                            actions.append({"intent": name, "params": params})

            # Backward compat: old single "action" field
            if not actions:
                action_name = data.get("action")
                if action_name and isinstance(action_name, str) and action_name.lower() != "null":
                    params = data.get("params", {})
                    if not isinstance(params, dict):
                        params = {}
                    params = self._normalize_params(action_name, params)
                    actions.append({"intent": action_name, "params": params})

            if actions:
                names = " -> ".join(a["intent"] for a in actions)
                print(f"[Brain] Actions: {names}")
            else:
                print("[Brain] No action (conversation only)")

            # Update conversation history
            self.history.append({"role": "user", "content": transcript})
            self.history.append({"role": "assistant", "content": content})

            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]

            return {
                "actions": actions,
                "response": response,
                "raw_llm": content,
            }

        except requests.Timeout:
            print(f"[Brain] Timeout after {timeout}s — model may be loading")
            return {"actions": [], "response": "", "raw_llm": ""}
        except requests.ConnectionError:
            print("[Brain] Cannot connect to Ollama. Is it running?")
            self._available = False
            return {"actions": [], "response": "", "raw_llm": ""}
        except Exception as e:
            print(f"[Brain] Error: {e}")
            return {"actions": [], "response": "", "raw_llm": ""}

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

    def query_vlm(self, image_bytes: bytes, question: str, yolo_hint: str = "") -> str:
        """Send image + question to the VLM for visual description.

        Args:
            image_bytes: JPEG image data from Spot camera.
            question: The user's original question (e.g. "what do you see?").
            yolo_hint: Optional YOLO detection result to guide VLM.

        Returns:
            VLM's text response describing the image.
        """
        # On-demand VLM warm-up (first call only)
        if not self._vlm_warmed:
            self.warm_up_vlm()

        image_b64 = base64.b64encode(image_bytes).decode()

        vlm_prompt = (
            f"You are Spot, a Boston Dynamics robot at Dartmouth College. "
            f"This image is what you see right now through your own camera eyes. "
            f"A user asked: \"{question}\". "
        )
        if yolo_hint:
            vlm_prompt += f"{yolo_hint} "
        vlm_prompt += (
            "Respond naturally in first person as if you are looking around, "
            "NOT as if you are analyzing a photograph. Never mention 'image', "
            "'photo', 'picture', 'angle', or 'vantage point'. "
            "Describe what you see in 2-3 sentences. "
            "Be specific about objects, people, and surroundings."
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
        if result["actions"]:
            for i, a in enumerate(result["actions"]):
                print(f"  -> Action {i+1}: {a['intent']}({a.get('params', {})})")
        else:
            print("  -> (no action)")
        print()
