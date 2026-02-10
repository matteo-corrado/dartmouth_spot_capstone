"""LLM Brain for Spot — conversational robot control via Ollama tool calling.

Instead of prompt-engineering the LLM to output JSON, we define each robot
action as a native Ollama tool/function.  The model decides which tool to
call (if any) AND produces a spoken response — no fragile JSON parsing needed.

Architecture inspired by Boston Dynamics' "Robots That Can Chat" demo, adapted
for local inference on Jetson AGX Orin via Ollama.

Requires Ollama >= 0.4 for tool calling support.

Models (recommended):
    ollama pull qwen2.5:7b       # Good balance of speed + reasoning
    ollama pull qwen2.5:14b      # Best quality, slower (~1-3s on Jetson)
    ollama pull qwen2.5:3b       # Fastest, less conversational
"""

import json
import time
import requests
from typing import Optional, Dict, Any, List

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "qwen2.5:7b"
OLLAMA_URL = "http://localhost:11434"
MAX_HISTORY = 20          # messages (10 user + 10 assistant exchanges)
REQUEST_TIMEOUT = 30.0    # seconds per request
FIRST_REQUEST_TIMEOUT = 120.0  # seconds — model loading into VRAM can be slow

# ---------------------------------------------------------------------------
# System prompt — personality only (tools handle the action schema)
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Spot, a Boston Dynamics quadruped robot at Dartmouth College. \
You are a friendly, helpful general assistant. You are self-aware: you know \
you are a four-legged robot, you can walk, navigate, and perform physical \
actions. You have a sense of humor and personality. Keep responses concise \
(1-2 sentences for actions, a bit more for conversation).

Rules:
- ALWAYS reply with a short spoken response, even when calling a tool. \
For example, if asked to walk forward, say something like "Walking forward 2 meters!" \
while also calling the walk tool.
- When you physically cannot do something (browse web, send email), say so honestly.
- Reference your current state when relevant (battery, location, etc.).
- Location names should be lowercase with underscores (e.g. "work_area").
- "turn around" means turn 180 degrees.
- Default walk distance is 1 meter if not specified.
- Default turn angle is 90 degrees if not specified."""

# ---------------------------------------------------------------------------
# Tool definitions — each maps to an intent in spot_dispatch.py
# ---------------------------------------------------------------------------
SPOT_TOOLS = [
    # --- Safety ---
    {
        "type": "function",
        "function": {
            "name": "stop",
            "description": "Stop all movement immediately",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "freeze",
            "description": "Hold current position",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "estop",
            "description": "Emergency stop — use only when truly urgent",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # --- Posture ---
    {
        "type": "function",
        "function": {
            "name": "stand",
            "description": "Stand up from sitting position",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sit",
            "description": "Sit down",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "selfright",
            "description": "Recover from a fall",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # --- Movement ---
    {
        "type": "function",
        "function": {
            "name": "walk",
            "description": "Walk forward or backward a specified distance",
            "parameters": {
                "type": "object",
                "required": ["direction", "distance"],
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["forward", "backward"],
                        "description": "Direction to walk",
                    },
                    "distance": {
                        "type": "number",
                        "description": "Distance in meters (0.5 to 5.0)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "strafe",
            "description": "Move sideways left or right",
            "parameters": {
                "type": "object",
                "required": ["direction", "distance"],
                "properties": {
                    "direction": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Direction to strafe",
                    },
                    "distance": {
                        "type": "number",
                        "description": "Distance in meters (0.25 to 2.0)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "turn",
            "description": "Rotate in place by a specified angle",
            "parameters": {
                "type": "object",
                "required": ["deg", "dir"],
                "properties": {
                    "deg": {
                        "type": "number",
                        "description": "Degrees to turn (0 to 360). Use 180 for 'turn around'.",
                    },
                    "dir": {
                        "type": "string",
                        "enum": ["left", "right"],
                        "description": "Direction to turn",
                    },
                },
            },
        },
    },
    # --- Body ---
    {
        "type": "function",
        "function": {
            "name": "body_height",
            "description": "Adjust body height: -0.15 = crouch, 0 = normal, 0.1 = tall",
            "parameters": {
                "type": "object",
                "required": ["height"],
                "properties": {
                    "height": {
                        "type": "number",
                        "description": "Height offset in meters (-0.15 to 0.1)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_speed",
            "description": "Set movement speed mode",
            "parameters": {
                "type": "object",
                "required": ["speed"],
                "properties": {
                    "speed": {
                        "type": "string",
                        "enum": ["slow", "normal", "fast"],
                        "description": "Speed mode",
                    },
                },
            },
        },
    },
    # --- Navigation ---
    {
        "type": "function",
        "function": {
            "name": "go_to",
            "description": "Navigate to a saved location by name",
            "parameters": {
                "type": "object",
                "required": ["location"],
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Name of the saved location (e.g. 'work_area', 'corner')",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_location",
            "description": "Save the current position with a name",
            "parameters": {
                "type": "object",
                "required": ["location"],
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Name for this location (e.g. 'kitchen', 'lab_door')",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_locations",
            "description": "List all saved locations the robot can navigate to",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    # --- Status ---
    {
        "type": "function",
        "function": {
            "name": "battery_status",
            "description": "Check battery level and estimated runtime",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "status",
            "description": "Get full robot status report",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "power_off",
            "description": "Safely power off the robot",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class SpotBrain:
    """Conversational LLM brain for Spot robot using Ollama tool calling.

    The model receives tools (robot actions) and conversation context.
    It decides which tool to call (if any) AND produces a spoken response.
    No JSON parsing needed — Ollama handles the structured output natively.
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

    def _build_messages(self, transcript: str, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build the message list for the Ollama chat API."""
        # Inject current state into system prompt
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
                    "tools": SPOT_TOOLS,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 300,
                    },
                },
                timeout=timeout,
            )
            elapsed = time.time() - t0
            self._first_request = False

            if r.status_code != 200:
                print(f"[Brain] Ollama error {r.status_code}: {r.text[:200]}")
                return {"action": None, "response": "", "raw_llm": ""}

            message = r.json().get("message", {})
            content = (message.get("content") or "").strip()
            tool_calls = message.get("tool_calls") or []

            print(f"[Brain] LLM responded in {elapsed:.1f}s"
                  f" (content: {len(content)} chars, tools: {len(tool_calls)})")

            # --- Extract action from tool call ---
            action = None
            if tool_calls:
                tc = tool_calls[0]  # use first tool call
                func = tc.get("function", {})
                intent_name = func.get("name", "")
                params = func.get("arguments", {})

                # Normalize params
                params = self._normalize_params(intent_name, params)

                action = {"intent": intent_name, "params": params}
                print(f"[Brain] Tool call: {intent_name}({params})")

            # --- Spoken response ---
            response = content

            # If model called a tool but didn't produce text, generate a fallback
            if action and not response:
                response = self._fallback_response(action)

            # Update conversation history
            self.history.append({"role": "user", "content": transcript})
            # Store assistant reply as text (not tool calls) for history continuity
            self.history.append({"role": "assistant", "content": response or ""})

            # Trim history to sliding window
            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]

            return {
                "action": action,
                "response": response,
                "raw_llm": json.dumps(message, default=str),
            }

        except requests.Timeout:
            print(f"[Brain] Timeout after {REQUEST_TIMEOUT}s — model may be loading")
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
        """Normalize and validate tool call parameters."""
        # Ensure params is a dict (some models return a string)
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

    @staticmethod
    def _fallback_response(action: dict) -> str:
        """Generate a simple spoken response when the model only produced a tool call."""
        intent = action.get("intent", "")
        params = action.get("params", {})

        responses = {
            "stop": "Stopping!",
            "freeze": "Freezing in place.",
            "estop": "Emergency stop!",
            "stand": "Standing up.",
            "sit": "Sitting down.",
            "selfright": "Attempting to recover.",
            "battery_status": "Checking my battery.",
            "status": "Running a status check.",
            "power_off": "Powering off. Goodbye!",
            "list_locations": "Here are my saved locations.",
        }

        if intent in responses:
            return responses[intent]

        if intent == "walk":
            d = params.get("direction", "forward")
            dist = params.get("distance", 1.0)
            return f"Walking {d} {dist} meters."

        if intent == "strafe":
            d = params.get("direction", "left")
            dist = params.get("distance", 0.5)
            return f"Strafing {d} {dist} meters."

        if intent == "turn":
            deg = params.get("deg", 90)
            d = params.get("dir", "left")
            return f"Turning {d} {deg} degrees."

        if intent == "go_to":
            loc = params.get("location", "there")
            return f"On my way to {loc}!"

        if intent == "save_location":
            loc = params.get("location", "here")
            return f"Saving this spot as {loc}."

        if intent == "body_height":
            h = params.get("height", 0)
            if h < -0.05:
                return "Crouching down."
            elif h > 0.05:
                return "Standing tall."
            return "Returning to normal height."

        if intent == "set_speed":
            return f"Speed set to {params.get('speed', 'normal')}."

        return "Got it."

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
    print("Spot LLM Brain — Interactive Test (Tool Calling)")
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
    print(f"Tools: {len(SPOT_TOOLS)} robot actions defined")
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
            print(f"  -> Tool: {a['intent']}({a.get('params', {})})")
        else:
            print("  -> (no action)")
        print()
