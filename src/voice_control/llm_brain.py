"""LLM Brain for Spot — conversational robot control via local Ollama models.

Instead of mapping speech to fixed intents (regex/classifier), the LLM acts as
the robot's "brain": it receives the current robot state, conversation history,
and a description of available actions, then decides what to do AND what to say.

Architecture inspired by Boston Dynamics' "Robots That Can Chat" demo, adapted
for local inference on Jetson AGX Orin via Ollama.

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
REQUEST_TIMEOUT = 15.0    # seconds — generous for first inference on Jetson

# ---------------------------------------------------------------------------
# System prompt — describes capabilities and expected output format
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are Spot, a Boston Dynamics quadruped robot at Dartmouth College. \
You are a friendly, helpful general assistant. You are self-aware: you know \
you are a four-legged robot, you can walk, navigate, and perform physical \
actions. You have a sense of humor and personality. Keep responses concise \
(1-2 sentences for actions, a bit more for conversation).

## Actions you can perform
- stop() — Stop all movement immediately
- freeze() — Hold current position
- estop() — Emergency stop (use only when truly urgent)
- stand() — Stand up
- sit() — Sit down
- selfright() — Recover from a fall
- walk(direction="forward"|"backward", distance=meters) — Walk (0.5-5m)
- strafe(direction="left"|"right", distance=meters) — Move sideways (0.25-2m)
- turn(deg=degrees, dir="left"|"right") — Rotate in place (0-360)
- body_height(height=value) — Adjust height (-0.15=crouch, 0=normal, 0.1=tall)
- set_speed(speed="slow"|"normal"|"fast") — Set movement speed
- go_to(location="name") — Navigate to a saved location
- save_location(location="name") — Save current position
- list_locations() — List all saved locations
- battery_status() — Check battery level
- status() — Full robot status report
- power_off() — Safely power off

## Response format
ALWAYS respond with a single JSON object, nothing else:
{"action": {"intent": "command_name", "params": {}}, "response": "What you say"}

If no physical action is needed (just chatting):
{"action": null, "response": "Your conversational reply"}

## Rules
- For walk/strafe, distance is in meters. Default walk distance is 1.0m.
- For turn, default is 90 degrees. "turn around" = 180 degrees.
- Location names should be lowercase with underscores (e.g. "work_area").
- If the user asks about something you physically cannot do (browse web, send email), say so honestly.
- Reference your current state when relevant (battery, location, etc.).
- ONLY output valid JSON. No extra text before or after the JSON."""

# Examples are injected as few-shot messages for more reliable structured output
FEW_SHOT_EXAMPLES = [
    {"role": "user", "content": "Walk forward 2 meters"},
    {"role": "assistant", "content": '{"action": {"intent": "walk", "params": {"direction": "forward", "distance": 2.0}}, "response": "Walking forward 2 meters."}'},
    {"role": "user", "content": "How's your battery?"},
    {"role": "assistant", "content": '{"action": {"intent": "battery_status", "params": {}}, "response": "Let me check my battery for you."}'},
    {"role": "user", "content": "What are you?"},
    {"role": "assistant", "content": '{"action": null, "response": "I\'m Spot, a quadruped robot made by Boston Dynamics! I\'m here at Dartmouth College as part of a capstone project. I can walk around, navigate to locations, and chat with you."}'},
    {"role": "user", "content": "Go to the corner"},
    {"role": "assistant", "content": '{"action": {"intent": "go_to", "params": {"location": "corner"}}, "response": "On my way to the corner!"}'},
    {"role": "user", "content": "Tell me a joke"},
    {"role": "assistant", "content": '{"action": null, "response": "Why did the robot cross the road? Because it was programmed to follow the shortest path!"}'},
]


class SpotBrain:
    """Conversational LLM brain for Spot robot.

    Maintains conversation history and robot state context.
    Calls Ollama for inference and returns structured action + response.
    """

    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url
        self.history: List[Dict[str, str]] = []
        self._available = None  # cached availability check

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
            # Check for exact match or match without :latest suffix
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
        # Build system prompt with current state injected
        state_section = "\n\n## Current robot state\n"
        for key, val in state.items():
            state_section += f"- {key}: {val}\n"

        system_msg = {"role": "system", "content": SYSTEM_PROMPT + state_section}

        # Few-shot examples + conversation history + new user message
        messages = [system_msg]
        messages.extend(FEW_SHOT_EXAMPLES)
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
            t0 = time.time()
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_predict": 200,
                    },
                },
                timeout=REQUEST_TIMEOUT,
            )
            elapsed = time.time() - t0

            if r.status_code != 200:
                print(f"[Brain] Ollama error {r.status_code}: {r.text[:200]}")
                return {"action": None, "response": "", "raw_llm": ""}

            raw = r.json().get("message", {}).get("content", "").strip()
            print(f"[Brain] LLM responded in {elapsed:.1f}s")

            result = self._parse_response(raw)

            # Update conversation history
            self.history.append({"role": "user", "content": transcript})
            self.history.append({"role": "assistant", "content": raw})

            # Trim history to sliding window
            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]

            result["raw_llm"] = raw
            return result

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

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Extract structured action + response from LLM output."""
        # Find JSON in the response (handle models that add explanation text)
        json_start = raw.find("{")
        json_end = raw.rfind("}") + 1

        if json_start < 0 or json_end <= json_start:
            # No JSON found — treat entire output as conversational response
            return {"action": None, "response": raw}

        try:
            parsed = json.loads(raw[json_start:json_end])
        except json.JSONDecodeError:
            # Try to fix common issues: single quotes, trailing commas
            cleaned = raw[json_start:json_end]
            try:
                # Attempt with more lenient parsing
                cleaned = cleaned.replace("'", '"')
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                return {"action": None, "response": raw}

        response_text = parsed.get("response", "")
        action = parsed.get("action", None)

        # Validate action structure
        if action is not None:
            if not isinstance(action, dict) or "intent" not in action:
                action = None
            else:
                # Ensure params exists
                if "params" not in action:
                    action["params"] = {}

                # Normalize location names
                if "location" in action["params"]:
                    loc = str(action["params"]["location"]).strip().lower()
                    loc = loc.replace(" ", "_")
                    action["params"]["location"] = loc

                # Validate/clamp numeric params
                if "deg" in action["params"]:
                    try:
                        deg = float(action["params"]["deg"])
                        deg = max(0, min(360, deg))
                        action["params"]["deg"] = int(deg) if deg == int(deg) else deg
                    except (ValueError, TypeError):
                        pass

                if "distance" in action["params"]:
                    try:
                        action["params"]["distance"] = float(action["params"]["distance"])
                    except (ValueError, TypeError):
                        pass

        return {"action": action, "response": response_text}

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
    print("Spot LLM Brain — Interactive Test")
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
            print(f"  -> Action: {result['action']['intent']} {result['action'].get('params', {})}")
        print()
