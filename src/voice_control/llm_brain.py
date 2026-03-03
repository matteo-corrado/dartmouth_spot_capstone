"""LLM Brain for Spot — conversational robot control via Dartmouth cloud APIs.

Uses ChatDartmouth (langchain-dartmouth) for LLM command parsing and VLM scene
description.  The LLM (on-prem meta.llama-3-1-8b-instruct, free/unlimited)
receives the action catalog in the system prompt and returns a JSON object
containing action(s) + spoken response.  The VLM (cloud
openai.gpt-4.1-mini-2025-04-14) handles multimodal "describe" queries.

Replaces the former Ollama-based inference (qwen2.5:7b / qwen2.5vl:7b).
"""

import json
import time
import base64
from typing import Optional, Dict, Any, List

from langchain_dartmouth.llms import ChatDartmouth
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import (
    LLM_MODEL, VLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS, VLM_MAX_TOKENS,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = LLM_MODEL
MAX_HISTORY = 12          # messages (6 user + 6 assistant exchanges)
REQUEST_TIMEOUT = 30.0    # seconds per request

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
User: "How are you doing?"
{"actions": [], "response": "I'm doing great! Battery is looking good and I'm ready to help."}

User: "Go to kitchen then hallway then lab"
{"actions": [{"action": "tour", "params": {"locations": ["kitchen", "hallway", "lab"]}}], "response": "On my way! I'll visit kitchen, hallway, and lab in order."}

User: "Go to the conference and come back"
{"actions": [{"action": "go_to", "params": {"location": "conference"}}, {"action": "come_back", "params": {}}], "response": "Going to conference and coming right back!"}

User: "Do you see the blue chair?"
{"actions": [{"action": "describe", "params": {"camera": "front", "query": "blue chair"}}], "response": "Let me check for the blue chair..."}

User: "Go to the red chair"
{"actions": [{"action": "go_to_object", "params": {"description": "red chair"}}], "response": "Looking for the red chair!"}"""


class SpotBrain:
    """Conversational LLM brain for Spot robot using Dartmouth cloud APIs.

    The LLM receives the action catalog in the system prompt and responds with
    a JSON object containing action + response.  ChatDartmouth handles auth and
    transport; we parse the returned text as JSON.
    """

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.history: List = []  # LangChain message objects
        self._llm: ChatDartmouth | None = None
        self._vlm: ChatDartmouth | None = None

    # -- lazy LLM / VLM construction (created on first use) ----------------

    def _get_llm(self) -> ChatDartmouth:
        if self._llm is None:
            self._llm = ChatDartmouth(
                model_name=self.model,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
        return self._llm

    def _get_vlm(self) -> ChatDartmouth:
        if self._vlm is None:
            self._vlm = ChatDartmouth(
                model_name=VLM_MODEL,
                max_tokens=VLM_MAX_TOKENS,
            )
        return self._vlm

    def is_available(self) -> bool:
        """Quick connectivity check for the Dartmouth Chat API."""
        try:
            llm = self._get_llm()
            # A lightweight invoke to confirm the model responds
            llm.invoke([HumanMessage(content="ping")])
            return True
        except Exception as e:
            print(f"[Brain] Dartmouth API not reachable: {e}")
            return False

    def warm_up(self):
        """No-op — cloud models do not need local VRAM warm-up."""
        pass

    def warm_up_vlm(self):
        """No-op — cloud models do not need local VRAM warm-up."""
        pass

    def _build_messages(self, transcript: str, state: Dict[str, Any]) -> list:
        """Build the LangChain message list for ChatDartmouth."""
        state_lines = "\n".join(f"- {k}: {v}" for k, v in state.items())
        system_content = SYSTEM_PROMPT + f"\n\nCurrent robot state:\n{state_lines}"

        messages = [SystemMessage(content=system_content)]
        messages.extend(self.history)
        messages.append(HumanMessage(content=transcript))

        return messages

    def process(self, transcript: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a user utterance and return actions + response.

        Args:
            transcript: What the user said (ASR output).
            state: Current robot state dict (battery, location, etc.).

        Returns:
            Dict with keys:
                "actions": list of {"intent": str, "params": dict}
                "response": str  -- what the robot says back
                "raw_llm": str   -- raw LLM output (for debugging)
        """
        if state is None:
            state = {}

        messages = self._build_messages(transcript, state)

        try:
            llm = self._get_llm()

            t0 = time.time()
            result = llm.invoke(messages)
            content = (result.content or "").strip()
            elapsed = time.time() - t0
            print(f"[Brain] LLM responded in {elapsed:.1f}s ({len(content)} chars)")

            # --- Parse JSON ---
            actions = []
            response = ""

            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Model may wrap JSON in markdown code fences -- try to extract
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    try:
                        data = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        print(f"[Brain] Failed to parse JSON: {content[:200]}")
                        return {"actions": [], "response": content, "raw_llm": content}
                else:
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

            # Update conversation history (LangChain message objects)
            self.history.append(HumanMessage(content=transcript))
            self.history.append(AIMessage(content=content))

            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]

            return {
                "actions": actions,
                "response": response,
                "raw_llm": content,
            }

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
        image_b64 = base64.b64encode(image_bytes).decode()

        vlm_system = (
            "You are Spot, a Boston Dynamics robot at Dartmouth College. "
            "This image is what you see right now through your own camera eyes. "
            "Respond naturally in first person as if you are looking around, "
            "NOT as if you are analyzing a photograph. Never mention 'image', "
            "'photo', 'picture', 'angle', or 'vantage point'. "
            "Describe what you see in 2-3 sentences. "
            "Be specific about objects, people, and surroundings."
        )

        user_text = f'A user asked: "{question}".'
        if yolo_hint:
            user_text += f" {yolo_hint}"

        # Multimodal message with image_url (OpenAI-compatible format)
        user_msg = HumanMessage(content=[
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{image_b64}"
            }},
        ])

        try:
            vlm = self._get_vlm()
            print(f"[Brain] Querying VLM ({VLM_MODEL})...")
            t0 = time.time()
            result = vlm.invoke([SystemMessage(content=vlm_system), user_msg])
            elapsed = time.time() - t0

            content = (result.content or "").strip()
            print(f"[Brain] VLM responded in {elapsed:.1f}s")
            return content or "I can see the image but I'm having trouble describing it."

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
    print("Spot LLM Brain — Interactive Test (Dartmouth API)")
    print("=" * 60)

    import sys as _sys
    model = _sys.argv[1] if len(_sys.argv) > 1 else DEFAULT_MODEL
    brain = SpotBrain(model=model)

    if not brain.is_available():
        print(f"\nDartmouth Chat API not reachable for model '{model}'.")
        print("1. Ensure DARTMOUTH_CHAT_API_KEY is set in .env")
        print("2. Ensure you are on campus WiFi")
        _sys.exit(1)

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
