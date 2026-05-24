"""LLM Brain for Spot — conversational robot control via single GBNF grammar.

Architecture (Stage 2A):
- Backend: llama.cpp (default) via SPOT_BRAIN_BACKEND=llamacpp, Ollama fallback.
- Single grammar (src/voice_control/grammar/spot_action.gbnf) on every turn.
- Output shape: {"actions": [...], "response": "..."}.
  Action utterances populate actions[]; freeform utterances leave actions=[].
- No regex intent router. Model decides per turn under grammar.
- Streaming: set self.on_token_callback to receive token deltas (T11 wiring).

Architecture inspired by Boston Dynamics' Robots That Can Chat.
"""

import json
import re
import time
import base64
import collections
import requests
from pathlib import Path
from typing import Optional, Dict, Any, List

from src.voice_control.brain import get_backend
from src.voice_control.chatbot.session_state import SessionState
from src.voice_control.chatbot.persona import (
    Persona, load_registry, get_persona, default_persona_name,
    PersonaRegistryError,
)

_BACKEND = None


def _backend():
    global _BACKEND
    if _BACKEND is None:
        _BACKEND = get_backend()
    return _BACKEND


# Cheap regex to pick the sampling profile. Hint only — GBNF still enforces
# the {actions, response} shape regardless of the profile chosen. If the hint
# is wrong, output is still valid; the temperature/top_p just may be
# suboptimal for that turn.
_ACTION_HINT_RE = re.compile(
    r"\b(stand|sit|walk|turn|go|come|follow|stop|freeze|estop|strafe|"
    r"tour|patrol|save|load|list|find|describe|check|battery|status|"
    r"power|height|speed|volume|self ?right)\b",
    re.IGNORECASE,
)


def _sampling_hint(transcript: str) -> str:
    return "action" if _ACTION_HINT_RE.search(transcript) else "freeform"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DEFAULT_MODEL = "gemma4:e4b"
VLM_MODEL = "gemma4:e4b"
OLLAMA_URL = "http://localhost:11434"
MAX_HISTORY = 24          # messages (12 user + 12 assistant exchanges) — 2E.1 bump
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
- set_volume: Set your speaking volume. Params: {"level": <1-100>} as a percentage where 100=full, 50=half, 10=quiet. For "louder"/"quieter" without a number, pick a sensible delta from current (e.g. +20).
- go_to: Navigate to saved location. Params: {"location": "<name>"}. Use lowercase_with_underscores.
- tour: Visit locations in sequence (one pass). Params: {"locations": ["loc1", "loc2"]} for specific stops, or {"locations": "all"} to visit all. Use for "loop the map", "visit everywhere".
- patrol: Loop through locations continuously until stopped. Params: same as tour. Use for "patrol", "keep looping", "keep patrolling".
- come_back: Return to position before last navigation. No params. Use for "come back", "go home", "return".
- save_location: Save current position. Params: {"location": "<name>"}.
- list_locations: List saved locations. No params.
- load_map: Switch to a different GraphNav map. Params: {"map": "<name>"} (optional — omit to reload the last-used map). The available maps are listed in the robot state under "available_maps". Loading a map disrupts any in-progress navigation and the robot will need to re-localize (best with a fiducial visible).
- list_maps: List the GraphNav maps available on disk. No params.
- go_to_object: Walk toward any visible object Spot can recognise. Params: {"description": "<what to find>"}. Spot uses YOLO/VLM to identify the target in its surroundings — anything you can describe in a few words. Works for ordinary objects (chairs, tables, posters, backpacks, people), architectural features (doors, exits, doorways, windows), and pick-up targets ("the cup on the floor"). Only for things you can SEE in the camera right now — use go_to for saved map locations.
- follow_me: Follow the nearest person, maintaining distance. No params. Use for "follow me", "come with me", "tag along".
- describe: Take a photo and describe what you see. Params: {"camera": "front"|"left"|"right"|"back", "query": "<specific object to look for, if any>"}. Default camera "front". Omit query for general "what do you see" questions.
- check_obstacles: Query Spot's built-in obstacle map for nearby obstacles. No params. Use for "is there anything in front of you", "is this room cluttered", "what direction is clearest", "can you move forward safely". Returns distances and bearings without using cameras or YOLO — it's a direct read of Spot's footstep-planning grid.
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
{"actions": [{"action": "go_to_object", "params": {"description": "red chair"}}], "response": "Looking for the red chair!"}

User: "Walk to the door"
{"actions": [{"action": "go_to_object", "params": {"description": "door"}}], "response": "Looking for a door."}

User: "Find me a chair"
{"actions": [{"action": "go_to_object", "params": {"description": "chair"}}], "response": "On it — finding a chair."}

User: "Go to the poster on the wall"
{"actions": [{"action": "go_to_object", "params": {"description": "poster"}}], "response": "Heading to the poster."}

User: "Is there anything in front of you?"
{"actions": [{"action": "check_obstacles", "params": {}}], "response": "Let me check."}

User: "Is this room cluttered?"
{"actions": [{"action": "check_obstacles", "params": {}}], "response": "Checking my obstacle map."}

User: "Set your volume to 80 percent"
{"actions": [{"action": "set_volume", "params": {"level": 80}}], "response": "Setting my volume to 80%."}

User: "A bit louder please"
{"actions": [{"action": "set_volume", "params": {"level": 90}}], "response": "Speaking up!"}"""


# ---------------------------------------------------------------------------
# Knowledge packs — appended to the system prompt at import time.
#
# Currently always-on: every LLM call carries the full knowledge pack(s).
# Combined size is ~12K tokens (~55 KB of markdown), which fits comfortably
# in qwen2.5:7b's 32K context with plenty of headroom for history and
# response. Always-on removes any need for a "did the user mention the
# tour?" trigger heuristic, which is the right call for the open-house
# demo where any visitor question may end up being tour- or Thayer-relevant.
#
# Order matters for KV-cache prefix sharing: list the more stable / more
# frequently consulted pack first.
#
# Future improvements (deliberately not implemented now):
#   1. Gate injection on intent — only attach the relevant pack when the
#      user mentions tour/Thayer/a known stop name, to keep the default
#      prompt smaller. Needs a trigger heuristic that doesn't miss; not
#      worth the risk before a live demo.
#   2. Add a dedicated `start_tour` dispatcher action that walks the full
#      Thayer route waypoint list with narration at each stop, instead of
#      relying on the LLM to chain the existing `tour` action with talking
#      points. Cleaner UX, but a real new dispatch path — defer until after
#      the demo and validate behind the existing `tour` action first.
# ---------------------------------------------------------------------------
_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_KNOWLEDGE_FILES = ["thayer_knowledge.md", "tour_route.md"]


def _load_knowledge_packs() -> str:
    chunks = []
    for name in _KNOWLEDGE_FILES:
        path = _KNOWLEDGE_DIR / name
        try:
            text = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"[Brain] Knowledge pack not found, skipping: {path}")
            continue
        except Exception as e:
            print(f"[Brain] Failed to load knowledge pack {path}: {e}")
            continue
        chunks.append(f"=== KNOWLEDGE PACK: {name} ===\n{text}")
    if not chunks:
        return ""
    return "\n\n".join(chunks)


_KNOWLEDGE_TEXT = _load_knowledge_packs()
if _KNOWLEDGE_TEXT:
    SYSTEM_PROMPT = (
        SYSTEM_PROMPT
        + "\n\nYou also have the following reference knowledge. Use it to "
          "answer questions and stay on-script when relevant. Do not invent "
          "facts that aren't in here; if asked something not covered, say so.\n\n"
        + _KNOWLEDGE_TEXT
    )


class LLMBrain:
    """Conversational LLM brain for Spot robot using Ollama structured JSON.

    The model receives the action catalog in the system prompt and responds
    with a JSON object containing action + response. Ollama enforces valid
    JSON at the grammar level, so parsing is reliable.
    """

    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL):
        self.model = model
        self.ollama_url = ollama_url
        # Bounded conversation history: deque drops the oldest message
        # automatically when full, replacing the older list-slice pattern
        # that re-allocated `self.history` on every turn.
        self.history: "collections.deque[Dict[str, Any]]" = collections.deque(maxlen=MAX_HISTORY)
        self._available = None  # cached availability check
        self._first_request = True

        # Stage 2E.1: persona registry + cross-turn session state.
        try:
            self._persona_registry = load_registry()
        except PersonaRegistryError as e:
            print(f"[Brain] WARN: persona registry load failed: {e}; using empty registry")
            self._persona_registry = {}
        self.session_state = SessionState(current_persona=default_persona_name())

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
        except Exception as e:
            # Surface the cause once so the user has a starting point
            # (e.g. "Connection refused" → ollama isn't running, or
            # "Name or service not known" → wrong host).
            print(f"[Brain] Ollama check failed: {type(e).__name__}: {e}")
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
                    # No "format": "json" here — warm-up discards response, and combining
                    # it with "think": False triggers Ollama bug #15260 (json constraint
                    # silently dropped). Plain text gen is fine for KV-cache priming.
                    "think": False,
                    "stream": False,
                    "keep_alive": -1,
                    "options": {"num_predict": 10, "num_gpu": 99, "num_ctx": 32768},
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
        """No-op since LLM and VLM are now the same model (gemma4:e4b).
        Kept for API compatibility with callers that still invoke it.
        """
        return

    def _build_messages(self, transcript: str, state: Dict[str, Any]) -> List[Dict[str, str]]:
        """Build message list. State is the live robot snapshot from
        get_robot_state_dict(); session state is owned by self.session_state.
        Persona prefix prepended BEFORE SYSTEM_PROMPT.
        """
        persona = get_persona(self.session_state.current_persona, self._persona_registry)
        merged = {**state, **self.session_state.as_dict()}
        state_lines = "\n".join(f"- {k}: {v}" for k, v in merged.items())
        system_content = (
            f"{persona.prompt_prefix}\n\n"
            f"{SYSTEM_PROMPT}\n\n"
            f"Current robot state:\n{state_lines}"
        )
        messages = [{"role": "system", "content": system_content}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": transcript})
        return messages

    def process(self, transcript: str, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Single-path brain call. Always returns {actions, response, raw_llm}.

        If on_token_callback is set on self, tokens stream as they arrive.
        Grammar emits each action with key "action"; this method renames to
        "intent" so spot_dispatch.dispatch_intent() can consume directly.
        """
        if state is None:
            state = {}

        messages = self._build_messages(transcript, state)
        system = messages[0]["content"]
        user_history = messages[1:]

        profile = _sampling_hint(transcript)
        backend = _backend()
        persona = get_persona(self.session_state.current_persona, self._persona_registry)
        overrides = persona.sampling_overrides.get(profile) or None
        t0 = time.time()
        raw = backend.chat(
            system,
            user_history,
            on_token=getattr(self, "on_token_callback", None),
            profile=profile,
            sampling_overrides=overrides,
        )
        elapsed_ms = int((time.time() - t0) * 1000)

        try:
            parsed = json.loads(raw)
            grammar_actions = parsed.get("actions", []) or []
            response = parsed.get("response", "").strip()
        except json.JSONDecodeError:
            # Grammar guarantees valid JSON — should never hit. Defensive.
            grammar_actions = []
            response = raw.strip()
            parsed = {"actions": [], "response": response}

        # Grammar emits {"action": "stand", ...}; dispatcher expects {"intent": "stand", ...}.
        actions = [
            {"intent": a["action"], "params": a.get("params", {})}
            for a in grammar_actions
            if isinstance(a, dict) and a.get("action")
        ]

        # For describe actions the LLM "response" is often hallucinated
        # (model can't see the camera until the dispatcher runs the VLM
        # call), so sanitize the stored assistant turn — otherwise the
        # fake description sticks for MAX_HISTORY turns and biases later
        # replies. Dispatcher handles the spoken side independently.
        self.history.append({"role": "user", "content": transcript})
        if any(a["intent"] == "describe" for a in actions):
            parsed["response"] = "Taking a look..."
            sanitized = json.dumps(parsed)
            self.history.append({"role": "assistant", "content": sanitized})
        else:
            self.history.append({"role": "assistant", "content": raw})

        self.session_state.turn_index += 1

        print(
            f"[Brain-timing] backend={backend.name} profile={profile} "
            f"actions={len(actions)} response_chars={len(response)} "
            f"total_ms={elapsed_ms}"
        )
        return {"actions": actions, "response": response, "raw_llm": raw}

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
                    "keep_alive": -1,
                    "think": False,  # disable chain-of-thought for VLM (gemma4 puts output in thinking otherwise)
                    "options": {"num_gpu": 99, "num_predict": 200},
                },
                timeout=VLM_TIMEOUT,
            )
            elapsed = time.time() - t0

            if r.status_code != 200:
                print(f"[Brain] VLM error {r.status_code}: {r.text[:200]}")
                return "Sorry, I couldn't process the image right now."

            _resp_json = r.json()
            content = _resp_json.get("message", {}).get("content", "").strip()
            print(f"[Brain] VLM responded in {elapsed:.1f}s")

            # --- Ollama duration breakdown (Stage 2 latency instrumentation) ---
            _total_ms   = _resp_json.get("total_duration",       0) // 1_000_000
            _load_ms    = _resp_json.get("load_duration",         0) // 1_000_000
            _pe_ms      = _resp_json.get("prompt_eval_duration",  0) // 1_000_000
            _pe_tok     = _resp_json.get("prompt_eval_count",     0)
            _eval_ms    = _resp_json.get("eval_duration",         0) // 1_000_000
            _eval_tok   = _resp_json.get("eval_count",            0)
            _tps = _eval_tok / (_eval_ms / 1000) if _eval_ms > 0 else 0.0
            print(
                f"[Brain-timing] path=vlm model={VLM_MODEL} "
                f"total={_total_ms}ms load={_load_ms}ms "
                f"prompt_eval={_pe_ms}ms ({_pe_tok} tok) "
                f"eval={_eval_ms}ms ({_eval_tok} tok @ {_tps:.1f} tok/s)"
            )

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
_brain: Optional[LLMBrain] = None


def get_brain(model: str = DEFAULT_MODEL) -> LLMBrain:
    """Get or create the singleton LLMBrain instance."""
    global _brain
    if _brain is None or _brain.model != model:
        _brain = LLMBrain(model=model)
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
    brain = LLMBrain(model=model)

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
