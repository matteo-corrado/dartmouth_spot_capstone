"""Local LLM-based intent parser using Ollama for natural language understanding.

This module provides robust intent extraction from natural language using local LLMs.
Optimized for Jetson AGX Orin with fallback to regex patterns.

Installation on Jetson:
    # Install Ollama
    curl -fsSL https://ollama.com/install.sh | sh

    # Pull recommended model (choose one based on your needs)
    ollama pull qwen2.5:3b          # RECOMMENDED: Fast, accurate, 2GB
    ollama pull phi3.5:latest       # Alternative: 2.4GB, very good for instructions
    ollama pull llama3.2:3b         # Alternative: 2GB, good general purpose

Performance comparison on Jetson AGX Orin:
    - qwen2.5:3b      : ~150-200ms inference, best accuracy for instructions
    - phi3.5:latest   : ~180-250ms inference, excellent instruction following
    - llama3.2:3b     : ~200-300ms inference, good general purpose
"""

import json
import requests
from typing import Optional, Dict, Any

# Default model - change based on what you have installed
DEFAULT_MODEL = "qwen2.5:3b"  # Best balance of speed and accuracy for Jetson

# Intent schema for the robot
INTENT_SCHEMA = {
    # Safety
    "stop": "Stop all movement immediately",
    "freeze": "Freeze in place and hold position",
    "estop": "Emergency stop",

    # Posture
    "stand": "Stand up from sitting position",
    "sit": "Sit down",
    "selfright": "Recover from a fall, self-right",

    # Body height
    "body_height": "Adjust body height - params: {height: number} (-0.15=crouch, 0=normal, 0.1=tall)",

    # Movement
    "walk": "Walk forward or backward - params: {direction: 'forward'|'backward', distance: number in meters}",
    "strafe": "Move sideways - params: {direction: 'left'|'right', distance: number in meters}",
    "turn": "Rotate in place - params: {deg: number 0-360, dir: 'left'|'right'}",

    # Speed
    "set_speed": "Set movement speed - params: {speed: 'slow'|'normal'|'fast'}",

    # Navigation
    "walk_to": "Walk to relative position - params: {relative: [x, y, yaw]}",
    "go_to": "Navigate to named location - params: {location: string}",
    "save_location": "Save current position with a name - params: {location: string}",
    "list_locations": "List all saved locations",

    # Status
    "battery_status": "Check battery level and runtime",
    "status": "Get overall robot status",
    "power_off": "Safely power off the robot",

    # Other
    "follow": "Follow a person (not yet implemented)",
    "ptz_aim": "Aim camera at target - params: {target: 'speaker'}"
}

# Optimized prompt for small models
SYSTEM_PROMPT = f"""You are a command parser for a Boston Dynamics Spot robot. Extract the intent and parameters from voice commands.

Available commands:
{json.dumps(INTENT_SCHEMA, indent=2)}

Return ONLY valid JSON in this exact format:
{{"intent": "command_name", "params": {{}}, "confidence": 0.95}}

If unclear, return:
{{"intent": null, "confidence": 0.0}}

Examples:
Input: "turn left 45 degrees"
Output: {{"intent": "turn", "params": {{"deg": 45, "dir": "left"}}, "confidence": 0.95}}

Input: "could you rotate counterclockwise about 45"
Output: {{"intent": "turn", "params": {{"deg": 45, "dir": "left"}}, "confidence": 0.85}}

Input: "walk forward 2 meters"
Output: {{"intent": "walk", "params": {{"direction": "forward", "distance": 2.0}}, "confidence": 0.95}}

Input: "move back a bit"
Output: {{"intent": "walk", "params": {{"direction": "backward", "distance": 1.0}}, "confidence": 0.8}}

Input: "strafe to the left"
Output: {{"intent": "strafe", "params": {{"direction": "left", "distance": 0.5}}, "confidence": 0.85}}

Input: "go to the kitchen"
Output: {{"intent": "go_to", "params": {{"location": "kitchen"}}, "confidence": 0.9}}

Input: "remember this spot as home base"
Output: {{"intent": "save_location", "params": {{"location": "home_base"}}, "confidence": 0.9}}

Input: "please stop now"
Output: {{"intent": "stop", "params": {{}}, "confidence": 0.95}}

Input: "how much battery do you have"
Output: {{"intent": "battery_status", "params": {{}}, "confidence": 0.9}}

Input: "crouch down"
Output: {{"intent": "body_height", "params": {{"height": -0.15}}, "confidence": 0.9}}

Input: "turn around"
Output: {{"intent": "turn", "params": {{"deg": 180, "dir": "left"}}, "confidence": 0.95}}

Input: "what's the weather"
Output: {{"intent": null, "confidence": 0.0}}
"""


def parse_intent_llm(
    text: str,
    model: str = DEFAULT_MODEL,
    timeout: float = 3.0,
    temperature: float = 0.1
) -> Optional[Dict[str, Any]]:
    """Parse intent using local Ollama model.

    Args:
        text: Voice command transcript
        model: Ollama model name (default: qwen2.5:3b)
        timeout: Request timeout in seconds (default: 3.0)
        temperature: Generation temperature 0-1, lower = more deterministic (default: 0.1)

    Returns:
        Dict with intent, params, confidence, and raw text, or None if failed
    """
    if not text or not text.strip():
        return None

    # Build prompt
    prompt = f"{SYSTEM_PROMPT}\n\nInput: \"{text}\"\nOutput:"

    try:
        # Call Ollama API (runs on localhost:11434 by default)
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "top_p": 0.9,
                    "top_k": 40,
                    "num_predict": 100,  # Limit tokens for faster response
                }
            },
            timeout=timeout
        )

        if response.status_code != 200:
            print(f"[LLM] Ollama error {response.status_code}: {response.text}")
            return None

        # Parse response
        result_text = response.json()["response"].strip()

        # Extract JSON from response (handle cases where model adds explanation)
        json_start = result_text.find("{")
        json_end = result_text.rfind("}") + 1

        if json_start >= 0 and json_end > json_start:
            json_str = result_text[json_start:json_end]
            result = json.loads(json_str)

            # Validate result structure
            if "intent" in result and "confidence" in result:
                # Only return if we found a valid intent
                if result.get("intent"):
                    # Ensure params exists
                    if "params" not in result:
                        result["params"] = {}

                    # Add raw text
                    result["raw"] = text

                    # Normalize location names (same as regex parser)
                    if "location" in result["params"]:
                        location = str(result["params"]["location"]).strip().lower()
                        location = location.replace(" ", "_")
                        result["params"]["location"] = location

                    # Validate numeric params
                    if "deg" in result["params"]:
                        try:
                            deg = float(result["params"]["deg"])
                            deg = max(0, min(360, deg))  # Clamp to [0, 360]
                            result["params"]["deg"] = int(deg) if deg.is_integer() else deg
                        except (ValueError, TypeError):
                            print(f"[LLM] Invalid degree value: {result['params']['deg']}")
                            return None

                    return result

        # If we got here, parsing failed
        print(f"[LLM] Could not extract valid intent from: {result_text}")
        return None

    except requests.Timeout:
        print(f"[LLM] Timeout after {timeout}s - model may be loading or too slow")
        return None
    except requests.ConnectionError:
        print("[LLM] Cannot connect to Ollama. Is it running? (sudo systemctl start ollama)")
        return None
    except json.JSONDecodeError as e:
        print(f"[LLM] JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return None


def test_ollama_connection(model: str = DEFAULT_MODEL) -> bool:
    """Test if Ollama is running and model is available.

    Returns:
        True if ready, False otherwise
    """
    try:
        # Check if Ollama is running
        response = requests.get("http://localhost:11434/api/version", timeout=2)
        if response.status_code != 200:
            return False

        # Check if model is available
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m["name"] for m in models]
            if model in model_names:
                return True
            else:
                print(f"[LLM] Model '{model}' not found. Available: {model_names}")
                print(f"[LLM] Run: ollama pull {model}")
                return False
        return False
    except:
        return False


if __name__ == "__main__":
    """Test the intent parser."""
    print("Testing LLM Intent Parser")
    print("=" * 60)

    # Check Ollama connection
    if not test_ollama_connection():
        print("\n❌ Ollama is not running or model not available!")
        print("\nSetup instructions:")
        print("1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print(f"2. Pull model: ollama pull {DEFAULT_MODEL}")
        print("3. Start service: sudo systemctl start ollama")
        exit(1)

    print("✓ Ollama is ready\n")

    # Test cases - natural language variations
    test_cases = [
        "turn left 45 degrees",
        "could you rotate counterclockwise about 45",
        "please stop right now",
        "go to the kitchen",
        "navigate to home base",
        "remember this location as the front door",
        "stand up",
        "walk forward 2 meters",
        "move back a little bit",
        "strafe to the left",
        "how's the battery doing",
        "crouch down",
        "turn around",
        "what's the weather like?",
        "step right",
        "go slow",
    ]

    for i, text in enumerate(test_cases, 1):
        print(f"\n[{i}/{len(test_cases)}] Input: \"{text}\"")
        result = parse_intent_llm(text)

        if result:
            print(f"   ✓ Intent: {result['intent']}")
            if result['params']:
                print(f"     Params: {result['params']}")
            print(f"     Confidence: {result['confidence']:.0%}")
        else:
            print("   ✗ No intent recognized")

    print("\n" + "=" * 60)
