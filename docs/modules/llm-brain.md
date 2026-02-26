# llm_brain.py

`src/voice_control/llm_brain.py` -- LLM decision engine (Ollama structured JSON).

Takes a user transcript and robot state, sends them to a local Ollama model,
and receives back a structured JSON object containing actions to execute and
a spoken response. Uses `format="json"` in the Ollama API to guarantee valid
JSON output at the grammar level.

## Architecture

The system prompt includes a full **action catalog** listing every command Spot
can execute (stop, walk, go_to, describe, follow_me, etc.) with parameter
schemas and usage rules. The model fills in a JSON object:

```json
{"actions": [{"action": "walk", "params": {"direction": "forward", "distance": 2.0}}],
 "response": "Walking forward two meters!"}
```

For conversation only (no physical action), `actions` is an empty list.
For chained commands ("go to kitchen then sit"), `actions` contains multiple
entries executed sequentially.

**History management:** The last `MAX_HISTORY` (12) messages are kept in
`self.history` and prepended to each request. The system prompt is rebuilt
each call with fresh robot state.

**Model sharing:** The text LLM (`qwen2.5:7b`) and VLM (`qwen2.5vl:7b`)
share VRAM on Jetson. Ollama swaps them in/out automatically. VLM warm-up
is deferred to first use to avoid evicting the text LLM.

## Key Class

### `SpotBrain(model, ollama_url)`

```python
class SpotBrain:
    def __init__(self, model: str = DEFAULT_MODEL, ollama_url: str = OLLAMA_URL)
```

#### `is_available() -> bool`

Check if Ollama is running and the specified model is pulled. Caches result.

#### `warm_up()`

Send a trivial prompt with the full system prompt to pre-load the model into
VRAM and cache the KV state. Makes the first real command fast (~0.2s prompt
eval instead of ~1.5s cold). Called at startup.

#### `warm_up_vlm()`

Pre-load the VLM by sending a tiny 1x1 JPEG. Called on-demand before the
first `query_vlm()` call, not at startup (would evict the text LLM).

#### `process(transcript, state) -> dict`

Main inference method.

```python
def process(self, transcript: str, state: dict | None = None) -> dict
```

**Args:**
- `transcript` -- User's speech (ASR output).
- `state` -- Current robot state dict (battery, location, etc.).

**Returns:**
```python
{
    "actions": [{"intent": str, "params": dict}, ...],
    "response": str,   # What the robot says back
    "raw_llm": str     # Raw LLM output for debugging
}
```

Ollama options: `temperature=0.3`, `top_p=0.9`, `num_predict=150`, `num_ctx=4096`.

#### `query_vlm(image_bytes, question, yolo_hint) -> str`

Send JPEG image + question to the VLM for visual description.

```python
def query_vlm(self, image_bytes: bytes, question: str, yolo_hint: str = "") -> str
```

The VLM prompt instructs the model to speak in first person ("what I see")
and never reference "image", "photo", or "angle". An optional `yolo_hint`
from YOLO-World detection results guides the VLM's attention.

#### `clear_history()`

Clear conversation history.

### Module-Level Convenience

```python
def get_brain(model: str = DEFAULT_MODEL) -> SpotBrain   # Singleton
def process_with_brain(transcript, state, model) -> dict  # One-shot
```

## Configuration Constants

| Name | Default | Description |
|------|---------|-------------|
| `DEFAULT_MODEL` | `"qwen2.5:7b"` | Text LLM model name |
| `VLM_MODEL` | `"qwen2.5vl:7b"` | Vision-language model name |
| `OLLAMA_URL` | `"http://localhost:11434"` | Ollama API base URL |
| `MAX_HISTORY` | 12 | Max conversation messages kept (6 exchanges) |
| `REQUEST_TIMEOUT` | 30.0 | Seconds per normal request |
| `FIRST_REQUEST_TIMEOUT` | 120.0 | Seconds for first request (model loading) |
| `VLM_TIMEOUT` | 60.0 | Seconds for VLM inference |

## Usage Example

```python
from llm_brain import SpotBrain

brain = SpotBrain(model="qwen2.5:7b")
if brain.is_available():
    brain.warm_up()

    state = {"battery_percent": 85, "is_standing": True,
             "saved_locations": "kitchen, lab"}

    result = brain.process("go to the kitchen and sit down", state)
    # result["actions"] = [
    #     {"intent": "go_to", "params": {"location": "kitchen"}},
    #     {"intent": "sit", "params": {}}
    # ]
    # result["response"] = "Heading to the kitchen, then I'll sit down!"
```

```bash
# Interactive CLI test
python src/voice_control/llm_brain.py
python src/voice_control/llm_brain.py qwen2.5:14b
```

## Dependencies

**Project modules:** none (standalone)

**External packages:** `requests`, `json`, `base64`

**External services:** Ollama (`sudo systemctl start ollama`)
