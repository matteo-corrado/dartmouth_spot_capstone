"""llama.cpp HTTP backend.

Single API: chat(system, messages, on_token=None) -> str
- If on_token is None: blocking, returns full response.
- If on_token is callable: streams tokens via SSE; on_token(text_delta) is called
  for each delta, and the full concatenated text is returned at the end.

Always sends the unified GBNF. Brain post-parses the JSON to extract actions vs response.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

LLAMACPP_URL = os.environ.get("SPOT_LLAMACPP_URL", "http://127.0.0.1:11435")
GRAMMAR_PATH = Path(__file__).resolve().parents[1] / "grammar" / "spot_action.gbnf"

# Three sampling profiles. chat() picks action|freeform via _sampling_hint;
# vlm_describe() pins profile="vlm". The hint only affects sampling, never
# the code path or grammar — if the hint is wrong, GBNF still produces a
# valid {actions, response} JSON; only the temperature/top_p is suboptimal.
# Profile values follow the Gemma 4 prompting guide (top_k=64, top_p=0.95)
# for freeform; tighter for action determinism; middle for VLM captions.
# Stage 2E.1 will read+override these per-persona (e.g. pirate may want
# higher vlm temp than butler).
SAMPLING_PROFILES: Dict[str, Dict] = {
    "action":   {"temperature": 0.3, "top_p": 0.9,  "top_k": 40, "repeat_penalty": 1.0},
    "freeform": {"temperature": 0.7, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.05},
    "vlm":      {"temperature": 0.5, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.0},
}

_GRAMMAR_CACHE: Optional[str] = None


def _grammar() -> str:
    global _GRAMMAR_CACHE
    if _GRAMMAR_CACHE is None:
        _GRAMMAR_CACHE = GRAMMAR_PATH.read_text(encoding="utf-8")
    return _GRAMMAR_CACHE


class LlamaCppBackend:
    name = "llamacpp"

    def __init__(self, url: str = LLAMACPP_URL):
        self.url = url

    def chat(
        self,
        system: str,
        messages: List[Dict[str, str]],
        on_token: Optional[Callable[[str], None]] = None,
        timeout: float = 60.0,
        profile: str = "action",
        sampling_overrides: Optional[Dict] = None,
        max_tokens: int = 512,
    ) -> str:
        # 2E.1 personas pass sampling_overrides to vary temp/top_p per persona
        # (vlm temp is the canonical example). Default (None) preserves 2A behavior.
        sampling = {**SAMPLING_PROFILES[profile], **(sampling_overrides or {})}
        payload: Dict = {
            "messages": [{"role": "system", "content": system}] + messages,
            "grammar": _grammar(),
            "max_tokens": max_tokens,
            "cache_prompt": True,
            **sampling,
        }
        if on_token is None:
            r = requests.post(
                f"{self.url}/v1/chat/completions", json=payload, timeout=timeout
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]

        # Streaming SSE path
        payload["stream"] = True
        full = []
        with requests.post(
            f"{self.url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
            stream=True,
        ) as r:
            r.raise_for_status()
            for raw_line in r.iter_lines(decode_unicode=True):
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("choices", [{}])[0].get("delta", {}).get("content")
                if delta:
                    full.append(delta)
                    on_token(delta)
        return "".join(full)

    def vlm_describe(
        self, system: str, prompt: str, image_path: Path, timeout: float = 60.0
    ) -> str:
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        payload = {
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        },
                    ],
                },
            ],
            "max_tokens": 400,
            "cache_prompt": True,
            **SAMPLING_PROFILES["vlm"],
        }
        r = requests.post(
            f"{self.url}/v1/chat/completions", json=payload, timeout=timeout
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
