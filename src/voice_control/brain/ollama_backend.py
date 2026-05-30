"""Ollama HTTP backend — Stage 1 path, kept as rollback.

Ollama bug #15260 (format=json + thinking) was fixed in PR #15678 (Apr 2026).
This backend assumes Ollama >= 0.27 with the fix; older versions need
think=False to be explicitly set elsewhere.
"""
from __future__ import annotations
import base64
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import requests

OLLAMA_URL = os.environ.get("SPOT_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("SPOT_OLLAMA_MODEL", "gemma4:e4b")

# Mirror SAMPLING_PROFILES from llamacpp_backend so rollback to Ollama
# preserves the action / freeform / vlm sampling distinction. Stage 2E.1
# will override these per-persona on top of either backend.
SAMPLING_PROFILES: Dict[str, Dict] = {
    "action":   {"temperature": 0.3, "top_p": 0.9,  "top_k": 40, "repeat_penalty": 1.0},
    "freeform": {"temperature": 0.7, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.05},
    "vlm":      {"temperature": 0.5, "top_p": 0.95, "top_k": 64, "repeat_penalty": 1.0},
}


class OllamaBackend:
    name = "ollama"

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
        sampling = {**SAMPLING_PROFILES[profile], **(sampling_overrides or {})}
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "system", "content": system}] + messages,
            "stream": on_token is not None,
            "format": "json",
            "options": {**sampling, "num_predict": max_tokens},
        }
        if on_token is None:
            r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()["message"]["content"]

        full = []
        with requests.post(
            f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout, stream=True
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                delta = obj.get("message", {}).get("content")
                if delta:
                    full.append(delta)
                    on_token(delta)
                if obj.get("done"):
                    break
        return "".join(full)

    def vlm_describe(
        self, system: str, prompt: str, image_bytes: bytes, timeout: float = 60.0
    ) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt, "images": [b64]},
            ],
            "stream": False,
            "options": {**SAMPLING_PROFILES["vlm"], "num_predict": 400},
        }
        r = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"]
