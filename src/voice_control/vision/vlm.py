"""Multi-image VLM client for the local llama-server (Gemma 4 E4B + mmproj).

Gemma's hidden thinking eats the token budget, so every request sets
``chat_template_kwargs.enable_thinking = False``.

Hard rule: one view per call. Do not mix different viewpoints in a single
request. The describe flow sends the stitch + gripper image as two images of
the SAME view (front), which is allowed.
"""

import base64

import cv2
import numpy as np
import requests

from src.voice_control.brain.llamacpp_backend import LLAMACPP_URL

SYSTEM_PROMPT = (
    "You are Spot, a Boston Dynamics robot at Dartmouth College. "
    "The image is what you see right now through your own camera eyes. "
    "Respond naturally in first person as if you are looking around, "
    "NOT as if you are analyzing a photograph. Never mention 'image', "
    "'photo', 'picture', 'angle', or 'vantage point'. "
    "Describe what you see in 2-3 sentences. "
    "Be specific about objects, people, and surroundings. "
    "NEVER say the words 'stop', 'halt', 'freeze' or 'e-stop' (not even "
    "'stop sign' or 'bus stop'): your own voice saying them triggers your "
    "emergency halt. Say 'sign' or 'pause' instead."
)


def encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR ndarray as JPEG bytes."""
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encode failed")
    return buf.tobytes()


def ask_vlm(
    prompt: str,
    images: list[bytes],
    system: str = SYSTEM_PROMPT,
    max_tokens: int = 200,
    timeout: float = 30.0,
) -> str | None:
    """Ask the VLM about JPEG images. Returns stripped text, None on error/empty."""
    content = [{"type": "text", "text": prompt}]
    for jpg in images:
        b64 = base64.b64encode(jpg).decode("ascii")
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        )
    payload = {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.5,
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": True,
    }
    try:
        r = requests.post(
            f"{LLAMACPP_URL}/v1/chat/completions", json=payload, timeout=timeout
        )
        r.raise_for_status()
        text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        print(f"[vlm] error: {e}")
        return None
    return text or None
