from unittest.mock import MagicMock, patch

import numpy as np
import requests

from src.voice_control.vision import vlm


def _resp(content, status=200):
    r = MagicMock()
    r.status_code = status
    r.raise_for_status.side_effect = (
        requests.HTTPError("boom") if status >= 400 else None
    )
    r.json.return_value = {"choices": [{"message": {"content": content}}]}
    return r


def test_payload_thinking_off_and_images():
    with patch("requests.post", return_value=_resp("  I see a door.  ")) as p:
        out = vlm.ask_vlm("what?", [b"a", b"b", b"c"])
    assert out == "I see a door."
    url = p.call_args.args[0]
    assert url.endswith("/v1/chat/completions")
    body = p.call_args.kwargs["json"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["cache_prompt"] is True
    assert body["temperature"] == 0.5
    assert body["max_tokens"] == 200
    assert body["messages"][0] == {"role": "system", "content": vlm.SYSTEM_PROMPT}
    parts = body["messages"][1]["content"]
    imgs = [x for x in parts if x["type"] == "image_url"]
    assert len(imgs) == 3
    assert imgs[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert [x for x in parts if x["type"] == "text"][0]["text"] == "what?"


def test_http_error_returns_none():
    with patch("requests.post", return_value=_resp("x", 500)):
        assert vlm.ask_vlm("q", [b"a"]) is None


def test_exception_returns_none():
    with patch("requests.post", side_effect=requests.ConnectionError()):
        assert vlm.ask_vlm("q", [b"a"]) is None


def test_empty_content_returns_none():
    with patch("requests.post", return_value=_resp("   ")):
        assert vlm.ask_vlm("q", [b"a"]) is None
    with patch("requests.post", return_value=_resp(None)):
        assert vlm.ask_vlm("q", [b"a"]) is None


def test_system_prompt_has_stop_rule():
    assert "NEVER say the words 'stop'" in vlm.SYSTEM_PROMPT


def test_encode_jpeg():
    b = vlm.encode_jpeg(np.zeros((8, 8, 3), np.uint8))
    assert b[:2] == b"\xff\xd8"
