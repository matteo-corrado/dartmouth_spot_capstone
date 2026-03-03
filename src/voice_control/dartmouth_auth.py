"""JWT authentication helper for Dartmouth Developer API (STT, object-detection).

The langchain-dartmouth library handles auth for ChatDartmouth (LLM/VLM)
automatically via DARTMOUTH_CHAT_API_KEY.  This module covers the *other*
REST endpoints (speech-recognition, etc.) that require a JWT bearer token
obtained by exchanging the developer API key.

Usage:
    from dartmouth_auth import get_auth
    headers = get_auth().auth_header()
    requests.post(url, headers=headers, ...)
"""
import time
import requests

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import DARTMOUTH_API_KEY, DARTMOUTH_JWT_URL


class DartmouthAuth:
    """Manages JWT tokens for the Dartmouth Developer API."""

    # Refresh the JWT this many seconds before it expires
    _REFRESH_MARGIN = 300  # 5 minutes

    def __init__(self, api_key: str = "", jwt_url: str = ""):
        self._api_key = api_key or DARTMOUTH_API_KEY
        self._jwt_url = jwt_url or DARTMOUTH_JWT_URL
        self._jwt: str = ""
        self._expires_at: float = 0.0

    # ------------------------------------------------------------------
    def get_jwt(self) -> str:
        """Return a valid JWT, refreshing if needed."""
        if self._jwt and time.time() < (self._expires_at - self._REFRESH_MARGIN):
            return self._jwt
        return self._refresh_jwt()

    def auth_header(self) -> dict:
        """Return an Authorization header dict with a valid bearer token."""
        return {"Authorization": f"Bearer {self.get_jwt()}"}

    # ------------------------------------------------------------------
    def _refresh_jwt(self) -> str:
        """Exchange the API key for a fresh JWT."""
        if not self._api_key:
            raise RuntimeError(
                "DARTMOUTH_API_KEY is not set.  "
                "Get one at https://developer.dartmouth.edu/keys"
            )

        resp = requests.post(
            self._jwt_url,
            headers={"Authorization": self._api_key},
            timeout=10,
        )
        resp.raise_for_status()

        data = resp.json()
        self._jwt = data.get("jwt", data.get("token", ""))
        if not self._jwt:
            raise RuntimeError(f"JWT exchange returned unexpected payload: {data}")

        # Assume 1-hour lifetime (3600 s) unless the response says otherwise
        expires_in = data.get("expires_in", 3600)
        self._expires_at = time.time() + expires_in

        print(f"[Auth] JWT refreshed (expires in {expires_in}s)")
        return self._jwt


# ---------- Module-level singleton ----------
_auth_instance: DartmouthAuth | None = None


def get_auth() -> DartmouthAuth:
    """Return (and lazily create) the module-level DartmouthAuth singleton."""
    global _auth_instance
    if _auth_instance is None:
        _auth_instance = DartmouthAuth()
    return _auth_instance
