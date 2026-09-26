"""JWT Authentication for LLM API Gateway."""

import os
import time
import hashlib
import hmac
import base64
import json
from typing import Optional


class AuthManager:
    """Simple JWT-like token auth (no external dependencies)."""

    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv("JWT_SECRET_KEY", "dev-secret-change-me")

    def create_token(self, user_id: str, roles: list[str] = None, expires_in: int = 3600) -> str:
        """Create an auth token for a user."""
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {
            "sub": user_id,
            "roles": roles or ["user"],
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in,
        }
        return self._encode(header, payload)

    def verify_token(self, token: str) -> Optional[dict]:
        """Verify a token and return claims, or None if invalid."""
        try:
            header, payload, signature = token.split(".")
            expected_sig = self._sign(header, payload)
            if not hmac.compare_digest(signature, expected_sig):
                return None
            if payload.get("exp", 0) < time.time():
                return None
            return payload
        except Exception:
            return None

    def check_permission(self, claims: dict, required_role: str) -> bool:
        """Check if the token claims include the required role."""
        return required_role in claims.get("roles", [])

    def _sign(self, header_b64: str, payload_b64: str) -> str:
        data = f"{header_b64}.{payload_b64}"
        sig = hmac.new(self.secret_key.encode(), data.encode(), hashlib.sha256).digest()
        return base64.urlsafe_b64encode(sig).rstrip(b"=").decode()

    def _encode(self, header: dict, payload: dict) -> str:
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).rstrip(b"=").decode()
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
        sig = self._sign(header_b64, payload_b64)
        return f"{header_b64}.{payload_b64}.{sig}"
