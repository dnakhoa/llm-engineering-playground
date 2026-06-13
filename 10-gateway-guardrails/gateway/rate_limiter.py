"""Token-bucket rate limiter (in-memory, no Redis dependency)."""

import time
from typing import Optional


class RateLimiter:
    """Per-user rate limiting using a token bucket algorithm."""

    def __init__(self):
        self._buckets: dict[str, dict] = {}

    def check_rate_limit(
        self, user_id: str, limit: int = 60, window: int = 60
    ) -> tuple[bool, Optional[int]]:
        """
        Check if user is within rate limit.

        Args:
            user_id: The user identifier
            limit: Max requests per window
            window: Time window in seconds

        Returns:
            (allowed, retry_after_seconds)
        """
        now = time.time()
        bucket = self._buckets.get(user_id)

        if bucket is None or now - bucket["window_start"] >= window:
            self._buckets[user_id] = {"window_start": now, "count": 1}
            return True, None

        bucket["count"] += 1
        if bucket["count"] > limit:
            retry_after = int(window - (now - bucket["window_start"])) + 1
            return False, retry_after

        return True, None

    def check_quota(self, user_id: str, tokens_needed: int, monthly_limit: int = 100_000) -> bool:
        """Check if user has remaining token quota (monthly)."""
        key = f"quota:{user_id}"
        remaining = self._buckets.get(key, {"remaining": monthly_limit})

        if remaining["remaining"] < tokens_needed:
            return False

        remaining["remaining"] -= tokens_needed
        self._buckets[key] = remaining
        return True
