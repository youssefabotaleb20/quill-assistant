"""A per-client rate limiter for questions answered with the server's own key.

On serverless hosting each instance keeps its own counts, so this is a soft
limit. It only matters when ANTHROPIC_API_KEY is set on the server; with
bring-your-own-key, visitors pay for their own questions.
"""

from __future__ import annotations

import threading
import time
from collections import deque


class RateLimiter:
    """Sliding one-hour window per key."""

    def __init__(self, per_hour: int) -> None:
        self._limit = per_hour
        self._hits: dict[str, deque[float]] = {}
        self._mutex = threading.Lock()

    def allow(self, key: str) -> bool:
        if self._limit <= 0:
            return True
        now = time.time()
        with self._mutex:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] < now - 3600:
                hits.popleft()
            if len(hits) >= self._limit:
                return False
            hits.append(now)
            return True
