"""In-memory document store and a per-client rate limiter.

Uploads live in memory and expire after a TTL. That keeps the demo free of a
database and means nobody's file is kept around. A restart clears everything.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque

from .documents import Document


class DocumentStore:
    def __init__(self, ttl_seconds: int, max_documents: int) -> None:
        self._ttl = ttl_seconds
        self._max = max_documents
        self._docs: OrderedDict[str, Document] = OrderedDict()
        self._locks: dict[str, threading.Lock] = {}
        self._mutex = threading.Lock()

    def add(self, doc: Document) -> None:
        with self._mutex:
            self._evict_expired()
            while len(self._docs) >= self._max:
                old_id, _ = self._docs.popitem(last=False)
                self._locks.pop(old_id, None)
            self._docs[doc.id] = doc
            self._locks[doc.id] = threading.Lock()

    def get(self, doc_id: str) -> Document | None:
        with self._mutex:
            self._evict_expired()
            return self._docs.get(doc_id)

    def lock(self, doc_id: str) -> threading.Lock | None:
        with self._mutex:
            return self._locks.get(doc_id)

    def remove(self, doc_id: str) -> bool:
        with self._mutex:
            self._locks.pop(doc_id, None)
            return self._docs.pop(doc_id, None) is not None

    def _evict_expired(self) -> None:
        cutoff = time.time() - self._ttl
        for doc_id in [d for d, doc in self._docs.items() if doc.created_at < cutoff]:
            self._docs.pop(doc_id, None)
            self._locks.pop(doc_id, None)


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
