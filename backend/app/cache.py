from __future__ import annotations

import json
import os
from typing import Any

import redis


class CacheClient:
    """Redis cache with no-op fallback.

    Cache is used for repeated questions to lower latency/cost.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._client: redis.Redis | None = None

        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return

        try:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            self.enabled = True
        except Exception:
            self.enabled = False
            self._client = None

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled or not self._client:
            return None
        raw = self._client.get(key)
        if not raw:
            return None
        return json.loads(raw)

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 1800) -> None:
        if not self.enabled or not self._client:
            return
        self._client.setex(key, ttl_seconds, json.dumps(value))
