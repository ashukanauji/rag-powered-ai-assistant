from __future__ import annotations

import json
import logging
from typing import Any

import redis

from .config import get_settings

logger = logging.getLogger(__name__)


class CacheClient:
    """Redis cache with no-op fallback.

    Cache is used for repeated questions to lower latency/cost.
    """

    def __init__(self) -> None:
        self.enabled = False
        self._client: redis.Redis | None = None

        redis_url = get_settings().redis_url
        if not redis_url:
            logger.info("Redis cache disabled: REDIS_URL is not configured")
            return

        try:
            self._client = redis.Redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            self.enabled = True
            logger.info("Redis cache enabled")
        except redis.RedisError as exc:
            logger.warning("Redis cache unavailable: %s", exc)
            self.enabled = False
            self._client = None

    def get(self, key: str) -> dict[str, Any] | None:
        if not self.enabled or not self._client:
            return None
        try:
            raw = self._client.get(key)
            if not raw:
                return None
            return json.loads(raw)
        except (redis.RedisError, json.JSONDecodeError, TypeError) as exc:
            logger.warning("Cache GET failed for key=%s: %s", key, exc)
            return None

    def set(self, key: str, value: dict[str, Any], ttl_seconds: int) -> None:
        if not self.enabled or not self._client:
            return
        try:
            self._client.setex(key, ttl_seconds, json.dumps(value))
        except (redis.RedisError, TypeError) as exc:
            logger.warning("Cache SET failed for key=%s: %s", key, exc)
