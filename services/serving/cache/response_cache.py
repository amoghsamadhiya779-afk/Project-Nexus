#!/usr/bin/env python3
"""
=============================================================================
Response Cache Layer (inspired by Netflix RecSysOps)
Uses Redis to store serialized JSON API payloads, implementing deterministic
request-parameter hashing to enforce sub-millisecond p99 cached latencies.
=============================================================================
"""

import json
import hashlib
import redis
from typing import Optional, Any

class ResponseCache:
    def __init__(self, host: str = "localhost", port: int = 6379, default_ttl: int = 60):
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            decode_responses=True,
            max_connections=50
        )
        self.r = redis.Redis(connection_pool=self.pool)
        self.default_ttl = default_ttl

    def _generate_key(self, endpoint: str, identifier: str, parameters: Optional[dict] = None) -> str:
        param_hash = ""
        if parameters:
            serialized_params = json.dumps(parameters, sort_keys=True)
            param_hash = ":" + hashlib.sha256(serialized_params.encode()).hexdigest()[:16]
        return f"cache:{endpoint}:{identifier}{param_hash}"

    def get(self, endpoint: str, identifier: str, parameters: Optional[dict] = None) -> Optional[Any]:
        key = self._generate_key(endpoint, identifier, parameters)
        try:
            cached_data = self.r.get(key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            print(f"[⚠️ Cache Read Error] Failed to read from key '{key}': {e}")
        return None

    def set(self, endpoint: str, identifier: str, data: Any, parameters: Optional[dict] = None, ttl: Optional[int] = None) -> bool:
        key = self._generate_key(endpoint, identifier, parameters)
        expiration = ttl if ttl is not None else self.default_ttl
        try:
            serialized = json.dumps(data)
            return self.r.setex(key, expiration, serialized)
        except Exception as e:
            print(f"[⚠️ Cache Write Error] Failed to write to key '{key}': {e}")
            return False

    def invalidate(self, endpoint: str, identifier: str) -> bool:
        pattern = f"cache:{endpoint}:{identifier}*"
        try:
            keys = self.r.keys(pattern)
            if keys:
                return bool(self.r.delete(*keys))
        except Exception as e:
            pass
        return False