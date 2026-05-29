#!/usr/bin/env python3
"""
=============================================================================
High-Performance Batched Feature Fetcher
Retrieves serialized analytical vectors in under 5ms p99 using Redis pipelining.
=============================================================================
"""

import redis
from typing import List, Dict, Any

class FeatureFetcher:
    def __init__(self, host: str = "localhost", port: int = 6379):
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            decode_responses=True,
            max_connections=100,
            socket_timeout=0.5
        )
        self.r = redis.Redis(connection_pool=self.pool)

    def fetch_user_features(self, user_ids: List[str]) -> List[Dict[str, Any]]:
        if not user_ids:
            return []

        pipe = self.r.pipeline(transaction=False)
        for uid in user_ids:
            pipe.hgetall(f"fv:user_aggregates:user:{uid}")
        
        raw_results = pipe.execute()
        
        materialized_profiles = []
        for uid, features in zip(user_ids, raw_results):
            if not features:
                features = {"user_view_count": "0", "user_purchase_count": "0", "user_conversion_rate": "0.0"}
            features["user_id"] = uid
            materialized_profiles.append(features)
            
        return materialized_profiles

    def fetch_item_features(self, item_ids: List[str]) -> List[Dict[str, Any]]:
        if not item_ids:
            return []

        pipe = self.r.pipeline(transaction=False)
        for iid in item_ids:
            pipe.hgetall(f"fv:item_aggregates:item:{iid}")
            
        raw_results = pipe.execute()
        
        materialized_items = []
        for iid, features in zip(item_ids, raw_results):
            if not features:
                features = {"category": "unknown", "base_price": "0.0", "popularity": "0"}
            features["item_id"] = iid
            materialized_items.append(features)
            
        return materialized_items