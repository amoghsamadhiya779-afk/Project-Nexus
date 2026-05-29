#!/usr/bin/env python3
"""
=============================================================================
Nexus-FS: Declarative Feature Definitions
Defines entities, data sources, and feature views as code (Inspired by Feast/Feathr).
This ensures feature consistency across offline training and online serving.
=============================================================================
"""

from typing import List, Dict, Optional
from pydantic import BaseModel

class DataSource(BaseModel):
    name: str
    type: str # 'kafka', 'postgres', 'parquet'
    path: str
    timestamp_column: str

class Entity(BaseModel):
    name: str
    join_key: str
    description: str

class FeatureView(BaseModel):
    name: str
    entities: List[str]
    ttl_days: int
    source: DataSource
    features: List[Dict[str, str]]

# --- DECLARATIVE DEFINITIONS ---

user_entity = Entity(
    name="user",
    join_key="user_id",
    description="Marketplace customer entity"
)

interaction_stream = DataSource(
    name="kafka_interactions",
    type="kafka",
    path="nexus.user.interactions",
    timestamp_column="event_timestamp"
)

user_aggregate_features = FeatureView(
    name="user_behavior_aggregates",
    entities=["user"],
    ttl_days=30,
    source=interaction_stream,
    features=[
        {"name": "user_view_count", "dtype": "INT32", "transformation": "COUNT(view)"},
        {"name": "user_purchase_count", "dtype": "INT32", "transformation": "COUNT(purchase)"},
        {"name": "user_conversion_rate", "dtype": "FLOAT", "transformation": "purchase / view"}
    ]
)

if __name__ == "__main__":
    print("[*] Compiling Nexus-FS Declarative Definitions...")
    print(f"[+] Successfully registered Feature View: {user_aggregate_features.name}")
    print(f"    - Backed by: {user_aggregate_features.source.type} ({user_aggregate_features.source.path})")
    print(f"    - Tracking {len(user_aggregate_features.features)} real-time features.")