"""
shared/utils/config.py
=======================
Central config loaded from environment variables + .env file.
All services import from here — no scattered os.getenv() calls.
"""
from __future__ import annotations

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NexusSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    env:          str   = "development"
    log_level:    str   = "INFO"
    secret_key:   str   = "dev-secret"

    # PostgreSQL
    postgres_url:          str = "postgresql://nexus:nexus_dev@localhost:5432/nexus"
    postgres_pool_size:    int = 20
    postgres_max_overflow: int = 10

    # Redis
    redis_url:              str = "redis://localhost:6379"
    redis_pool_size:        int = 20
    feature_ttl_seconds:    int = 86400

    # Kafka
    kafka_brokers:          str = "localhost:19092"
    kafka_schema_registry:  str = "http://localhost:18081"

    # Offline store
    offline_store_path:     str = "/data/features/offline"

    # MLflow
    mlflow_tracking_uri:    str = "http://localhost:5000"
    mlflow_experiment_name: str = "nexus"

    # Service URLs
    feature_store_url:      str = "http://localhost:8001"
    gateway_url:            str = "http://localhost:8000"

    # Model
    two_tower_dim:          int = 128
    candidate_pool_size:    int = 500
    ranking_candidates:     int = 100
    faiss_index_path:       str = "/data/models/faiss"

    # Simulation
    num_users:              int = 100_000
    num_items:              int = 500_000
    num_categories:         int = 50

    class Config:
        env_prefix = "NEXUS_"


@lru_cache(maxsize=1)
def get_settings() -> NexusSettings:
    return NexusSettings()


settings = get_settings()
