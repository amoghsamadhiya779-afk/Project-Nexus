"""
services/feature_store/api/server.py
======================================
Feature Store REST API.

Endpoints:
  GET  /features/online        — fetch online features for entity list
  POST /features/batch         — fetch features for training (point-in-time)
  GET  /registry               — list all feature views
  GET  /registry/{view_name}   — feature view schema
  POST /materialise/{view}     — trigger batch materialisation
  GET  /health                 — health check
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import redis.asyncio as aioredis
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import make_asgi_app
from pydantic import BaseModel

from services.feature_store.core.feature_view import (
    build_marketplace_registry, FeatureRegistry
)
from services.feature_store.storage.stores import (
    RedisOnlineStore, ParquetOfflineStore, MaterialisationEngine
)
from shared.utils.config import settings
from shared.monitoring.metrics import feature_store_metrics


# ─── Lifespan: startup / shutdown ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Feature Store API starting up...")

    # Initialise stores
    app.state.online_store  = RedisOnlineStore(
        host=settings.redis_url.split("//")[1].split(":")[0],
        port=int(settings.redis_url.split(":")[-1]),
    )
    app.state.offline_store = ParquetOfflineStore(settings.offline_store_path)
    app.state.engine        = MaterialisationEngine(
        app.state.online_store, app.state.offline_store
    )

    # Build feature registry
    app.state.registry: FeatureRegistry = build_marketplace_registry()
    logger.info(f"Registry loaded: {app.state.registry.list_views()}")

    # Health check
    healthy = await app.state.online_store.health_check()
    if not healthy:
        logger.warning("Redis health check failed — online store may be unavailable")

    yield

    # Cleanup
    logger.info("Feature Store API shutting down...")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Nexus Feature Store",
    description="Production feature serving API with point-in-time correctness",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Prometheus metrics endpoint
app.mount("/metrics", make_asgi_app())


# ─── Request / Response schemas ────────────────────────────────────────────────

class OnlineFeatureRequest(BaseModel):
    feature_view: str
    entity_ids:   List[str]            # user_ids or item_ids
    feature_names: Optional[List[str]] = None   # None = all features


class OnlineFeatureResponse(BaseModel):
    feature_view: str
    features:     Dict[str, Optional[Dict[str, Any]]]  # entity_id → {feature: value}
    latency_ms:   float
    cache_hits:   int
    cache_misses: int


class MaterialiseRequest(BaseModel):
    feature_view: str
    partition:    Optional[str] = None   # e.g. "2024-01-15"


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    online_ok = await app.state.online_store.health_check()
    return {
        "status":       "ok" if online_ok else "degraded",
        "online_store": "ok" if online_ok else "unavailable",
        "registry":     len(app.state.registry.list_views()),
    }


@app.post("/features/online", response_model=OnlineFeatureResponse)
async def get_online_features(req: OnlineFeatureRequest):
    """
    Batch fetch features from Redis online store.

    Designed for serving-time use: single round-trip to Redis
    regardless of entity count. p99 target: < 5ms for 200 features.
    """
    start = time.perf_counter()
    registry: FeatureRegistry = app.state.registry
    store: RedisOnlineStore   = app.state.online_store

    view = registry.get_view(req.feature_view)
    if view is None:
        raise HTTPException(404, f"Feature view '{req.feature_view}' not found")

    # Build Redis keys
    keys = [view.get_feature_key(eid) for eid in req.entity_ids]

    # Batch fetch
    raw_results = await store.get(keys)

    # Build response
    result_map: Dict[str, Optional[Dict[str, Any]]] = {}
    cache_hits = cache_misses = 0

    for entity_id, raw in zip(req.entity_ids, raw_results):
        if raw is None:
            result_map[entity_id] = None
            cache_misses += 1
        else:
            # Filter to requested features
            if req.feature_names:
                raw = {k: v for k, v in raw.items() if k in req.feature_names}
            result_map[entity_id] = raw
            cache_hits += 1

    latency_ms = (time.perf_counter() - start) * 1000
    feature_store_metrics.online_read_latency.observe(latency_ms)

    hit_rate = cache_hits / max(len(req.entity_ids), 1)
    feature_store_metrics.cache_hit_rate.labels(
        feature_view=req.feature_view
    ).set(hit_rate)

    return OnlineFeatureResponse(
        feature_view=req.feature_view,
        features=result_map,
        latency_ms=round(latency_ms, 2),
        cache_hits=cache_hits,
        cache_misses=cache_misses,
    )


@app.get("/registry")
async def list_views():
    registry: FeatureRegistry = app.state.registry
    return {
        "feature_views": registry.list_views(),
        "fingerprint":   registry.fingerprint(),
        "schema":        registry.schema(),
    }


@app.get("/registry/{view_name}")
async def get_view(view_name: str):
    registry: FeatureRegistry = app.state.registry
    view = registry.get_view(view_name)
    if view is None:
        raise HTTPException(404, f"View '{view_name}' not found")
    return {
        "name":        view.name,
        "features":    view.feature_names,
        "entities":    [e.name for e in view.entities],
        "source_type": view.source.source_type.value,
        "ttl_days":    view.ttl.days,
        "tags":        view.tags,
        "description": view.description,
    }


@app.post("/materialise/{view_name}")
async def trigger_materialisation(
    view_name: str,
    background_tasks: BackgroundTasks,
):
    """Trigger async batch materialisation for a feature view."""
    registry: FeatureRegistry = app.state.registry
    view = registry.get_view(view_name)
    if view is None:
        raise HTTPException(404, f"View '{view_name}' not found")

    async def run_materialisation():
        from services.feature_store.pipeline.batch_pipeline import BatchPipeline
        pipeline = BatchPipeline(app.state.engine, registry)
        await pipeline.run_view(view_name)

    background_tasks.add_task(run_materialisation)
    return {"status": "materialisation_triggered", "feature_view": view_name}
