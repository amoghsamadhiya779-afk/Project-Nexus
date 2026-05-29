#!/usr/bin/env python3
import time
import os
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from services.serving.feature_fetcher.fetcher import FeatureFetcher
from services.serving.model_server.server import LocalModelServer
from services.search.retrieval.dense_retrieval import DenseRetrievalEngine
from services.serving.cache.response_cache import ResponseCache

app = FastAPI(
    title="Nexus RecSys Inference Gateway",
    description="Unified High-Performance RecSysOps Pipeline Serving Engine",
    version="1.0.0"
)

# Initialize service singletons
fetcher = FeatureFetcher(host="localhost", port=6379)
model_server = LocalModelServer()
search_engine = DenseRetrievalEngine()
cache = ResponseCache(default_ttl=30) 

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., example="user_75")
    k_candidates: int = Field(default=20, ge=1, le=100)
    use_cache: bool = Field(default=True)

class RecommendationResponse(BaseModel):
    user_id: str
    items: List[str]
    ctr_predictions: List[float]
    cvr_predictions: List[float]
    cached: bool
    latency_ms: float

class SearchRequest(BaseModel):
    user_id: str = Field(..., example="user_75")
    query: str = Field(..., example="electronics under 200")
    k_results: int = Field(default=10, ge=1, le=50)

class SearchResponse(BaseModel):
    user_id: str
    query: str
    results: List[str]
    relevance_scores: List[float]
    latency_ms: float

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    start_time = time.perf_counter()

    if request.use_cache:
        cached_res = cache.get("recommend", request.user_id, {"k": request.k_candidates})
        if cached_res:
            latency = (time.perf_counter() - start_time) * 1000.0
            cached_res["cached"] = True
            cached_res["latency_ms"] = round(latency, 2)
            return cached_res

    try:
        user_features = fetcher.fetch_user_features([request.user_id])[0]
        user_emb = model_server.predict_user_embedding(user_features)
        
        candidate_ids, _ = search_engine.retrieve_candidates(user_emb, k=request.k_candidates)
        item_features_list = fetcher.fetch_item_features(candidate_ids)
        
        ctr_preds, cvr_preds = model_server.score_ranking_batch(user_features, item_features_list)
        sorted_idx = np.argsort(ctr_preds)[::-1]
        
        latency = (time.perf_counter() - start_time) * 1000.0
        response_payload = {
            "user_id": request.user_id,
            "items": [candidate_ids[idx] for idx in sorted_idx],
            "ctr_predictions": [float(ctr_preds[idx]) for idx in sorted_idx],
            "cvr_predictions": [float(cvr_preds[idx]) for idx in sorted_idx],
            "cached": False,
            "latency_ms": round(latency, 2)
        }

        if request.use_cache:
            cache.set("recommend", request.user_id, response_payload, {"k": request.k_candidates})

        return response_payload
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation serving failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
def get_search_results(request: SearchRequest):
    start_time = time.perf_counter()
    try:
        user_features = fetcher.fetch_user_features([request.user_id])[0]
        user_emb = model_server.predict_user_embedding(user_features)
        
        candidate_ids, _ = search_engine.retrieve_candidates(user_emb, k=request.k_results * 2)
        item_features_list = fetcher.fetch_item_features(candidate_ids)
        
        ranked_items = search_engine.rescore_with_ltr(user_features, item_features_list)[:request.k_results]

        latency = (time.perf_counter() - start_time) * 1000.0
        return SearchResponse(
            user_id=request.user_id, query=request.query,
            results=[item[0] for item in ranked_items],
            relevance_scores=[item[1] for item in ranked_items],
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search serving failed: {str(e)}")

@app.post("/index/refresh")
def refresh_vector_index():
    try:
        item_keys = fetcher.r.keys("fv:item_aggregates:item:*")
        if not item_keys:
            raise HTTPException(status_code=404, detail="No hydrated item metrics in Redis.")
            
        item_ids = [k.split(":")[-1] for k in item_keys]
        
        # Build vectors and pass to the new DenseRetrievalEngine
        embeddings_arr = np.array([np.random.randn(16).astype(np.float32) for _ in item_ids])
        search_engine.build_index(embeddings_arr, item_ids)
        
        return {"status": "success", "indexed_items": len(item_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector index update failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
