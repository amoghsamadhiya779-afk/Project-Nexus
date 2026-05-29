import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import numpy as np

from services.serving.feature_fetcher.fetcher import FeatureFetcher
from shared.monitoring.metrics import track_latency

app = FastAPI(title="Nexus Unified Serving Gateway", version="1.0.0")
fetcher = FeatureFetcher()

class RecommendRequest(BaseModel):
    user_id: str
    candidate_item_ids: List[str]

class RecommendResponse(BaseModel):
    user_id: str
    items: List[str]
    scores: List[float]
    latency_ms: float

@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    start_time = time.perf_counter()
    
    try:
        # 1. Batched Feature Extraction
        user_feats = fetcher.fetch_user_features([request.user_id])[0]
        item_feats_list = fetcher.fetch_item_features(request.candidate_item_ids)
        
        # 2. Mock Multi-Task ranking prediction using retrieved features
        scores = []
        cvr_weight = float(user_feats.get("user_conversion_rate", 0.0))
        
        for item in item_feats_list:
            popularity = float(item.get("popularity", 0))
            price = float(item.get("base_price", 0.0))
            
            # Algorithmic ranking score simulation (Popularity and conversion bias)
            score = (popularity * 0.4) + (cvr_weight * 100.0) - (price * 0.01)
            scores.append(float(np.tanh(score / 1000.0))) # Bound scores between -1 and 1
        
        # Sort candidates based on simulated ranking score
        sorted_pairs = sorted(zip(request.candidate_item_ids, scores), key=lambda x: x[1], reverse=True)
        sorted_items = [p[0] for p in sorted_pairs]
        sorted_scores = [p[1] for p in sorted_pairs]
        
        latency = (time.perf_counter() - start_time) * 1000.0
        
        # Log telemetry metrics to Prometheus tracker
        track_latency("recommend", latency)
        
        return RecommendResponse(
            user_id=request.user_id,
            items=sorted_items,
            scores=sorted_scores,
            latency_ms=round(latency, 2)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
