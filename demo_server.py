#!/usr/bin/env python3
"""
Nexus OS — Standalone Demo API Server
Mirrors all endpoints from the full gateway but uses mock data,
so no Redis, Ray, FAISS, or model weights are required.
"""
import time, random, os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(
    title="Nexus RecSys Inference Gateway (Demo)",
    description="Standalone demo server — mock data, no external deps",
    version="1.0.0-demo",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ──────────────────────────────────────────────────

class RecommendationRequest(BaseModel):
    user_id: str = Field(default="user_75")
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
    user_id: str = Field(default="user_75")
    query: str = Field(default="electronics under 200")
    k_results: int = Field(default=10, ge=1, le=50)

class SearchResponse(BaseModel):
    user_id: str
    query: str
    results: List[str]
    relevance_scores: List[float]
    latency_ms: float

class ChatRequest(BaseModel):
    message: str
    user_id: str = "guest"

class ChatResponse(BaseModel):
    response: str
    command_executed: Optional[str] = None
    data: Optional[dict] = None

class TelemetryResponse(BaseModel):
    redis_connected: bool
    cache_hit_rate: float
    ray_gpu_utilization: float
    inference_throughput: float
    flink_status: str
    mlflow_status: str
    active_alerts: List[str]
    system_latency_p99: float

# ── Mock Data Generators ─────────────────────────────────────────────

ITEM_POOL = [
    "item_1042", "item_2198", "item_3347", "item_4561", "item_5023",
    "item_6714", "item_7899", "item_8245", "item_9310", "item_1156",
    "item_2287", "item_3409", "item_4672", "item_5534", "item_6801",
    "item_7063", "item_8190", "item_9452", "item_1378", "item_2590",
]

def mock_recommendations(user_id: str, k: int):
    start = time.perf_counter()
    items = random.sample(ITEM_POOL, min(k, len(ITEM_POOL)))
    ctr = sorted([round(random.uniform(0.05, 0.95), 4) for _ in items], reverse=True)
    cvr = [round(c * random.uniform(0.1, 0.5), 4) for c in ctr]
    latency = (time.perf_counter() - start) * 1000 + random.uniform(8, 18)
    return {
        "user_id": user_id,
        "items": items,
        "ctr_predictions": ctr,
        "cvr_predictions": cvr,
        "cached": False,
        "latency_ms": round(latency, 2),
    }

def mock_search(query: str, k: int):
    start = time.perf_counter()
    results = random.sample(ITEM_POOL, min(k, len(ITEM_POOL)))
    scores = sorted([round(random.uniform(0.4, 0.99), 4) for _ in results], reverse=True)
    latency = (time.perf_counter() - start) * 1000 + random.uniform(5, 12)
    return results, scores, round(latency, 2)

# ── API Routes ────────────────────────────────────────────────────────

@app.post("/recommend", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest):
    return mock_recommendations(request.user_id, request.k_candidates)

@app.post("/search", response_model=SearchResponse)
def get_search_results(request: SearchRequest):
    results, scores, latency = mock_search(request.query, request.k_results)
    return SearchResponse(
        user_id=request.user_id, query=request.query,
        results=results, relevance_scores=scores, latency_ms=latency,
    )

@app.get("/api/telemetry", response_model=TelemetryResponse)
def get_telemetry():
    return TelemetryResponse(
        redis_connected=True,
        cache_hit_rate=round(random.uniform(92.0, 99.5), 1),
        ray_gpu_utilization=round(random.uniform(45.0, 72.0), 1),
        inference_throughput=round(random.uniform(3800.0, 4400.0), 0),
        flink_status="RUNNING",
        mlflow_status="ACTIVE",
        active_alerts=(
            ["Feature Drift Detected in Category: Electronics"]
            if random.random() > 0.85 else []
        ),
        system_latency_p99=round(random.uniform(8.5, 14.5), 2),
    )

@app.post("/api/chat", response_model=ChatResponse)
def chat_agent(request: ChatRequest):
    msg = request.message.strip()
    msg_lower = msg.lower()

    if msg_lower == "/help":
        help_text = (
            "### Nexus OS Command Interface\n"
            "Supported commands:\n"
            "- `/recommend <user_id>` : Run the two-tower and MMoE recommender pipeline for a user.\n"
            "- `/search <query>` : Run the dense retrieval semantic search and LambdaMART LTR reranker.\n"
            "- `/drift` : Run a real-time data drift analysis on interaction logs.\n"
            "- `/train` : Simulate Ray distributed model training for the recommendation towers.\n"
            "- `/help` : Display this menu."
        )
        return ChatResponse(response=help_text, command_executed="/help")

    if msg_lower.startswith("/recommend"):
        parts = msg.split()
        target_user = parts[1] if len(parts) > 1 else "user_75"
        res = mock_recommendations(target_user, 20)
        items_str = ", ".join([f"`{item}`" for item in res["items"][:5]])
        resp_msg = (
            f"### [EXECUTION SUCCESS] Recommendation Pipeline\n"
            f"- **Target User**: `{target_user}`\n"
            f"- **Online Features**: Fetched user profile from Redis online store (latency: 1.1ms).\n"
            f"- **Candidate Gen**: Two-Tower vector search returned {len(res['items'])} items (latency: 3.5ms).\n"
            f"- **MMoE Ranking**: Scored CTR/CVR predictions using PyTorch model (latency: 7.2ms).\n"
            f"- **Top Recommendations**: {items_str} (and {len(res['items']) - 5} more).\n"
            f"- **Total Latency**: `{res['latency_ms']} ms`"
        )
        return ChatResponse(response=resp_msg, command_executed="/recommend", data=res)

    if msg_lower.startswith("/search"):
        parts = msg.split(None, 1)
        query = parts[1] if len(parts) > 1 else "laptops under 500"
        results, scores, latency = mock_search(query, 5)
        items_str = "\n".join(
            [f"{i+1}. `{item}` (LTR score: {scores[i]:.4f})" for i, item in enumerate(results)]
        )
        resp_msg = (
            f"### [EXECUTION SUCCESS] Semantic Search Query\n"
            f"- **Query**: *\"{query}\"*\n"
            f"- **Bi-Encoder Dense Retrieval**: Generated query embedding and searched FAISS index (latency: 4.8ms).\n"
            f"- **LambdaMART LTR Rerank**: Rescored candidates with LightGBM (latency: 5.4ms).\n"
            f"- **Reranked Results**:\n{items_str}\n"
            f"- **Total Latency**: `{latency} ms`"
        )
        search_data = {"user_id": "user_75", "query": query, "results": results, "relevance_scores": scores, "latency_ms": latency}
        return ChatResponse(response=resp_msg, command_executed="/search", data=search_data)

    if msg_lower.startswith("/drift"):
        drift_logs = (
            "### [MLOPS PIPELINE] Data Drift Analysis\n"
            "`[INFO]` Analyzing baseline features from offline store (Parquet logs)...\n"
            "`[INFO]` Fetching window features (last 24h) from Flink stream aggregations...\n"
            "`[WARN]` Drift detected in features `user_view_count` and `item_popularity`.\n"
            "- **KS-Test statistic**: 0.084 (p-value: 0.0021 < alpha 0.05)\n"
            "- **Population Stability Index (PSI)**: 0.24 (> 0.20 threshold - Action Required)\n"
            "🚨 **System Status**: Drift alert raised. Re-training is recommended to prevent recommendation decay. Run `/train` to rebuild towers."
        )
        return ChatResponse(response=drift_logs, command_executed="/drift")

    if msg_lower.startswith("/train"):
        train_logs = (
            "### [MLOPS PIPELINE] Model Retraining Started\n"
            "`[1/6]` Provisioning Ray training cluster on 4x NVIDIA A100... [OK]\n"
            "`[2/6]` Loading Parquet interaction dataset (2.4M rows)... [OK]\n"
            "`[3/6]` Training Two-Tower retrieval model (PyTorch, InfoNCE loss):\n"
            "   - Epoch 1/5 | loss: 0.4521 | recall@100: 0.724\n"
            "   - Epoch 2/5 | loss: 0.2104 | recall@100: 0.798\n"
            "   - Epoch 3/5 | loss: 0.0984 | recall@100: 0.841\n"
            "   - Epoch 4/5 | loss: 0.0451 | recall@100: 0.879\n"
            "   - Epoch 5/5 | loss: 0.0210 | recall@100: 0.892\n"
            "`[4/6]` Re-scoring LightGBM LambdaMART LTR ranker... [OK]\n"
            "`[5/6]` Serializing models and exporting to MLflow Model Registry... [OK]\n"
            "`[6/6]` Running canary deployments on Triton Inference Server... [OK]\n"
            "🎉 **Retraining Success!** New model registered in MLflow (`v2.4.0`) and activated. p99 Serving latency stabilized at 11.2ms."
        )
        return ChatResponse(response=train_logs, command_executed="/train")

    # Fallback — keyword matching
    ans = ""
    if any(kw in msg_lower for kw in ["feature store", "nexus-fs", "redis", "flink"]):
        ans = (
            "### 🪐 Nexus Declarative Feature Store (Nexus-FS)\n"
            "The feature store combines real-time streaming pipelines and offline point-in-time correct calculations.\n\n"
            "- **Streaming aggregation**: Handled by **Apache Flink** consuming marketplace interactions from Kafka/Redpanda.\n"
            "- **Online Store**: Hydrates **Redis Hashes** for sub-5ms feature fetching.\n"
            "- **Offline Store**: Stores historical interactions in **PostgreSQL + Parquet** formats for time-travel joins and training dataset generation.\n"
            "- **Declarative API**: Defined in YAML. Lines of lineage map sources to entities, preventing training-serving skew."
        )
    elif any(kw in msg_lower for kw in ["recommender", "two tower", "mmoe", "faiss"]):
        ans = (
            "### 🗼 Two-Stage Recommendation Architecture\n"
            "Nexus operates a high-throughput recommendation flow designed for scale:\n\n"
            "1. **Candidate Generation (Retrieval)**:\n"
            "   - Uses a PyTorch **Two-Tower Model** mapping user and item features to 64D embeddings.\n"
            "   - **FAISS (HNSW index)** indexes item vectors to retrieve the top 50 candidates in < 3ms.\n"
            "2. **Ranking Stage**:\n"
            "   - Uses a **Multi-gate Mixture-of-Experts (MMoE)** neural network written in PyTorch.\n"
            "   - Predicts independent probability scores for **Click-Through Rate (CTR)** and **Conversion Rate (CVR)**.\n"
            "   - Combines scores using scalar calibration for business-constrained sorting."
        )
    elif any(kw in msg_lower for kw in ["forecasting", "causal", "n-beats"]):
        ans = (
            "### 📈 Hierarchical Forecasting & CausalImpact\n"
            "- **Models**: Uses an ensemble of **N-BEATS** and **Temporal Fusion Transformers (TFT)** to output multi-horizon forecasts.\n"
            "- **Causal Inference**: Uses Google's **CausalImpact** (DiD and Synthetic Control) to run intervention analyses."
        )
    elif any(kw in msg_lower for kw in ["fraud", "gnn", "graphsage"]):
        ans = (
            "### 🛡️ Graph Fraud Detection (Grab Inspired)\n"
            "- **Network Structure**: Builds user-device-payment graphs of marketplace transactions.\n"
            "- **Inference**: Executes a **GraphSAGE** GNN (PyTorch Geometric) to compute structural node embeddings.\n"
            "- **Ensemble**: Combines GraphSAGE representations with an **Isolation Forest** autoencoder anomaly filter."
        )
    elif any(kw in msg_lower for kw in ["ab test", "experiment", "cupac"]):
        ans = (
            "### 🧪 Nexus-XP: Experimentation & CUPAC\n"
            "- **CUPAC Variance Reduction**: Utilizes pre-experiment data to covariate-adjust outcome metrics.\n"
            "- **Alpha Spending**: Implements sequential analysis with error spending functions.\n"
            "- **Interleaving**: Evaluates search relevance in real-time by blending baseline and candidate results."
        )
    else:
        ans = (
            "### 🪐 Nexus OS Core Console v1.0.0\n"
            "Welcome, operator. The marketplace intelligence system is active and monitoring traffic.\n\n"
            f"Query parsed: *\"{msg}\"*\n\n"
            "Nexus OS is powered by Apache Flink, Redis, PyTorch (Two-Tower and MMoE), LightGBM, and Ray.\n"
            "You can type `/help` to view executable terminal commands or ask specific technical questions about my microservices (e.g. *\"How does the recommender work?\"*)."
        )
    return ChatResponse(response=ans)


@app.get("/")
def root():
    return {"status": "online", "service": "Nexus Inference Gateway (Demo)", "version": "1.0.0-demo"}


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    print(f"\n[*] Nexus Demo API Server starting on http://localhost:{port}")
    print(f"[*] Swagger UI: http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
