from fastapi import FastAPI, HTTPException
import redis
import uvicorn
from shared.utils.config import config

app = FastAPI(title="Nexus Feature Ingress/Egress Gateway", version="1.0.0")

# Redis Connector instance
r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)

@app.get("/health")
def health():
    try:
        r.ping()
        return {"status": "healthy", "redis": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "redis": str(e)}

@app.get("/features/user/{user_id}")
def get_user_features(user_id: str):
    key = f"fv:user_aggregates:user:{user_id}"
    features = r.hgetall(key)
    if not features:
        # Generate default cold-start profile for missing keys
        return {
            "user_id": user_id,
            "user_view_count": "0",
            "user_purchase_count": "0",
            "user_conversion_rate": "0.0"
        }
    features["user_id"] = user_id
    return features

@app.get("/features/item/{item_id}")
def get_item_features(item_id: str):
    key = f"fv:item_aggregates:item:{item_id}"
    features = r.hgetall(key)
    if not features:
        return {
            "item_id": item_id,
            "category": "unknown",
            "base_price": "0.0",
            "popularity": "0"
        }
    features["item_id"] = item_id
    return features

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
