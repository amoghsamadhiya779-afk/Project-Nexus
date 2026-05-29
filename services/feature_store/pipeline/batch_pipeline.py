import os
import pandas as pd
import redis
from services.feature_store.core.entity import Entity
from services.feature_store.core.source import BatchSource
from services.feature_store.core.feature_view import FeatureView, Feature, FeatureRegistry
from shared.utils.config import config

def main():
    print("[*] Running feature aggregation and materialization pipeline...")
    
    # Establish connection to Redis Online Store
    try:
        r = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)
        r.ping()
        print("[+] Online feature cache (Redis) connection successful.")
    except Exception as e:
        print(f"[❌] Redis connection failed: {e}. Ensure docker compose is running.")
        return

    # Define registry entities and schemas
    user_entity = Entity(name="user", join_key="user_id", description="System customer")
    item_entity = Entity(name="item", join_key="item_id", description="Product listing")
    
    user_fv = FeatureView(
        name="user_aggregates",
        entities=[user_entity],
        features=[
            Feature(name="user_view_count", value_type="float"),
            Feature(name="user_purchase_count", value_type="float"),
            Feature(name="user_conversion_rate", value_type="float")
        ],
        batch_source=BatchSource(path=f"{config.BASE_DATA_DIR}/features/offline/historical_interactions.parquet")
    )
    
    registry = FeatureRegistry()
    registry.register_feature_view(user_fv)
    
    # Process offline source Parquet log aggregates
    parquet_path = f"{config.BASE_DATA_DIR}/features/offline/historical_interactions.parquet"
    if not os.path.exists(parquet_path):
        print(f"[❌] Source file missing: {parquet_path}. Run data simulator first.")
        return

    df = pd.read_parquet(parquet_path)
    print(f"[*] Processing {len(df)} interaction aggregates...")

    # Calculate analytical features per user
    user_groups = df.groupby("user_id")
    for user_id, group in user_groups:
        total_views = int((group["event_type"] == "view").sum())
        total_purchases = int((group["event_type"] == "purchase").sum())
        cvr = float(total_purchases / total_views) if total_views > 0 else 0.0

        # Load into Redis Hash Maps
        redis_key = f"fv:user_aggregates:user:{user_id}"
        r.hset(redis_key, mapping={
            "user_view_count": str(total_views),
            "user_purchase_count": str(total_purchases),
            "user_conversion_rate": f"{cvr:.4f}"
        })

    # Calculate analytical features per item
    print("[*] Materializing item category attributes...")
    item_groups = df.groupby("item_id")
    for item_id, group in item_groups:
        category = group["category"].iloc[0]
        price = float(group["price"].iloc[0])
        total_views = int((group["event_type"] == "view").sum())
        
        redis_key = f"fv:item_aggregates:item:{item_id}"
        r.hset(redis_key, mapping={
            "category": category,
            "base_price": str(price),
            "popularity": str(total_views)
        })

    print(f"[+] Successfully materialized {len(user_groups)} user feature maps and {len(item_groups)} item feature maps to Redis.")

if __name__ == "__main__":
    main()
