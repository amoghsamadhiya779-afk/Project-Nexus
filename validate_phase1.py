#!/usr/bin/env python3
"""
Nexus Platform - Phase 1 Feature Store API Validator
Loads a real generated UUID from the local Parquet file, queries the active 
FastAPI serving gateway, and benchmarks the round-trip latency.
"""

import json
import urllib.request
import urllib.error
import time
from pathlib import Path

# Load settings from central config
try:
    from shared.utils.config import config
    parquet_path = Path(config.BASE_DATA_DIR) / "features/offline/historical_interactions.parquet"
except ImportError:
    # Fallback to default path if imports are not fully configured
    parquet_path = Path("C:/data/features/offline/historical_interactions.parquet")

def validate_serving():
    print("=====================================================================")
    print("          NEXUS FEATURE STORE API VALIDATION UTILITY                 ")
    print("=====================================================================")

    # 1. Check if Parquet file exists
    if not parquet_path.exists():
        print(f"[❌] Offline Parquet dataset not found at: {parquet_path}")
        print("     Please run the data simulator first to generate interaction logs.")
        return

    # 2. Extract real active UUIDs using Pandas
    try:
        import pandas as pd
        print("[*] Reading simulated Parquet database to extract active UUIDs...")
        df = pd.read_parquet(parquet_path)
        
        # Get first non-null active user and item UUIDs
        active_user_id = str(df["user_id"].dropna().iloc[0])
        active_item_id = str(df["item_id"].dropna().iloc[0])
        
        print(f"[+] Found Active User UUID: {active_user_id}")
        print(f"[+] Found Active Item UUID: {active_item_id}\\n")
    except Exception as e:
        print(f"[❌] Failed to inspect Parquet file: {e}")
        return

    # 3. Test API endpoints with real IDs
    for entity_type, entity_id in [("user", active_user_id), ("item", active_item_id)]:
        url = f"http://localhost:8000/features/{entity_type}/{entity_id}"
        print(f"[*] Querying API: {url}")
        
        start_time = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000
                data = json.loads(response.read().decode())
                
                print(f"  [+] Status: 200 OK")
                print(f"  [+] Latency: {latency_ms:.2f} ms")
                print("  [+] Returned Payload:")
                print(json.dumps(data, indent=4))
                print("-" * 69)
        except urllib.error.URLError as e:
            print(f"  [❌] HTTP Request failed: {e}")
            print("       Make sure your API server is running on port 8000!")
            print("       Start it with: python -m services.feature_store.api.server")
            return

    print("\\n🎉 Phase 1 Validation Complete! Your Feature Store is active and serving data perfectly.")

if __name__ == "__main__":
    validate_serving()
