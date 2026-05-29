#!/usr/bin/env python3
"""
=============================================================================
Nexus Feature Store: PySpark Batch Materialization
Inspired by Airbnb's Zipline architecture. Processes massive offline Parquet 
datasets, computes historical aggregates, and bulk-loads them into the Redis 
Online Store for sub-5ms low-latency serving.
=============================================================================
"""

import os
import json
import redis
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

# Configuration
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
PARQUET_PATH = "C:/data/features/offline/historical_interactions.parquet"

def write_partition_to_redis(partition):
    """
    Executes a pipelined Redis write for a single Spark partition.
    This avoids opening a new Redis connection for every single row.
    """
    # Initialize connection pool per partition (worker node)
    pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    pipe = r.pipeline(transaction=False)
    
    count = 0
    for row in partition:
        user_id = row['user_id']
        
        # Construct the hash map for Redis
        feature_map = {
            "user_view_count": str(row['view_count']),
            "user_purchase_count": str(row['purchase_count']),
            "user_conversion_rate": str(round(row['conversion_rate'], 4))
        }
        
        redis_key = f"fv:user_aggregates:user:{user_id}"
        pipe.hset(redis_key, mapping=feature_map)
        
        count += 1
        # Execute batch in chunks of 1000 to optimize TCP network throughput
        if count % 1000 == 0:
            pipe.execute()
            
    # Flush any remaining items in the pipeline
    if count % 1000 != 0:
        pipe.execute()

def run_batch_pipeline():
    print("\n" + "="*80)
    print("      NEXUS MLOPS: PYSPARK BATCH FEATURE MATERIALIZATION")
    print("="*80)
    
    # Initialize Spark Session configured for local/standalone execution
    spark = SparkSession.builder \
        .appName("Nexus_Batch_Feature_Materializer") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
        
    print(f"[*] Spark Session Initialized. Reading Parquet from: {PARQUET_PATH}")
    
    try:
        # Load the raw offline interactions
        df = spark.read.parquet(PARQUET_PATH)
        
        # Distributed Aggregation: Compute views, purchases, and CVR per user
        aggregated_df = df.groupBy("user_id").agg(
            F.sum(F.when(F.col("event_type") == "view", 1).otherwise(0)).alias("view_count"),
            F.sum(F.when(F.col("event_type") == "purchase", 1).otherwise(0)).alias("purchase_count")
        ).withColumn(
            "conversion_rate", 
            F.when(F.col("view_count") > 0, F.col("purchase_count") / F.col("view_count")).otherwise(0.0)
        )
        
        total_users = aggregated_df.count()
        print(f"[*] Computed aggregates for {total_users} unique entities.")
        print("[*] Initiating distributed bulk-load to Redis Online Store...")
        
        # Distribute the Redis writing process across Spark worker nodes
        aggregated_df.rdd.foreachPartition(write_partition_to_redis)
        
        print("[+] Batch Materialization Complete! Redis is fully hydrated.")
        
    except Exception as e:
        print(f"[❌] Spark Job Failed. Ensure the Parquet data exists. Error: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    run_batch_pipeline()