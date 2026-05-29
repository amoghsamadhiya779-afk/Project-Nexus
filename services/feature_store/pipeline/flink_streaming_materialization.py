#!/usr/bin/env python3
"""
=============================================================================
Nexus Feature Store: Apache Flink Streaming Materialization
Inspired by DoorDash's Riviera. Attaches to a Kafka streaming bus, processes 
live user interaction events in real-time, and applies stateful updates 
directly to the Redis Online Store.
=============================================================================
"""

import json
import redis
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.datastream.functions import MapFunction
from pyflink.common.typeinfo import Types

# Configuration
KAFKA_BROKER = "localhost:9092"
KAFKA_TOPIC = "nexus.user.interactions"
REDIS_HOST = "localhost"
REDIS_PORT = 6379

class RedisStateUpdater(MapFunction):
    """
    A Flink MapFunction that processes incoming Kafka JSON events 
    and applies incremental state updates to Redis.
    """
    def __init__(self):
        self.r = None

    def open(self, runtime_context):
        """Initializes the Redis connection when the Flink TaskManager starts."""
        pool = redis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        self.r = redis.Redis(connection_pool=pool)

    def map(self, value):
        """Processes a single real-time event from the Kafka stream."""
        try:
            event = json.loads(value)
            user_id = event.get("user_id")
            event_type = event.get("event_type")
            
            if not user_id or not event_type:
                return value
                
            redis_key = f"fv:user_aggregates:user:{user_id}"
            
            # Use Redis atomic increments to update state safely in real-time
            if event_type == "view":
                self.r.hincrby(redis_key, "user_view_count", 1)
            elif event_type == "purchase":
                self.r.hincrby(redis_key, "user_purchase_count", 1)
                
            # Note: In a true production Flink job, CVR calculation would be 
            # handled via Flink stateful aggregations (ValueState). For simplicity 
            # here, we rely on the batch job to correct the CVR ratio nightly.
            
            print(f"[Streaming] Processed {event_type} event for {user_id}")
            return value
            
        except Exception as e:
            print(f"[Streaming Error] Failed to process payload: {e}")
            return value

def run_flink_pipeline():
    print("\n" + "="*80)
    print("     NEXUS MLOPS: APACHE FLINK REAL-TIME STREAMING PIPELINE")
    print("="*80)
    
    # Initialize Flink Execution Environment
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1) # Set to 1 for local testing readability
    
    print(f"[*] Attaching to Kafka broker at {KAFKA_BROKER} | Topic: {KAFKA_TOPIC}")
    print("[*] Waiting for live events...")

    # NOTE: To run this against a real Kafka cluster locally, you must add the 
    # Flink-Kafka connector JAR to your PyFlink installation. 
    # For this portfolio demonstration, we simulate the DataStream source.
    
    # Simulated Kafka Stream Source
    simulated_kafka_stream = env.from_collection([
        '{"user_id": "user_100", "event_type": "view", "timestamp": "2026-05-29T10:00:00Z"}',
        '{"user_id": "user_100", "event_type": "purchase", "timestamp": "2026-05-29T10:05:00Z"}',
        '{"user_id": "user_404", "event_type": "view", "timestamp": "2026-05-29T10:06:00Z"}',
    ], type_info=Types.STRING())
    
    # Apply the Redis State Updater Map Function
    simulated_kafka_stream.map(RedisStateUpdater(), output_type=Types.STRING())
    
    # Execute the streaming topology
    env.execute("Nexus_Flink_Streaming_Materializer")

if __name__ == "__main__":
    run_flink_pipeline()