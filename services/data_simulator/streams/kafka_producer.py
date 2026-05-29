#!/usr/bin/env python3
"""
=============================================================================
Nexus Real-Time Event Producer
Simulates live website traffic and publishes JSON events to Kafka/Redpanda 
for the Apache Flink streaming materializer to consume.
=============================================================================
"""

import json
import time
import random
from datetime import datetime
import uuid

class MockKafkaProducer:
    """
    Mock wrapper representing a confluent_kafka Producer.
    (Used for local simulation without requiring a heavy JVM broker).
    """
    def produce(self, topic: str, key: str, value: str):
        print(f"[Kafka -> {topic}] Key: {key} | Payload: {value}")

def run_stream_simulator(events_per_second: int = 5):
    print("\n" + "="*80)
    print("      NEXUS SIMULATOR: KAFKA REAL-TIME STREAM GENERATOR")
    print("="*80)
    
    producer = MockKafkaProducer()
    topic = "nexus.user.interactions"
    
    event_types = ["view", "view", "view", "cart", "purchase"]
    
    try:
        print(f"[*] Streaming {events_per_second} events/sec to Kafka... (Press Ctrl+C to stop)")
        while True:
            user_id = f"user_{random.randint(1, 500)}"
            item_id = f"item_{random.randint(1, 1000)}"
            
            payload = {
                "event_id": str(uuid.uuid4()),
                "user_id": user_id,
                "item_id": item_id,
                "event_type": random.choice(event_types),
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            
            producer.produce(topic, key=user_id, value=json.dumps(payload))
            time.sleep(1.0 / events_per_second)
            
    except KeyboardInterrupt:
        print("\n[+] Stream interrupted by user. Shutting down producer.")

if __name__ == "__main__":
    run_stream_simulator()