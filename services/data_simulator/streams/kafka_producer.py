"""
services/data_simulator/streams/kafka_producer.py
===================================================
Kafka/Redpanda producer for streaming synthetic events.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from confluent_kafka import Producer
from loguru import logger


class NexusKafkaProducer:
    def __init__(self, brokers: str = "localhost:19092"):
        self._producer = Producer({
            "bootstrap.servers":           brokers,
            "queue.buffering.max.messages": 500_000,
            "queue.buffering.max.kbytes":   512_000,
            "batch.num.messages":           10_000,
            "linger.ms":                   100,
            "compression.type":            "lz4",
        })

    def _delivery_report(self, err, msg):
        if err:
            logger.warning(f"Kafka delivery failed: {err}")

    def send_event(self, topic: str, key: str, value: Dict[str, Any]) -> None:
        def default_serialiser(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")

        self._producer.produce(
            topic,
            key=key.encode(),
            value=json.dumps(value, default=default_serialiser).encode(),
            callback=self._delivery_report,
        )
        self._producer.poll(0)   # non-blocking

    def send_events(self, df: pd.DataFrame) -> None:
        for _, row in df.iterrows():
            self.send_event(
                topic="user-interactions",
                key=str(row["user_id"]),
                value=row.to_dict(),
            )

    def flush(self) -> None:
        remaining = self._producer.flush(timeout=30)
        if remaining > 0:
            logger.warning(f"{remaining} messages still in Kafka queue after flush")

    def close(self) -> None:
        self.flush()
