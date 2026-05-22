"""
shared/monitoring/metrics.py
=============================
Centralised Prometheus metrics registry.
All services import their relevant metric objects from here.
"""
from prometheus_client import Counter, Histogram, Gauge, Summary

# ── Feature Store ──────────────────────────────────────────────────────────────
class FeatureStoreMetrics:
    online_read_latency   = Histogram(
        "nexus_feature_store_online_read_latency_ms",
        "Redis feature fetch latency in ms",
        buckets=[0.5, 1, 2, 5, 10, 20, 50, 100],
    )
    online_read_batch_size = Histogram(
        "nexus_feature_store_online_read_batch_size",
        "Number of keys per batch read",
        buckets=[1, 5, 10, 20, 50, 100, 200, 500],
    )
    materialise_total = Counter(
        "nexus_feature_store_materialise_total",
        "Total feature materialisation runs",
        ["feature_view", "status"],
    )
    online_keys = Gauge(
        "nexus_feature_store_online_keys_total",
        "Approximate number of keys in online store",
    )
    cache_hit_rate = Gauge(
        "nexus_feature_store_cache_hit_rate",
        "Feature cache hit rate",
        ["feature_view"],
    )

feature_store_metrics = FeatureStoreMetrics()

# ── Serving Gateway ────────────────────────────────────────────────────────────
class ServingMetrics:
    request_latency = Histogram(
        "nexus_serving_request_latency_ms",
        "End-to-end serving latency in ms",
        ["endpoint"],
        buckets=[10, 25, 50, 75, 100, 150, 200, 500, 1000],
    )
    request_total = Counter(
        "nexus_serving_requests_total",
        "Total serving requests",
        ["endpoint", "status"],
    )
    candidate_pool_size = Histogram(
        "nexus_serving_candidate_pool_size",
        "Size of candidate pool before ranking",
        buckets=[10, 50, 100, 200, 500],
    )

serving_metrics = ServingMetrics()

# ── Recommender ────────────────────────────────────────────────────────────────
class RecommenderMetrics:
    inference_latency = Histogram(
        "nexus_recommender_inference_latency_ms",
        "Model inference latency in ms",
        ["model"],
        buckets=[1, 5, 10, 25, 50, 100],
    )
    faiss_search_latency = Histogram(
        "nexus_recommender_faiss_search_latency_ms",
        "FAISS ANN search latency in ms",
        buckets=[0.5, 1, 2, 5, 10, 20],
    )

recommender_metrics = RecommenderMetrics()

# ── Data Pipeline ──────────────────────────────────────────────────────────────
class PipelineMetrics:
    events_processed = Counter(
        "nexus_pipeline_events_processed_total",
        "Total events processed",
        ["topic"],
    )
    pipeline_lag = Gauge(
        "nexus_pipeline_consumer_lag",
        "Kafka consumer lag",
        ["topic", "partition"],
    )
    data_quality_failures = Counter(
        "nexus_data_quality_failures_total",
        "Data quality check failures",
        ["check_name"],
    )

pipeline_metrics = PipelineMetrics()

# ── Experimentation ────────────────────────────────────────────────────────────
class ExperimentMetrics:
    active_experiments = Gauge(
        "nexus_experiments_active_total",
        "Number of currently running experiments",
    )
    guardrail_violations = Counter(
        "nexus_experiment_guardrail_violations_total",
        "Guardrail metric violations",
        ["experiment_id", "metric"],
    )

experiment_metrics = ExperimentMetrics()
