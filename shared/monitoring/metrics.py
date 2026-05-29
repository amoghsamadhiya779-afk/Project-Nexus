from prometheus_client import Counter, Histogram

# Initialize standard metrics counters
REQUEST_COUNT = Counter(
    "nexus_api_requests_total",
    "Total API requests received by the serving layer",
    ["endpoint"]
)

LATENCY_HISTOGRAM = Histogram(
    "nexus_api_latency_milliseconds",
    "API gateway serving execution latency distribution (ms)",
    ["endpoint"],
    buckets=[1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]
)

def track_latency(endpoint: str, latency_ms: float):
    """Utility function to track operational metrics outside of app wrappers."""
    REQUEST_COUNT.labels(endpoint=endpoint).inc()
    LATENCY_HISTOGRAM.labels(endpoint=endpoint).observe(latency_ms)
