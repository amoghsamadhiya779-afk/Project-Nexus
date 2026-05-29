#!/usr/bin/env python3
"""
=============================================================================
Prometheus Metrics Exporter Middleware
Exposes a `/metrics` endpoint for Grafana to scrape. Tracks request counts,
latency histograms, and HTTP error rates across the Serving Gateway.
=============================================================================
"""

from fastapi import FastAPI
from prometheus_client import make_asgi_app, Counter, Histogram
import time

# Standard RED (Rate, Errors, Duration) Metrics
HTTP_REQUESTS = Counter(
    "nexus_http_requests_total",
    "Total HTTP requests handled by the API gateway",
    ["method", "endpoint", "http_status"]
)

HTTP_LATENCY = Histogram(
    "nexus_http_latency_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]
)

def add_prometheus_middleware(app: FastAPI):
    """
    Mounts the Prometheus ASGI application and adds middleware to track
    inbound requests automatically.
    """
    # Mount the /metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    @app.middleware("http")
    async def track_prometheus_metrics(request, call_next):
        start_time = time.perf_counter()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        process_time = time.perf_counter() - start_time
        
        # Extract routing information
        endpoint = request.url.path
        method = request.method
        status_code = str(response.status_code)
        
        # Ignore the /metrics endpoint itself to prevent metric noise
        if endpoint != "/metrics":
            HTTP_REQUESTS.labels(method=method, endpoint=endpoint, http_status=status_code).inc()
            HTTP_LATENCY.labels(method=method, endpoint=endpoint).observe(process_time)
            
        return response
        
    print("[+] Prometheus Monitoring Middleware attached to API Gateway (/metrics).")