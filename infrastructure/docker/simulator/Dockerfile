# =============================================================================
# Nexus Data Simulator - Production Dockerfile
# Containerizes the vectorized NumPy engine to generate 10M+ events in isolation.
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN addgroup --system nexus && adduser --system --group nexus

WORKDIR /app

# Install fast numerical processing libraries
RUN pip install --no-cache-dir numpy pandas pyarrow

# Copy simulator logic
COPY ./services/data_simulator ./services/data_simulator
COPY ./shared ./shared

# Create output data directory with proper permissions
RUN mkdir -p /data/features/offline && chown -R nexus:nexus /data /app

USER nexus

# Default command: Generate 1M events and output to parquet
ENTRYPOINT ["python", "-m", "services.data_simulator.main"]
CMD ["generate", "--n-events", "1000000", "--n-users", "10000", "--n-items", "5000", "--output-dir", "/data/features/offline", "--to-parquet"]