# =============================================================================
# Nexus Feature Store Batch Pipeline - Production Dockerfile
# Executes the historical log aggregation and Redis materialization pipeline.
# =============================================================================

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

RUN addgroup --system nexus && adduser --system --group nexus

WORKDIR /app

# Install data processing and Redis client libraries
RUN pip install --no-cache-dir pandas pyarrow redis pydantic pydantic-settings

# Copy core feature store logic
COPY ./services/feature_store ./services/feature_store
COPY ./shared ./shared

USER nexus

# Default command: Run the batch materialization pipeline
CMD ["python", "-m", "services.feature_store.pipeline.batch_pipeline"]