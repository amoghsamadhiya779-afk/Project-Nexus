# =============================================================================
# Nexus Serving Gateway - Production Dockerfile
# Optimized multi-stage build, non-root user, and cached dependency layers.
# =============================================================================

# Stage 1: Builder
FROM python:3.11-slim as builder

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (Leveraging Docker layer caching)
COPY pyproject.toml .
RUN pip install --upgrade pip && \
    pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels -e .

# Install specific serving dependencies directly to wheels
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /build/wheels fastapi uvicorn redis torch numpy lightgbm

# Stage 2: Runtime
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Create a non-root user for enterprise security compliance
RUN addgroup --system nexus && adduser --system --group nexus

WORKDIR /app

# Copy compiled wheels from builder and install
COPY --from=builder /build/wheels /wheels
RUN pip install --no-cache /wheels/*

# Copy the necessary application code
COPY ./services/serving ./services/serving
COPY ./services/recommender ./services/recommender
COPY ./services/search ./services/search
COPY ./shared ./shared

# Ensure the model export directory exists and has correct permissions
RUN mkdir -p /app/models_export && chown -R nexus:nexus /app

USER nexus

EXPOSE 8080

# Launch the asynchronous FastAPI Uvicorn server
CMD ["python", "-m", "uvicorn", "services.serving.gateway.app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]