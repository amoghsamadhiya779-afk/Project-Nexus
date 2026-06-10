FROM python:3.10-slim

WORKDIR /app

# Install essential system build libraries for LightGBM/SciPy/PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy build config and install dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy source code
COPY . .

# Expose default Hugging Face Spaces port
EXPOSE 7860

# Run the unified inference gateway service
CMD ["uvicorn", "services.serving.gateway.app:app", "--host", "0.0.0.0", "--port", "7860"]
