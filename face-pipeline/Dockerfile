# Face Pipeline Dockerfile - Optimized Multi-Stage Build
FROM python:3.11-slim as builder

# Install build dependencies for InsightFace compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage - minimal runtime image
FROM python:3.11-slim as production

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libopencv-core-dev \
    libopencv-imgproc-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    INSIGHTFACE_HOME=/models/insightface \
    ONNX_PROVIDERS_CSV=CPUExecutionProvider \
    DET_SCORE_THRESH=0.20 \
    DET_SIZE=1280,1280

WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY . .

# Create model cache directory
RUN mkdir -p /models/insightface

# Warm model cache - download and cache models during build
# This pre-downloads InsightFace models so container starts are instant
RUN python scripts/warm_models.py

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; from pipeline.embedder import load_model; load_model(); sys.exit(0)"

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]

