# syntax=docker/dockerfile:1.7-labs
ARG ENABLE_GPU=0
FROM python:3.11-slim AS base
ARG ENABLE_GPU
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl build-essential g++ gcc make \
    wget gnupg ca-certificates \
    # Playwright dependencies
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    fonts-unifont fonts-liberation && \
    rm -rf /var/lib/apt/lists/*

# GPU-specific dependencies
RUN if [ "$ENABLE_GPU" = "1" ]; then \
    # Install CUDA dependencies (for NVIDIA)
    apt-get update && apt-get install -y --no-install-recommends \
    nvidia-cuda-toolkit \
    libcudnn8 \
    libcudnn8-dev && \
    rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install GPU-specific Python packages if enabled
RUN if [ "$ENABLE_GPU" = "1" ]; then \
    # Install GPU-accelerated packages
    pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    onnxruntime-gpu==1.19.2 \
    opencv-contrib-python==4.10.0.84; \
    fi

# Install Playwright browsers for JavaScript rendering
RUN playwright install chromium

COPY . .

# Set GPU environment variables if enabled
RUN if [ "$ENABLE_GPU" = "1" ]; then \
    echo "export CUDA_VISIBLE_DEVICES=0" >> /etc/environment; \
    echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512" >> /etc/environment; \
    fi

CMD ["bash", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker -c uvicorn_gunicorn_config.py app.main:app"]
