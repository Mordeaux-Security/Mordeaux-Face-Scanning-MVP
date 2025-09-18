FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the local py-common package
COPY ../../packages/py-common /tmp/py-common
RUN pip install /tmp/py-common

# Copy application code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash worker
USER worker

CMD ["python", "worker.py"]
