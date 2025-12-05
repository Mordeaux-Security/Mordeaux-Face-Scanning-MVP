FROM python:3.11-slim

WORKDIR /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY sites.txt ./

# Copy .env if it exists (optional, won't fail if missing)
COPY .env* ./

# Set PYTHONPATH
ENV PYTHONPATH=/app/src

# Default command (can be overridden)
CMD ["python", "-m", "diabetes_crawler.main", "--sites-file", "sites.txt"]

