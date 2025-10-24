# syntax=docker/dockerfile:1.7-labs
ARG ENABLE_GPU=0
FROM python:3.11-slim AS base
ARG ENABLE_GPU
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 curl build-essential g++ gcc make \
    wget gnupg ca-certificates \
    # Playwright dependencies
    libnss3 libnspr4 libdbus-1-3 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libxkbcommon0 libxcomposite1 libxdamage1 libxfixes3 \
    libxrandr2 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    fonts-unifont fonts-liberation && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install Playwright browsers for JavaScript rendering
RUN playwright install chromium
COPY . .
CMD ["bash", "-lc", "gunicorn -k uvicorn.workers.UvicornWorker -c uvicorn_gunicorn_config.py app.main:app"]
