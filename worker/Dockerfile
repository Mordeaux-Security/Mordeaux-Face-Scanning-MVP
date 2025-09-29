# syntax=docker/dockerfile:1.7-labs
ARG ENABLE_GPU=0
FROM python:3.11-slim AS base
ARG ENABLE_GPU
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 build-essential g++ gcc make && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["celery", "-A", "worker.worker", "worker", "--loglevel=INFO"]
