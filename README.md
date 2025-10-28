# Face Scanning Protection (MVP)

## Overview
This project helps users detect where their face appears online, lawfully and safely.

## Structure
- `/backend` – FastAPI service with face detection and search endpoints
- `/worker` – Celery worker for face detection + embedding tasks
- `/face-pipeline` – **NEW**: Redis Streams-based face processing pipeline with global deduplication
- `/frontend` – Web frontend
- `/docs` – Design docs, policies, decisions
- `/data` – Local dev images (ignored in git)

## Face Pipeline (Phase 3)

The `/face-pipeline` directory contains a production-ready face processing service with:

### Key Features
- **Global Deduplication**: Redis-based pHash deduplication prevents duplicate processing
- **Queue Worker**: Redis Streams-based async message processing with `--once` testing mode
- **Performance Metrics**: Real-time timing and counter metrics via Redis hashes
- **Qdrant Integration**: Vector database with payload indexes for fast filtering
- **Comprehensive Health Checks**: Dependency monitoring for all services

### Quick Start
```bash
# Start the face pipeline service
cd face-pipeline
python main.py

# Run worker in single-batch mode (testing)
python worker.py --once

# Run worker in production mode
python worker.py --max-batch 16

# Publish test message
python scripts/publish_test_message.py
```

### Documentation
- **[PHASE3_RUNBOOK.md](face-pipeline/PHASE3_RUNBOOK.md)** - Operations guide for running, testing, and troubleshooting the queue worker
- **[PHASE3_IMPLEMENTATION.md](face-pipeline/PHASE3_IMPLEMENTATION.md)** - Technical implementation details and architecture
- **[README.md](face-pipeline/README.md)** - Developer documentation and API contracts

## Getting Started
1. Copy `.env.example` → `.env` and fill in Pinecone/AWS keys.
2. Run local stack:
   ```bash
   docker compose up