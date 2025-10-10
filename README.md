# Face Scanning Protection (MVP)

## Overview
This project helps users detect where their face appears online, lawfully and safely.

## Structure
- `/backend` – FastAPI service with face detection and search endpoints
- `/worker` – Celery worker for face detection + embedding tasks
- `/frontend` – Web frontend
- `/docs` – Design docs, policies, decisions
- `/data` – Local dev images (ignored in git)

## Getting Started
1. Copy `.env.example` → `.env` and fill in Pinecone/AWS keys.
2. Run local stack:
   ```bash
   docker compose up