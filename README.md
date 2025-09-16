# Face Scanning Protection (MVP)

## Overview
This project helps users detect where their face appears online, lawfully and safely.

## Structure
- `/api` – FastAPI service for search endpoints
- `/pipeline` – Face detection + embedding workers
- `/search` – Pinecone indexing & query helpers
- `/ui` – Web frontend
- `/ops` – Deployment, configs, infra scripts
- `/docs` – Design docs, policies, decisions
- `/data` – Local dev images (ignored in git)

## Getting Started
1. Copy `.env.example` → `.env` and fill in Pinecone/AWS keys.
2. Run local stack:
   ```bash
   docker compose up