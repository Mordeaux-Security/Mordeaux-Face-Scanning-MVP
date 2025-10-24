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

### CPU Setup (Default)
The default setup uses CPU-only processing, which is suitable for most development and testing scenarios.

1. Copy `.env.example` → `.env` and fill in Pinecone/AWS keys.
2. Run local stack:
   ```bash
   # PowerShell (Windows)
   .\start-local.ps1
   
   # Bash (Linux/Mac)
   ./start-local.sh
   
   # Or directly with docker-compose
   docker-compose up --build -d
   ```

### GPU Setup (Optional)
For GPU acceleration, use the GPU-specific scripts:

```bash
# PowerShell (Windows)
.\start-gpu.ps1

# Bash (Linux/Mac)  
./start-gpu.sh
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Toolkit
- `nvidia-smi` command available

### Configuration Details
- **Default**: CPU-only processing (`ENABLE_GPU=0`)
- **GPU**: CUDA acceleration (`ENABLE_GPU=1`)
- **Docker Compose Files**:
  - `docker-compose.yml` - CPU default
  - `docker-compose.gpu.yml` - GPU acceleration
  - `docker-compose.gpu-simple.yml` - Simplified GPU setup