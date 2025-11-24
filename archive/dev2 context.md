## Dev C – Local Context (this machine)

### System
- OS: Windows 10 (build 26200)
- Shell: PowerShell
- Python: 3.12.10 (venv created at `./venv`)
- Docker Desktop: available (docker compose works)

### Environment Files
- Created `.env.example` (root) with all required variables
- Generated `.env` (UTF-8, fixed encoding issue) from example

### Docker Work
- Updated `docker-compose.yml`
  - Added service `face-pipeline` (port 8001→8000 in container)
  - Ensured healthchecks, volumes, and dependencies
- Backend image: uses existing `backend/Dockerfile`
- Worker image: uses existing `worker/Dockerfile`
- Frontend image: uses existing `frontend/Dockerfile`
- Nginx: updated routing in `nginx/default.conf`
  - `location /` -> frontend
  - `location /api/` -> backend
  - `location /face-pipeline/` -> face-pipeline

### New/Updated Files (created by Dev C)
- `.env.example` – comprehensive environment template
- `face-pipeline/Dockerfile` – completed, installs deps, healthcheck, runs `main.py`
- `docker-compose.yml` – added `face-pipeline` service (+ healthcheck/ports)
- `nginx/default.conf` – added `/face-pipeline/` proxy block
- `build-docker.ps1` – Windows build/run helper (start/stop/status/logs/cleanup)
- `build-docker.sh` – Linux/Mac build/run helper
- `DOCKER_README.md` – end-to-end Docker guide + troubleshooting
- `Makefile` – convenience targets for compose, logs, health, shells, db helpers
- `setup-local.ps1` – local setup (venv, deps, models, frontend, docker verify)
- `setup-local.sh` – same as above for Linux/Mac
- `setup-local-windows.ps1` – Windows-specific setup with InsightFace handling

### Commands Executed (high-level)
- Validated compose: `docker-compose config --quiet`
  - Fixed `.env` encoding (recreated as UTF-8)
- Tested scripts: `.\\build-docker.ps1 help`
- Attempted local venv install via `setup-local.ps1`
  - InsightFace wheel build failed due to missing MSVC build tools (expected on Windows)
  - Provided Windows flow in `setup-local-windows.ps1` with Docker-only option

### Known Local Constraints / Notes
- InsightFace on Windows may require Microsoft C++ Build Tools. Workarounds provided:
  - Use Docker for full functionality
  - Or install Build Tools (link in `setup-local-windows.ps1`), then re-run setup
- `.env` must be UTF-8 (PowerShell copying can introduce BOM issues)

### How to Run (current local state)
- Docker (recommended):
  - PowerShell: `.\\build-docker.ps1` (starts all services)
  - Manual: `docker-compose up --build -d`
- Local (limited if InsightFace not installed):
  - PowerShell: `.\\setup-local-windows.ps1` (choose options)
  - Activate: `.\\venv\\Scripts\\Activate.ps1`
  - Backend: `cd backend && uvicorn app.main:app --reload`
  - Face Pipeline: `cd face-pipeline && python main.py`

### Service URLs (after Docker up)
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- Face Pipeline: http://localhost:8001
- MinIO Console: http://localhost:9001
- pgAdmin: http://localhost:5050
- Qdrant: http://localhost:6333
- Nginx (main): http://localhost:80

### Next Suggestions
- If Windows-native dev needed with InsightFace, install MSVC Build Tools and re-run `setup-local-windows.ps1` without `-UseDockerOnly`
- Seed demo: after containers up, run `docker compose exec backend-cpu python /app/seed_demo.py`


