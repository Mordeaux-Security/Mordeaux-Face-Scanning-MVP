# Windows GPU Worker Setup

## Overview
This GPU worker runs natively on Windows to provide DirectML-accelerated face detection for the Docker-based crawler system.

## Prerequisites

1. **Windows 10/11** with DirectX 12 compatible GPU
2. **Python 3.11** or higher
3. **DirectML-compatible GPU** (AMD, NVIDIA, or Intel)

## Installation

### 1. Install Python Dependencies

Open PowerShell in the `backend/gpu_worker/` directory and run:

```powershell
pip install -r requirements.txt
```

### 2. Verify DirectML Availability

```powershell
python -c "import onnxruntime as ort; print('DirectML Available:', 'DmlExecutionProvider' in ort.get_available_providers())"
```

Expected output: `DirectML Available: True`

## Running the GPU Worker

### Option 1: Using Python Launcher (Recommended)

```powershell
python launch.py
```

This launcher provides:
- Automatic dependency checking
- Health monitoring with auto-restart
- Process management with single-instance enforcement
- Detailed logging to `gpu_worker.log`

### Option 2: Using Batch File

```powershell
launch.bat
```

Simple batch launcher that calls `launch.py`.

### Option 3: Direct Worker Launch

```powershell
python worker.py
```

Runs the worker directly without monitoring (not recommended for production).

## Configuration

The worker listens on **port 8765** by default. This matches the Docker configuration which expects the GPU worker at `http://host.docker.internal:8765`.

To change the port, edit `worker.py` line 736:
```python
uvicorn.run(app, host="0.0.0.0", port=8765)
```

And update `launch.py` line 57:
```python
WORKER_PORT = 8765
```

## Verifying Connection

Once the GPU worker is running, verify it's accessible from Docker:

```powershell
# From Windows host
curl http://localhost:8765/health

# From inside Docker container
docker-compose exec new-crawler curl http://host.docker.internal:8765/health
```

Expected response:
```json
{
  "status": "healthy",
  "gpu_available": true,
  "directml_available": true,
  "model_loaded": true,
  "uptime_seconds": 123.45
}
```

## Monitoring

### Logs
Check `gpu_worker.log` for detailed activity logs.

### Health Endpoint
```powershell
curl http://localhost:8765/health
```

### GPU Info
```powershell
curl http://localhost:8765/gpu_info
```

## Troubleshooting

### Port 8765 Already in Use
```powershell
# Find process using port 8765
netstat -ano | findstr :8765

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F
```

### DirectML Not Available
- Ensure GPU drivers are up to date
- Verify Windows 10/11 version supports DirectX 12
- Try reinstalling `onnxruntime-directml`:
  ```powershell
  pip uninstall onnxruntime-directml
  pip install onnxruntime-directml
  ```

### Worker Crashes on Startup
- Check `gpu_worker.log` for error messages
- Verify all dependencies are installed: `python launch.py` will check automatically
- Ensure sufficient disk space for model downloads (~200MB in `~/.insightface`)

## Integration with Docker Crawler

The Docker crawler is pre-configured to connect to the Windows GPU worker at `http://host.docker.internal:8765`. No additional Docker configuration is required once the GPU worker is running on Windows.

To start the full system:

1. Launch GPU worker on Windows: `python launch.py`
2. Start Docker services: `docker-compose up -d`
3. Run a crawl: `docker-compose exec new-crawler python -m new_crawler.main --sites-file /app/new_crawler/example_sites.txt`

The crawler will automatically detect and use the GPU worker for face detection.

