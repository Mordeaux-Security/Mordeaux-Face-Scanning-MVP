# GPU Worker Setup Guide

This guide covers setting up the Windows GPU worker for AMD GPU acceleration in the Mordeaux face scanning application.

## Overview

The GPU worker is a Windows native service that provides GPU-accelerated face detection using DirectML (Direct Machine Learning) for AMD GPUs. It communicates with the Linux Docker container via HTTP to provide high-performance face detection and embedding.

## Architecture

```
┌─────────────────┐    HTTP     ┌─────────────────┐
│   Linux         │◄────────────►│   Windows       │
│   Container     │   :8765      │   GPU Worker    │
│                 │              │                 │
│ ┌─────────────┐ │              │ ┌─────────────┐ │
│ │ GPU Client  │ │              │ │ DirectML    │ │
│ │ (httpx)     │ │              │ │ InsightFace │ │
│ └─────────────┘ │              │ └─────────────┘ │
│ ┌─────────────┐ │              │                 │
│ │ Face Service│ │              │                 │
│ │ (fallback)  │ │              │                 │
│ └─────────────┘ │              │                 │
└─────────────────┘              └─────────────────┘
```

## Prerequisites

### Windows Requirements

- Windows 10/11 with AMD GPU
- Python 3.8+ with pip
- Docker Desktop for Windows
- PowerShell (for startup scripts)

### GPU Requirements

- AMD GPU with DirectML support
- Latest AMD drivers
- ONNX Runtime DirectML package

## Installation

### 1. Install Dependencies

```powershell
# Navigate to GPU worker directory
cd backend\gpu_worker

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Verify DirectML Support

```python
import onnxruntime as ort
providers = ort.get_available_providers()
print("Available providers:", providers)
# Should include 'DmlExecutionProvider'
```

### 3. Test GPU Worker

```powershell
# Start GPU worker manually
python launch.py

# In another terminal, test the worker
python test_gpu_worker_integration.py
```

## Configuration

### Environment Variables

Create a `.env` file with the following GPU worker settings:

```env
# GPU Worker Configuration
GPU_WORKER_ENABLED=true
GPU_WORKER_URL=http://host.docker.internal:8765
GPU_WORKER_TIMEOUT=60
GPU_WORKER_MAX_RETRIES=5
GPU_WORKER_BATCH_SIZE=32
GPU_WORKER_HEALTH_CHECK_INTERVAL=10

# GPU Settings
GPU_BACKEND=directml
FACE_DETECTION_GPU=true
FACE_EMBEDDING_GPU=true
```

### Docker Configuration

The `docker-compose.yml` has been updated to include GPU worker settings:

```yaml
backend-cpu:
  environment:
    GPU_WORKER_ENABLED: ${GPU_WORKER_ENABLED:-true}
    GPU_WORKER_URL: ${GPU_WORKER_URL:-http://host.docker.internal:8765}
    # ... other settings
  extra_hosts:
    - "host.docker.internal:host-gateway"
```

## Usage

### Quick Start

Use the provided PowerShell script to start everything:

```powershell
# Start GPU worker and Docker services
.\start-gpu-worker.ps1

# Or start components separately
.\start-gpu-worker.ps1 -SkipWorker  # Skip GPU worker
.\start-gpu-worker.ps1 -SkipDocker  # Skip Docker services
```

### Manual Startup

1. **Start GPU Worker**:

   ```powershell
   cd backend\gpu_worker
   python launch.py
   ```

2. **Start Docker Services**:

   ```powershell
   docker-compose up -d
   ```

3. **Verify Services**:

   ```powershell
   # Check GPU worker
   curl http://localhost:8765/health

   # Check backend
   curl http://localhost:8000/health
   ```

## Testing

### Integration Tests

```powershell
# Run comprehensive integration tests
python test_gpu_worker_integration.py
```

Tests include:

- Worker startup and health checks
- Batch processing with various sizes
- Connection failure recovery
- CPU fallback behavior
- Performance metrics

### Performance Tests

```powershell
# Run performance benchmarks
python test_gpu_performance.py
```

Measures:

- Throughput (images/second)
- Latency (p50, p95, p99)
- GPU utilization
- Resource usage (CPU, memory)
- Concurrent request handling

## Monitoring

### Health Endpoints

- **GPU Worker Health**: `http://localhost:8765/health`
- **GPU Info**: `http://localhost:8765/gpu_info`
- **Backend Health**: `http://localhost:8000/health`

### Key Metrics

Monitor these metrics for optimal performance:

- **Success Rate**: >95% for GPU worker requests
- **Average Latency**: <500ms per batch
- **GPU Usage Rate**: >80% when GPU is available
- **Fallback Rate**: <5% (CPU fallback frequency)

## Troubleshooting

### Common Issues

#### 1. GPU Worker Won't Start

**Symptoms**: Port 8765 not available, process fails to start

**Solutions**:

- Check if another instance is running: `netstat -an | findstr 8765`
- Kill existing processes: `taskkill /f /im python.exe`
- Check Windows mutex: Use Process Explorer to find "MordeauxGPUWorkerMutex"

#### 2. DirectML Not Available

**Symptoms**: GPU worker starts but uses CPU, logs show "DirectML not available"

**Solutions**:

- Update AMD drivers to latest version
- Verify DirectML support: `python -c "import onnxruntime; print(onnxruntime.get_available_providers())"`
- Check Windows version (DirectML requires Windows 10 1903+)

#### 3. Connection Refused Errors

**Symptoms**: Linux container can't connect to GPU worker

**Solutions**:

- Verify `host.docker.internal` resolves correctly
- Check Windows firewall settings
- Ensure Docker Desktop is running
- Test connectivity: `docker run --rm curlimages/curl curl http://host.docker.internal:8765/health`

#### 4. High CPU Fallback Rate

**Symptoms**: Most requests fall back to CPU processing

**Solutions**:

- Check GPU worker health: `curl http://localhost:8765/health`
- Monitor GPU worker logs: `backend\gpu_worker\gpu_worker.log`
- Increase circuit breaker thresholds in `gpu_client.py`
- Check network connectivity between container and host

### Debug Mode

Enable detailed logging:

```python
# In gpu_client.py, set logging level
logging.getLogger().setLevel(logging.DEBUG)
```

### Performance Tuning

#### Batch Size Optimization

Start with conservative batch sizes and increase based on success rate:

```env
GPU_WORKER_BATCH_SIZE=16  # Start small
```

#### Circuit Breaker Tuning

Adjust circuit breaker settings for your environment:

```python
# In gpu_client.py
CircuitBreaker(
    failure_threshold=15,     # Higher = more lenient
    recovery_timeout=20.0,     # Faster recovery
    success_threshold=3,       # Fewer successes needed
    min_requests=10           # Minimum before opening circuit
)
```

## Performance Optimization

### GPU Worker Settings

1. **Batch Size**: Start with 16, increase to 32-64 based on success rate
2. **Timeout**: Set to 60s for larger batches
3. **Retries**: Use 5 retries with exponential backoff
4. **Health Check**: Check every 10 seconds

### Docker Settings

1. **Memory Limits**: Ensure sufficient memory for batch processing
2. **Network**: Use `host.docker.internal` for best connectivity
3. **DNS**: Configure proper DNS resolution

### Windows Settings

1. **Power Plan**: Use "High Performance" power plan
2. **GPU Drivers**: Keep AMD drivers updated
3. **Windows Updates**: Ensure latest Windows updates

## Architecture Details

### Process Management

- **Windows Mutex**: Prevents multiple GPU worker instances
- **Watchdog**: Monitors worker health and restarts if needed
- **Graceful Shutdown**: Handles SIGINT/SIGTERM signals

### Connection Management

- **HTTP/2**: Enabled for better multiplexing
- **Connection Pooling**: Reuses connections for efficiency
- **Keep-Alive**: Maintains persistent connections
- **Circuit Breaker**: Prevents cascading failures

### Error Handling

- **Transient Errors**: Retry with exponential backoff
- **Persistent Errors**: Open circuit, fallback to CPU
- **Partial Failures**: Process successful images, retry failed ones
- **Startup Race**: Wait for worker readiness

## Security Considerations

- **Network**: GPU worker listens on all interfaces (0.0.0.0:8765)
- **Authentication**: No authentication implemented (local network only)
- **Data**: Images are processed in memory, not persisted
- **Logs**: Sensitive data may appear in logs (disable debug logging in production)

## Future Improvements

1. **Authentication**: Add API key authentication
2. **Load Balancing**: Support multiple GPU worker instances
3. **Metrics**: Prometheus/InfluxDB integration
4. **Auto-scaling**: Dynamic worker scaling based on load
5. **GPU Memory**: Better GPU memory management
6. **Model Caching**: Cache models for faster startup

## Support

For issues and questions:

1. Check the logs: `backend\gpu_worker\gpu_worker.log`
2. Run integration tests: `python test_gpu_worker_integration.py`
3. Monitor performance: `python test_gpu_performance.py`
4. Review this documentation
5. Check GitHub issues for known problems
