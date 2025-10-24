# GPU Acceleration Guide

This guide explains how to enable and configure GPU acceleration for the face scanning crawler, supporting NVIDIA CUDA, AMD ROCm, and Apple Metal backends.

## Overview

The GPU acceleration system provides optional hardware acceleration for compute-intensive operations:

- **Face Detection**: InsightFace with ONNX Runtime GPU backend
- **Face Embedding**: GPU-accelerated embedding generation
- **Image Processing**: GPU-accelerated resizing, enhancement, and quality checks
- **Quality Assessment**: GPU-accelerated blur, brightness, contrast, and sharpness detection

## Supported Platforms

### NVIDIA CUDA
- **Requirements**: NVIDIA GPU with CUDA support
- **Dependencies**: `onnxruntime-gpu`, `torch`, `torchvision`
- **Docker**: NVIDIA Container Toolkit required

### AMD ROCm
- **Requirements**: AMD GPU with ROCm support (RX 6700XT, etc.)
- **Dependencies**: `torch` with ROCm backend
- **Docker**: ROCm runtime required

### Apple Metal (M1/M2)
- **Requirements**: Apple Silicon Mac
- **Dependencies**: `torch` with MPS backend
- **Docker**: Limited support (use native installation)

## Configuration

### Environment Variables

```bash
# Master GPU control
ALL_GPU=false                    # Enable all GPU operations

# Granular controls
FACE_DETECTION_GPU=false         # Face detection acceleration
FACE_EMBEDDING_GPU=false         # Face embedding acceleration
IMAGE_PROCESSING_GPU=false       # Image processing acceleration
IMAGE_ENHANCEMENT_GPU=false      # Image enhancement acceleration
QUALITY_CHECKS_GPU=false         # Quality checks acceleration

# GPU backend selection
GPU_BACKEND=auto                 # auto|cuda|rocm|mps|cpu
GPU_DEVICE_ID=0                  # GPU device ID
GPU_MEMORY_LIMIT_GB=8            # GPU memory limit
GPU_BATCH_SIZE=32                # Batch size for GPU operations
```

### Settings Configuration

The GPU configuration is managed through the `Settings` class in `backend/app/core/settings.py`:

```python
from app.core.settings import get_settings

settings = get_settings()

# Check if GPU is enabled for specific operation
if settings.is_gpu_enabled_for_operation('face_detection'):
    print("Face detection GPU acceleration enabled")
```

## Usage

### Basic GPU Detection

```python
from app.services.gpu_manager import get_gpu_manager

gpu_manager = get_gpu_manager()

# Check available backends
backends = gpu_manager.get_available_backends()
print(f"Available backends: {[b.value for b in backends]}")

# Get preferred device
device = gpu_manager.get_preferred_device()
if device:
    print(f"Using GPU: {device}")
```

### Face Detection with GPU

```python
from app.services.face import get_face_service

face_service = get_face_service()

# Single image detection (automatically uses GPU if enabled)
faces = await face_service.detect_and_embed_async(image_bytes)

# Batch detection (optimized for GPU)
batch_results = await face_service.detect_and_embed_batch_async(
    image_list, batch_size=32
)
```

### Image Enhancement with GPU

```python
from app.services.face import get_face_service

face_service = get_face_service()

# Enhanced image processing (uses GPU if enabled)
enhanced_bytes, scale = face_service.enhance_image_for_face_detection(image_bytes)
```

### Quality Assessment with GPU

```python
from face_pipeline.pipeline.quality import QualityChecker
import numpy as np

quality_checker = QualityChecker()

# GPU-accelerated quality assessment
image_array = np.array(image)  # Convert PIL to numpy
metrics = quality_checker.assess_quality(image_array)

print(f"Blur score: {metrics.blur_score}")
print(f"Brightness: {metrics.brightness}")
print(f"Contrast: {metrics.contrast}")
print(f"Overall quality: {metrics.overall_score}")
```

## Docker Setup

### NVIDIA CUDA

1. Install NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

2. Run with GPU support:
```bash
# Build GPU-enabled image
docker-compose build backend-gpu

# Run with GPU access
docker-compose up backend-gpu
```

### AMD ROCm

1. Install ROCm Docker support:
```bash
# Install ROCm
wget https://repo.radeon.com/rocm/apt/5.7/ubuntu/jammy/rocm.gpg.key
sudo apt-key add rocm.gpg.key
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.7/ubuntu jammy main' | sudo tee /etc/apt/sources.list.d/rocm.list
sudo apt update
sudo apt install rocm-dkms
```

2. Configure Docker for ROCm:
```bash
# Add user to render group
sudo usermod -a -G render $USER

# Run with ROCm runtime
docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined -it your-image
```

## Performance Testing

### Running Benchmarks

```bash
# Run comprehensive GPU benchmark
python backend/scripts/benchmark_gpu.py

# Run specific operation benchmark
python backend/scripts/benchmark_gpu.py \
  --operation face_detection \
  --image-size medium \
  --gpu \
  --batch-size 10 \
  --iterations 20

# Run with CPU baseline
python backend/scripts/benchmark_gpu.py \
  --operation face_detection \
  --image-size medium \
  --no-gpu \
  --batch-size 10 \
  --iterations 20
```

### Benchmark Results

The benchmark script generates detailed performance reports:

```json
{
  "benchmark_results": {
    "face_detection": {
      "medium": {
        "gpu_true": {
          "mean_time_ms": 45.2,
          "mean_throughput": 22.1,
          "success_rate": 1.0
        },
        "gpu_false": {
          "mean_time_ms": 78.5,
          "mean_throughput": 12.7,
          "success_rate": 1.0
        }
      }
    }
  },
  "analysis": {
    "performance_gains": {
      "face_detection": {
        "medium": {
          "time_improvement_percent": 42.4,
          "throughput_improvement_percent": 74.0
        }
      }
    }
  }
}
```

## Troubleshooting

### Common Issues

1. **GPU not detected**:
   ```bash
   # Check GPU availability
   nvidia-smi  # For NVIDIA
   rocm-smi    # For AMD
   ```

2. **CUDA out of memory**:
   ```bash
   # Reduce batch size
   export GPU_BATCH_SIZE=16
   export GPU_MEMORY_LIMIT_GB=4
   ```

3. **PyTorch MPS not available**:
   ```bash
   # Check MPS availability
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

### Debug Mode

Enable debug logging to troubleshoot GPU issues:

```bash
export LOG_LEVEL=DEBUG
export DEBUG=true
```

### Fallback Behavior

The system automatically falls back to CPU if GPU operations fail:

- GPU unavailable → CPU fallback
- GPU memory exhausted → CPU fallback
- GPU operation error → CPU fallback

## Performance Optimization

### Recommended Settings

**For AMD RX 6700XT (Test Rig)**:
```bash
FACE_DETECTION_GPU=true
FACE_EMBEDDING_GPU=true
IMAGE_ENHANCEMENT_GPU=false  # CPU often faster for small images
QUALITY_CHECKS_GPU=true
GPU_BATCH_SIZE=32
GPU_MEMORY_LIMIT_GB=8
```

**For Apple M1 Mac (Production)**:
```bash
FACE_DETECTION_GPU=true
FACE_EMBEDDING_GPU=true
IMAGE_ENHANCEMENT_GPU=true
QUALITY_CHECKS_GPU=true
GPU_BATCH_SIZE=16  # Lower for M1
GPU_MEMORY_LIMIT_GB=4
```

### Memory Management

- Monitor GPU memory usage with `nvidia-smi` or `rocm-smi`
- Adjust `GPU_BATCH_SIZE` based on available memory
- Use `GPU_MEMORY_LIMIT_GB` to prevent OOM errors

## Development

### Adding New GPU Operations

1. Add operation flag to settings:
```python
# In settings.py
new_operation_gpu: bool = Field(default=False, env="NEW_OPERATION_GPU")
```

2. Implement GPU/CPU versions:
```python
def new_operation(self, data):
    if gpu_manager.is_operation_gpu_enabled('new_operation'):
        return self._new_operation_gpu(data)
    else:
        return self._new_operation_cpu(data)
```

3. Add to benchmark tests:
```python
# In test_gpu_performance.py
async def benchmark_new_operation(self, use_gpu: bool, iterations: int = 10):
    # Implementation
```

## Monitoring

### GPU Usage Metrics

```python
from app.services.gpu_manager import get_gpu_manager

gpu_manager = get_gpu_manager()

# Get memory info
memory_info = gpu_manager.get_memory_info()
print(f"GPU Memory: {memory_info['used']}/{memory_info['total']} MB")

# Get device info
device = gpu_manager.get_preferred_device()
print(f"GPU Device: {device}")
```

### Performance Monitoring

The system logs GPU performance metrics:

```
INFO - GPU batch processing completed: 32 images, 32 results
INFO - Face analysis model loaded with GPU acceleration: GPUDevice(id=0, name='NVIDIA GeForce RTX 4090', memory=8192/24576MB, backend=cuda)
```

## Best Practices

1. **Start with CPU**: Always test CPU performance first as baseline
2. **Gradual enablement**: Enable GPU operations one at a time
3. **Monitor performance**: Use benchmarks to validate improvements
4. **Memory management**: Set appropriate limits to prevent OOM
5. **Fallback testing**: Ensure CPU fallback works correctly
6. **Production readiness**: Test thoroughly before production deployment

## Support

For GPU acceleration issues:

1. Check the logs for GPU-related errors
2. Run the benchmark script to identify performance issues
3. Verify GPU drivers and runtime are properly installed
4. Test with CPU fallback to isolate GPU-specific problems
