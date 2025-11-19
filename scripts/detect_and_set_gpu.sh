#!/bin/bash
# GPU Detection and Environment Setup Script

echo "Detecting GPU hardware..."

# Run GPU detection
python backend/scripts/detect_gpu.py > /tmp/gpu_detection.log 2>&1

# Check detection results
if grep -q "GPU Type: amd" /tmp/gpu_detection.log; then
    echo "AMD GPU detected - configuring for ROCm"
    export GPU_TYPE=amd
    export GPU_DRIVER=rocm
    export GPU_BACKEND=rocm
elif grep -q "GPU Type: nvidia" /tmp/gpu_detection.log; then
    echo "NVIDIA GPU detected - configuring for CUDA"
    export GPU_TYPE=nvidia
    export GPU_DRIVER=nvidia
    export GPU_BACKEND=cuda
elif grep -q "GPU Type: intel" /tmp/gpu_detection.log; then
    echo "Intel GPU detected - configuring for Intel GPU"
    export GPU_TYPE=intel
    export GPU_DRIVER=intel
    export GPU_BACKEND=cpu
else
    echo "No compatible GPU detected - using CPU-only mode"
    export GPU_TYPE=cpu
    export GPU_DRIVER=cpu
    export GPU_BACKEND=cpu
fi

echo "GPU Configuration:"
echo "  Type: $GPU_TYPE"
echo "  Driver: $GPU_DRIVER"
echo "  Backend: $GPU_BACKEND"

# Write to .env file
echo "GPU_TYPE=$GPU_TYPE" >> .env
echo "GPU_DRIVER=$GPU_DRIVER" >> .env
echo "GPU_BACKEND=$GPU_BACKEND" >> .env

echo "GPU configuration written to .env file"

