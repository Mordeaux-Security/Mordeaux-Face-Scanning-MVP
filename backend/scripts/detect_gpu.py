#!/usr/bin/env python3
"""
GPU Hardware Detection Script

This script detects the available GPU hardware and returns the appropriate
configuration for Docker builds and runtime.
"""

import subprocess
import sys
import platform
import os
from typing import Dict, List, Optional


def run_command(cmd: List[str]) -> Optional[str]:
    """Run a command and return its output."""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def detect_nvidia_gpu() -> bool:
    """Detect if NVIDIA GPU is available."""
    # Check for nvidia-smi
    if run_command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"]):
        return True
    
    # Check for NVIDIA in Windows
    if platform.system() == "Windows":
        try:
            result = run_command(["wmic", "path", "win32_VideoController", "get", "name"])
            if result and "nvidia" in result.lower():
                return True
        except:
            pass
    
    return False


def detect_amd_gpu() -> bool:
    """Detect if AMD GPU is available."""
    # Check for AMD in Windows
    if platform.system() == "Windows":
        try:
            result = run_command(["wmic", "path", "win32_VideoController", "get", "name"])
            if result and ("amd" in result.lower() or "radeon" in result.lower()):
                return True
        except:
            pass
    
    # Check for AMD on Linux
    if platform.system() == "Linux":
        # Check for AMD GPU in lspci
        result = run_command(["lspci", "-nn"])
        if result and ("amd" in result.lower() or "radeon" in result.lower()):
            return True
    
    return False


def detect_apple_silicon() -> bool:
    """Detect if running on Apple Silicon (M1/M2/M3)."""
    if platform.system() == "Darwin":
        try:
            result = run_command(["uname", "-m"])
            if result and "arm64" in result:
                return True
        except:
            pass
    return False


def detect_intel_gpu() -> bool:
    """Detect if Intel GPU is available."""
    if platform.system() == "Windows":
        try:
            result = run_command(["wmic", "path", "win32_VideoController", "get", "name"])
            if result and "intel" in result.lower():
                return True
        except:
            pass
    
    return False


def get_gpu_config() -> Dict[str, any]:
    """Detect GPU hardware and return configuration."""
    config = {
        "has_gpu": False,
        "gpu_type": "none",
        "gpu_backend": "cpu",
        "docker_runtime": None,
        "install_packages": [],
        "python_packages": [],
        "environment_vars": {}
    }
    
    # Detect GPU types
    has_nvidia = detect_nvidia_gpu()
    has_amd = detect_amd_gpu()
    has_apple = detect_apple_silicon()
    has_intel = detect_intel_gpu()
    
    if has_nvidia:
        config.update({
            "has_gpu": True,
            "gpu_type": "nvidia",
            "gpu_backend": "cuda",
            "docker_runtime": "nvidia",
            "install_packages": [
                "nvidia-cuda-toolkit",
                "libcudnn8",
                "libcudnn8-dev"
            ],
            "python_packages": [
                "torch==2.1.0",
                "torchvision==0.16.0",
                "onnxruntime-gpu==1.19.2",
                "opencv-contrib-python==4.10.0.84"
            ],
            "environment_vars": {
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"
            }
        })
    elif has_amd:
        config.update({
            "has_gpu": True,
            "gpu_type": "directml",
            "gpu_backend": "directml",
            "docker_runtime": None,  # DirectML works in Windows Docker Desktop
            "install_packages": [],  # No system packages needed for DirectML
            "python_packages": [
                "opencv-contrib-python==4.10.0.84"
            ],
            "environment_vars": {}  # DirectML environment variables set automatically
        })
    elif has_apple:
        config.update({
            "has_gpu": True,
            "gpu_type": "apple",
            "gpu_backend": "mps",
            "docker_runtime": None,  # Not supported in Docker
            "install_packages": [],
            "python_packages": [
                "torch==2.1.0",
                "torchvision==0.16.0",
                "opencv-contrib-python==4.10.0.84"
            ],
            "environment_vars": {
                "PYTORCH_ENABLE_MPS_FALLBACK": "1"
            }
        })
    elif has_intel:
        config.update({
            "has_gpu": True,
            "gpu_type": "intel",
            "gpu_backend": "cpu",  # Intel GPU support is limited
            "docker_runtime": None,
            "install_packages": [
                "intel-opencl-icd",
                "intel-media-va-driver-non-free"
            ],
            "python_packages": [
                "opencv-contrib-python==4.10.0.84"
            ],
            "environment_vars": {}
        })
    
    return config


def main():
    """Main function to detect and print GPU configuration."""
    config = get_gpu_config()
    
    print("GPU Detection Results:")
    print(f"Has GPU: {config['has_gpu']}")
    print(f"GPU Type: {config['gpu_type']}")
    print(f"GPU Backend: {config['gpu_backend']}")
    print(f"Docker Runtime: {config['docker_runtime']}")
    print(f"Install Packages: {config['install_packages']}")
    print(f"Python Packages: {config['python_packages']}")
    print(f"Environment Variables: {config['environment_vars']}")
    
    # Write config to file for Docker build
    import json
    import tempfile
    config_file = os.path.join(tempfile.gettempdir(), "gpu_config.json")
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)
    
    return 0 if config['has_gpu'] else 1


if __name__ == "__main__":
    sys.exit(main())
