"""
GPU Manager Service

Handles GPU backend detection, device management, and provides unified GPU operations
across different platforms (NVIDIA CUDA, AMD ROCm, Apple Metal).
"""

import logging
import os
import platform
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class GPUBackend(Enum):
    """Available GPU backends."""
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    CPU = "cpu"


class GPUDevice:
    """Represents a GPU device with capabilities."""
    
    def __init__(self, device_id: int, name: str, memory_total: int, memory_free: int, backend: GPUBackend):
        self.device_id = device_id
        self.name = name
        self.memory_total = memory_total
        self.memory_free = memory_free
        self.backend = backend
        self.is_available = True
    
    def __repr__(self):
        return f"GPUDevice(id={self.device_id}, name='{self.name}', memory={self.memory_free}/{self.memory_total}MB, backend={self.backend.value})"


class GPUManager:
    """
    Manages GPU resources and provides unified GPU operations.
    
    Features:
    - Auto-detection of available GPU backends
    - Device enumeration and capability checking
    - Memory management and monitoring
    - Fallback to CPU when GPU unavailable
    - Thread-safe operations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._devices: List[GPUDevice] = []
        self._available_backends: List[GPUBackend] = []
        self._preferred_backend: Optional[GPUBackend] = None
        self._lock = threading.Lock()
        self._initialized = False
        
        # Configuration from environment
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load GPU configuration from environment variables."""
        return {
            'all_gpu': os.getenv('ALL_GPU', 'false').lower() == 'true',
            'face_detection_gpu': os.getenv('FACE_DETECTION_GPU', 'false').lower() == 'true',
            'face_embedding_gpu': os.getenv('FACE_EMBEDDING_GPU', 'false').lower() == 'true',
            'image_processing_gpu': os.getenv('IMAGE_PROCESSING_GPU', 'false').lower() == 'true',
            'image_enhancement_gpu': os.getenv('IMAGE_ENHANCEMENT_GPU', 'false').lower() == 'true',
            'quality_checks_gpu': os.getenv('QUALITY_CHECKS_GPU', 'false').lower() == 'true',
            'gpu_backend': os.getenv('GPU_BACKEND', 'auto').lower(),
            'gpu_device_id': int(os.getenv('GPU_DEVICE_ID', '0')),
            'gpu_memory_limit_gb': float(os.getenv('GPU_MEMORY_LIMIT_GB', '8')),
            'gpu_batch_size': int(os.getenv('GPU_BATCH_SIZE', '32')),
        }
    
    def initialize(self) -> bool:
        """
        Initialize GPU manager and detect available backends.
        
        Returns:
            True if at least one GPU backend is available, False otherwise
        """
        with self._lock:
            if self._initialized:
                return len(self._available_backends) > 0
            
            self.logger.info("Initializing GPU manager...")
            
            # Detect available backends
            self._detect_backends()
            
            # Enumerate devices for each backend
            self._enumerate_devices()
            
            # Set preferred backend
            self._set_preferred_backend()
            
            self._initialized = True
            
            if self._available_backends:
                self.logger.info(f"GPU manager initialized with backends: {[b.value for b in self._available_backends]}")
                self.logger.info(f"Available devices: {[str(d) for d in self._devices]}")
            else:
                self.logger.info("No GPU backends available, falling back to CPU")
            
            return len(self._available_backends) > 0
    
    def _detect_backends(self):
        """Detect available GPU backends."""
        self._available_backends = []
        
        # Detect CUDA
        if self._detect_cuda():
            self._available_backends.append(GPUBackend.CUDA)
        
        # Detect ROCm
        if self._detect_rocm():
            self._available_backends.append(GPUBackend.ROCM)
        
        # Detect MPS (Apple Metal)
        if self._detect_mps():
            self._available_backends.append(GPUBackend.MPS)
        
        # Always add CPU as fallback
        self._available_backends.append(GPUBackend.CPU)
    
    def _detect_cuda(self) -> bool:
        """Detect CUDA availability."""
        try:
            # Check for nvidia-smi
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.debug("CUDA detected via nvidia-smi")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        try:
            # Check for CUDA in Python
            import torch
            if torch.cuda.is_available():
                self.logger.debug("CUDA detected via PyTorch")
                return True
        except ImportError:
            pass
        
        return False
    
    def _detect_rocm(self) -> bool:
        """Detect ROCm availability."""
        try:
            # Check for rocm-smi
            result = subprocess.run(['rocm-smi'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.logger.debug("ROCm detected via rocm-smi")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        try:
            # Check for ROCm in Python
            import torch
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                self.logger.debug("ROCm detected via PyTorch")
                return True
        except ImportError:
            pass
        
        return False
    
    def _detect_mps(self) -> bool:
        """Detect Apple Metal Performance Shaders availability."""
        if platform.system() != 'Darwin':
            return False
        
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.logger.debug("MPS detected via PyTorch")
                return True
        except ImportError:
            pass
        
        return False
    
    def _enumerate_devices(self):
        """Enumerate GPU devices for each available backend."""
        self._devices = []
        
        for backend in self._available_backends:
            if backend == GPUBackend.CPU:
                continue
            
            devices = self._get_devices_for_backend(backend)
            self._devices.extend(devices)
    
    def _get_devices_for_backend(self, backend: GPUBackend) -> List[GPUDevice]:
        """Get devices for a specific backend."""
        devices = []
        
        try:
            if backend == GPUBackend.CUDA:
                devices = self._get_cuda_devices()
            elif backend == GPUBackend.ROCM:
                devices = self._get_rocm_devices()
            elif backend == GPUBackend.MPS:
                devices = self._get_mps_devices()
        except Exception as e:
            self.logger.warning(f"Error enumerating {backend.value} devices: {e}")
        
        return devices
    
    def _get_cuda_devices(self) -> List[GPUDevice]:
        """Get CUDA devices."""
        devices = []
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)  # MB
                    memory_free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) // (1024 * 1024)  # MB
                    devices.append(GPUDevice(i, name, memory_total, memory_free, GPUBackend.CUDA))
        except Exception as e:
            self.logger.warning(f"Error getting CUDA devices: {e}")
        
        return devices
    
    def _get_rocm_devices(self) -> List[GPUDevice]:
        """Get ROCm devices."""
        devices = []
        try:
            import torch
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                # ROCm device enumeration is similar to CUDA
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    memory_total = torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)  # MB
                    memory_free = (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_reserved(i)) // (1024 * 1024)  # MB
                    devices.append(GPUDevice(i, name, memory_total, memory_free, GPUBackend.ROCM))
        except Exception as e:
            self.logger.warning(f"Error getting ROCm devices: {e}")
        
        return devices
    
    def _get_mps_devices(self) -> List[GPUDevice]:
        """Get MPS devices."""
        devices = []
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS typically has one device
                devices.append(GPUDevice(0, "Apple GPU", 0, 0, GPUBackend.MPS))  # Memory info not easily available
        except Exception as e:
            self.logger.warning(f"Error getting MPS devices: {e}")
        
        return devices
    
    def _set_preferred_backend(self):
        """Set the preferred GPU backend based on configuration and availability."""
        backend_pref = self._config['gpu_backend']
        
        if backend_pref == 'auto':
            # Priority: ROCm > CUDA > MPS > CPU
            for backend in [GPUBackend.ROCM, GPUBackend.CUDA, GPUBackend.MPS]:
                if backend in self._available_backends:
                    self._preferred_backend = backend
                    break
        else:
            try:
                preferred = GPUBackend(backend_pref)
                if preferred in self._available_backends:
                    self._preferred_backend = preferred
            except ValueError:
                self.logger.warning(f"Invalid GPU backend preference: {backend_pref}")
        
        if self._preferred_backend is None:
            self._preferred_backend = GPUBackend.CPU
        
        self.logger.info(f"Preferred GPU backend: {self._preferred_backend.value}")
    
    def get_preferred_device(self) -> Optional[GPUDevice]:
        """Get the preferred GPU device."""
        if not self._initialized:
            self.initialize()
        
        if not self._devices:
            return None
        
        # Return device with preferred backend
        for device in self._devices:
            if device.backend == self._preferred_backend:
                return device
        
        # Fallback to first available device
        return self._devices[0] if self._devices else None
    
    def get_device_by_id(self, device_id: int) -> Optional[GPUDevice]:
        """Get GPU device by ID."""
        for device in self._devices:
            if device.device_id == device_id:
                return device
        return None
    
    def is_operation_gpu_enabled(self, operation: str) -> bool:
        """
        Check if a specific operation should use GPU.
        
        Args:
            operation: Operation name (face_detection, face_embedding, image_processing, etc.)
            
        Returns:
            True if operation should use GPU, False otherwise
        """
        if not self._initialized:
            self.initialize()
        
        # Check if any GPU backend is available
        if not self._available_backends or self._preferred_backend == GPUBackend.CPU:
            return False
        
        # Check master flag
        if self._config['all_gpu']:
            return True
        
        # Check specific operation flag
        operation_key = f"{operation}_gpu"
        return self._config.get(operation_key, False)
    
    def get_gpu_context_id(self) -> int:
        """
        Get GPU context ID for InsightFace and other libraries.
        
        Returns:
            Context ID (0 for GPU, -1 for CPU)
        """
        if self.is_operation_gpu_enabled('face_detection') and self._preferred_backend != GPUBackend.CPU:
            return 0
        return -1
    
    def get_available_backends(self) -> List[GPUBackend]:
        """Get list of available GPU backends."""
        if not self._initialized:
            self.initialize()
        return self._available_backends.copy()
    
    def get_devices(self) -> List[GPUDevice]:
        """Get list of available GPU devices."""
        if not self._initialized:
            self.initialize()
        return self._devices.copy()
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get GPU memory information."""
        device = self.get_preferred_device()
        if not device:
            return {'total': 0, 'free': 0, 'used': 0}
        
        return {
            'total': device.memory_total,
            'free': device.memory_free,
            'used': device.memory_total - device.memory_free
        }
    
    def cleanup(self):
        """Clean up GPU resources."""
        with self._lock:
            self.logger.info("Cleaning up GPU resources...")
            # GPU cleanup is typically handled by the frameworks
            # This is a placeholder for any custom cleanup needed
            self._initialized = False


# Global GPU manager instance
_gpu_manager: Optional[GPUManager] = None
_manager_lock = threading.Lock()


def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance."""
    global _gpu_manager
    
    with _manager_lock:
        if _gpu_manager is None:
            _gpu_manager = GPUManager()
            _gpu_manager.initialize()
        
        return _gpu_manager


def cleanup_gpu_manager():
    """Clean up the global GPU manager."""
    global _gpu_manager
    
    with _manager_lock:
        if _gpu_manager is not None:
            _gpu_manager.cleanup()
            _gpu_manager = None
