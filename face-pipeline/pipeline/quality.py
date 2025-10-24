"""
Face Quality Assessment Module

Comprehensive face quality checks with GPU acceleration support.
Implements blur detection, brightness/contrast checks, face size validation,
pose estimation, occlusion detection, and sharpness checks.

PARTIAL DUPLICATE: backend/app/services/crawler.py has basic quality checks
"""

import logging
import os
import sys
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import cv2

# Add backend path for GPU manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'backend'))

try:
    from app.services.gpu_manager import get_gpu_manager, GPUBackend
    from app.core.settings import get_settings
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    get_gpu_manager = None
    get_settings = None

logger = logging.getLogger(__name__)


class QualityMetrics:
    """Container for quality assessment metrics."""
    
    def __init__(self):
        self.blur_score: float = 0.0
        self.brightness: float = 0.0
        self.contrast: float = 0.0
        self.sharpness: float = 0.0
        self.face_size: Tuple[int, int] = (0, 0)
        self.pose_angles: Dict[str, float] = {}
        self.occlusion_score: float = 0.0
        self.overall_score: float = 0.0
    
    def is_acceptable(self, thresholds: Dict[str, float]) -> bool:
        """
        Check if quality meets minimum thresholds.
        
        Args:
            thresholds: Dictionary of threshold values for each metric
            
        Returns:
            True if all metrics meet thresholds, False otherwise
        """
        checks = [
            self.blur_score >= thresholds.get('min_blur', 0.0),
            self.brightness >= thresholds.get('min_brightness', 0.0),
            self.contrast >= thresholds.get('min_contrast', 0.0),
            self.sharpness >= thresholds.get('min_sharpness', 0.0),
            self.occlusion_score <= thresholds.get('max_occlusion', 1.0),
        ]
        return all(checks)


class QualityChecker:
    """Face quality assessment service."""
    
    def __init__(
        self,
        min_face_size: int = 50,
        min_brightness: float = 30.0,
        max_brightness: float = 225.0,
        max_blur: float = 100.0,
        min_sharpness: float = 100.0,
        max_pose_angle: float = 45.0
    ):
        """
        Initialize quality checker.
        
        Args:
            min_face_size: Minimum face size in pixels
            min_brightness: Minimum average brightness
            max_brightness: Maximum average brightness
            max_blur: Maximum blur score (lower = sharper)
            min_sharpness: Minimum sharpness score
            max_pose_angle: Maximum pose angle in degrees
        
        TODO: Add configurable thresholds
        """
        self.min_face_size = min_face_size
        self.min_brightness = min_brightness
        self.max_brightness = max_brightness
        self.max_blur = max_blur
        self.min_sharpness = min_sharpness
        self.max_pose_angle = max_pose_angle
    
    def check_blur(self, image: np.ndarray) -> float:
        """
        Compute blur score using Laplacian variance.
        
        Args:
            image: Face crop as numpy array
        
        Returns:
            Blur score (higher = sharper)
        """
        if GPU_AVAILABLE and get_gpu_manager().is_operation_gpu_enabled('quality_checks'):
            return self._check_blur_gpu(image)
        else:
            return self._check_blur_cpu(image)
    
    def _check_blur_cpu(self, image: np.ndarray) -> float:
        """CPU-based blur detection using Laplacian variance."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Compute Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(laplacian_var)
        except Exception as e:
            logger.warning(f"Error in CPU blur detection: {e}")
            return 0.0
    
    def _check_blur_gpu(self, image: np.ndarray) -> float:
        """GPU-accelerated blur detection."""
        try:
            import torch
            import torch.nn.functional as F
            
            # Get GPU device
            gpu_manager = get_gpu_manager()
            device = gpu_manager.get_preferred_device()
            
            if device and device.backend.value != 'cpu':
                if device.backend == GPUBackend.CUDA:
                    torch_device = torch.device('cuda')
                elif device.backend == GPUBackend.ROCM:
                    torch_device = torch.device('cuda')  # ROCm uses CUDA API
                elif device.backend == GPUBackend.MPS:
                    torch_device = torch.device('mps')
                else:
                    torch_device = torch.device('cpu')
            else:
                torch_device = torch.device('cpu')
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image[:, :, 0]
            else:
                gray = image
            
            # Convert to tensor and move to GPU
            tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(torch_device)
            
            # Apply Laplacian kernel
            laplacian_kernel = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], dtype=torch.float32).to(torch_device)
            laplacian = F.conv2d(tensor, laplacian_kernel, padding=1)
            
            # Compute variance
            variance = torch.var(laplacian).item()
            
            return float(variance)
            
        except ImportError:
            logger.warning("PyTorch not available for GPU blur detection, falling back to CPU")
            return self._check_blur_cpu(image)
        except Exception as e:
            logger.warning(f"GPU blur detection failed, falling back to CPU: {e}")
            return self._check_blur_cpu(image)
    
    def check_brightness(self, image: np.ndarray) -> float:
        """
        Compute average brightness.
        
        Args:
            image: Face crop as numpy array
            
        Returns:
            Average brightness value (0-255)
        """
        if GPU_AVAILABLE and get_gpu_manager().is_operation_gpu_enabled('quality_checks'):
            return self._check_brightness_gpu(image)
        else:
            return self._check_brightness_cpu(image)
    
    def _check_brightness_cpu(self, image: np.ndarray) -> float:
        """CPU-based brightness calculation."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            return float(np.mean(gray))
        except Exception as e:
            logger.warning(f"Error in CPU brightness calculation: {e}")
            return 0.0
    
    def _check_brightness_gpu(self, image: np.ndarray) -> float:
        """GPU-accelerated brightness calculation."""
        try:
            import torch
            
            # Get GPU device
            gpu_manager = get_gpu_manager()
            device = gpu_manager.get_preferred_device()
            
            if device and device.backend.value != 'cpu':
                if device.backend == GPUBackend.CUDA:
                    torch_device = torch.device('cuda')
                elif device.backend == GPUBackend.ROCM:
                    torch_device = torch.device('cuda')
                elif device.backend == GPUBackend.MPS:
                    torch_device = torch.device('mps')
                else:
                    torch_device = torch.device('cpu')
            else:
                torch_device = torch.device('cpu')
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image[:, :, 0]
            else:
                gray = image
            
            # Convert to tensor and move to GPU
            tensor = torch.from_numpy(gray).float().to(torch_device)
            
            # Compute mean brightness
            brightness = torch.mean(tensor).item()
            
            return float(brightness)
            
        except ImportError:
            logger.warning("PyTorch not available for GPU brightness calculation, falling back to CPU")
            return self._check_brightness_cpu(image)
        except Exception as e:
            logger.warning(f"GPU brightness calculation failed, falling back to CPU: {e}")
            return self._check_brightness_cpu(image)
    
    def check_contrast(self, image: np.ndarray) -> float:
        """
        Compute contrast ratio using standard deviation.
        
        Args:
            image: Face crop as numpy array
            
        Returns:
            Contrast value (standard deviation)
        """
        if GPU_AVAILABLE and get_gpu_manager().is_operation_gpu_enabled('quality_checks'):
            return self._check_contrast_gpu(image)
        else:
            return self._check_contrast_cpu(image)
    
    def _check_contrast_cpu(self, image: np.ndarray) -> float:
        """CPU-based contrast calculation."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            return float(np.std(gray))
        except Exception as e:
            logger.warning(f"Error in CPU contrast calculation: {e}")
            return 0.0
    
    def _check_contrast_gpu(self, image: np.ndarray) -> float:
        """GPU-accelerated contrast calculation."""
        try:
            import torch
            
            # Get GPU device
            gpu_manager = get_gpu_manager()
            device = gpu_manager.get_preferred_device()
            
            if device and device.backend.value != 'cpu':
                if device.backend == GPUBackend.CUDA:
                    torch_device = torch.device('cuda')
                elif device.backend == GPUBackend.ROCM:
                    torch_device = torch.device('cuda')
                elif device.backend == GPUBackend.MPS:
                    torch_device = torch.device('mps')
                else:
                    torch_device = torch.device('cpu')
            else:
                torch_device = torch.device('cpu')
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image[:, :, 0]
            else:
                gray = image
            
            # Convert to tensor and move to GPU
            tensor = torch.from_numpy(gray).float().to(torch_device)
            
            # Compute standard deviation (contrast)
            contrast = torch.std(tensor).item()
            
            return float(contrast)
            
        except ImportError:
            logger.warning("PyTorch not available for GPU contrast calculation, falling back to CPU")
            return self._check_contrast_cpu(image)
        except Exception as e:
            logger.warning(f"GPU contrast calculation failed, falling back to CPU: {e}")
            return self._check_contrast_cpu(image)
    
    def check_sharpness(self, image: np.ndarray) -> float:
        """
        Compute sharpness score using gradient magnitude.
        
        Args:
            image: Face crop as numpy array
            
        Returns:
            Sharpness score (higher = sharper)
        """
        if GPU_AVAILABLE and get_gpu_manager().is_operation_gpu_enabled('quality_checks'):
            return self._check_sharpness_gpu(image)
        else:
            return self._check_sharpness_cpu(image)
    
    def _check_sharpness_cpu(self, image: np.ndarray) -> float:
        """CPU-based sharpness calculation."""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Compute gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Compute gradient magnitude
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            return float(np.mean(gradient_magnitude))
        except Exception as e:
            logger.warning(f"Error in CPU sharpness calculation: {e}")
            return 0.0
    
    def _check_sharpness_gpu(self, image: np.ndarray) -> float:
        """GPU-accelerated sharpness calculation."""
        try:
            import torch
            import torch.nn.functional as F
            
            # Get GPU device
            gpu_manager = get_gpu_manager()
            device = gpu_manager.get_preferred_device()
            
            if device and device.backend.value != 'cpu':
                if device.backend == GPUBackend.CUDA:
                    torch_device = torch.device('cuda')
                elif device.backend == GPUBackend.ROCM:
                    torch_device = torch.device('cuda')
                elif device.backend == GPUBackend.MPS:
                    torch_device = torch.device('mps')
                else:
                    torch_device = torch.device('cpu')
            else:
                torch_device = torch.device('cpu')
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image[:, :, 0]
            else:
                gray = image
            
            # Convert to tensor and move to GPU
            tensor = torch.from_numpy(gray).float().unsqueeze(0).unsqueeze(0).to(torch_device)
            
            # Sobel kernels for gradient calculation
            sobel_x = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32).to(torch_device)
            sobel_y = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32).to(torch_device)
            
            # Compute gradients
            grad_x = F.conv2d(tensor, sobel_x, padding=1)
            grad_y = F.conv2d(tensor, sobel_y, padding=1)
            
            # Compute gradient magnitude
            gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2)
            
            # Compute mean sharpness
            sharpness = torch.mean(gradient_magnitude).item()
            
            return float(sharpness)
            
        except ImportError:
            logger.warning("PyTorch not available for GPU sharpness calculation, falling back to CPU")
            return self._check_sharpness_cpu(image)
        except Exception as e:
            logger.warning(f"GPU sharpness calculation failed, falling back to CPU: {e}")
            return self._check_sharpness_cpu(image)
    
    def check_pose(self, landmarks: np.ndarray) -> Dict[str, float]:
        """
        Estimate head pose angles.
        
        Args:
            landmarks: Facial landmarks as numpy array
        
        Returns:
            Dict with pitch, yaw, roll angles
        
        TODO: Implement pose estimation from landmarks
        TODO: Use PnP or similar algorithm
        """
        pass
    
    def check_occlusion(self, image: np.ndarray, landmarks: np.ndarray) -> float:
        """
        Detect facial occlusions.
        
        TODO: Implement occlusion detection
        TODO: Check if key landmarks are visible
        """
        pass
    
    def assess_quality(
        self, 
        image: np.ndarray, 
        bbox: Optional[List[float]] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> QualityMetrics:
        """
        Comprehensive quality assessment.
        
        Args:
            image: Face crop as numpy array
            bbox: Optional bounding box [x1, y1, x2, y2]
            landmarks: Optional facial landmarks
        
        Returns:
            QualityMetrics object with all scores
        """
        metrics = QualityMetrics()
        
        try:
            # Calculate face size
            if bbox:
                metrics.face_size = (int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1]))
            
            # Run quality checks
            metrics.blur_score = self.check_blur(image)
            metrics.brightness = self.check_brightness(image)
            metrics.contrast = self.check_contrast(image)
            metrics.sharpness = self.check_sharpness(image)
            
            # Pose estimation if landmarks available
            if landmarks is not None:
                metrics.pose_angles = self.check_pose(landmarks)
            
            # Occlusion detection
            metrics.occlusion_score = self.check_occlusion(image, landmarks)
            
            # Calculate overall quality score
            metrics.overall_score = self._calculate_overall_score(metrics)
            
        except Exception as e:
            logger.warning(f"Error in quality assessment: {e}")
            # Return default metrics with zero scores
        
        return metrics
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """
        Calculate overall quality score from individual metrics.
        
        Args:
            metrics: QualityMetrics object with individual scores
            
        Returns:
            Overall quality score (0.0-1.0)
        """
        # Normalize individual scores
        blur_norm = min(metrics.blur_score / 1000.0, 1.0)  # Normalize blur score
        brightness_norm = 1.0 - abs(metrics.brightness - 128.0) / 128.0  # Optimal around 128
        contrast_norm = min(metrics.contrast / 50.0, 1.0)  # Normalize contrast
        sharpness_norm = min(metrics.sharpness / 100.0, 1.0)  # Normalize sharpness
        occlusion_norm = 1.0 - metrics.occlusion_score  # Lower occlusion is better
        
        # Weighted average
        weights = {
            'blur': 0.25,
            'brightness': 0.20,
            'contrast': 0.20,
            'sharpness': 0.25,
            'occlusion': 0.10
        }
        
        overall_score = (
            weights['blur'] * blur_norm +
            weights['brightness'] * brightness_norm +
            weights['contrast'] * contrast_norm +
            weights['sharpness'] * sharpness_norm +
            weights['occlusion'] * occlusion_norm
        )
        
        return max(0.0, min(1.0, overall_score))
    
    async def assess_quality_async(
        self,
        image: np.ndarray,
        bbox: Optional[List[float]] = None,
        landmarks: Optional[np.ndarray] = None
    ) -> QualityMetrics:
        """
        Async wrapper for quality assessment.
        
        TODO: Run quality checks in thread pool
        """
        pass

