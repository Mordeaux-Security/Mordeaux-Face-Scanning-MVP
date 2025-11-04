"""
Face Detection Detectors Package

Provides batched face detection backends for GPU worker.
"""

from .scrfd_onnx import SCRFDOnnx
from .common import letterbox, nms, DetectionResult

__all__ = ['SCRFDOnnx', 'letterbox', 'nms', 'DetectionResult']

