"""
Common utilities for face detection detectors.

Provides letterbox preprocessing, NMS, and shared type definitions.
"""

import numpy as np
import cv2
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class LetterboxInfo:
    """Information about letterbox transformation for coordinate mapping."""
    def __init__(self, scale: float, pad_x: float, pad_y: float, 
                 orig_w: int, orig_h: int, new_w: int, new_h: int):
        self.scale = scale
        self.pad_x = pad_x
        self.pad_y = pad_y
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.new_w = new_w
        self.new_h = new_h
    
    def map_box_to_original(self, box: np.ndarray) -> np.ndarray:
        """
        Map bounding box from letterbox coordinates to original image coordinates.
        
        Args:
            box: Bounding box in (x1, y1, x2, y2) format in letterbox space
        
        Returns:
            Bounding box in original image coordinates
        """
        x1, y1, x2, y2 = box
        # Map from letterbox space to original
        x1 = (x1 - self.pad_x) / self.scale
        y1 = (y1 - self.pad_y) / self.scale
        x2 = (x2 - self.pad_x) / self.scale
        y2 = (y2 - self.pad_y) / self.scale
        
        # Clamp to original image bounds
        x1 = max(0, min(x1, self.orig_w))
        y1 = max(0, min(y1, self.orig_h))
        x2 = max(0, min(x2, self.orig_w))
        y2 = max(0, min(y2, self.orig_h))
        
        return np.array([x1, y1, x2, y2], dtype=np.float32)
    
    def map_kps_to_original(self, kps: np.ndarray) -> np.ndarray:
        """
        Map keypoints from letterbox coordinates to original image coordinates.
        
        Args:
            kps: Keypoints array (K, 5, 2) in letterbox space
        
        Returns:
            Keypoints in original image coordinates
        """
        if kps is None or len(kps) == 0:
            return kps
        
        kps_orig = kps.copy().astype(np.float32)
        # Map each keypoint
        kps_orig[:, :, 0] = (kps[:, :, 0] - self.pad_x) / self.scale
        kps_orig[:, :, 1] = (kps[:, :, 1] - self.pad_y) / self.scale
        
        # Clamp to original image bounds
        kps_orig[:, :, 0] = np.clip(kps_orig[:, :, 0], 0, self.orig_w)
        kps_orig[:, :, 1] = np.clip(kps_orig[:, :, 1], 0, self.orig_h)
        
        return kps_orig


class DetectionResult:
    """Container for detection results per image."""
    def __init__(self, boxes: np.ndarray, scores: np.ndarray, 
                 kps: Optional[np.ndarray] = None):
        """
        Initialize detection result.
        
        Args:
            boxes: Bounding boxes (K, 4) in xyxy format
            scores: Confidence scores (K,)
            kps: Keypoints (K, 5, 2) or None
        """
        self.boxes = boxes
        self.scores = scores
        self.kps = kps


def letterbox(image: np.ndarray, target_size: int = 640, 
              fill_color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, LetterboxInfo]:
    """
    Resize image to target_size while preserving aspect ratio, padding with fill_color.
    
    Args:
        image: Input image (H, W, 3) in BGR format
        target_size: Target size (both width and height)
        fill_color: Fill color for padding (B, G, R)
    
    Returns:
        Tuple of (letterboxed_image, LetterboxInfo)
    """
    h, w = image.shape[:2]
    orig_h, orig_w = h, w
    
    # Calculate scale to fit target_size
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize image
    if scale != 1.0:
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = image.copy()
    
    # Create canvas with fill color
    canvas = np.full((target_size, target_size, 3), fill_color, dtype=np.uint8)
    
    # Calculate padding to center image
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    
    # Place resized image on canvas
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    # Create letterbox info
    info = LetterboxInfo(
        scale=scale,
        pad_x=pad_x,
        pad_y=pad_y,
        orig_w=orig_w,
        orig_h=orig_h,
        new_w=new_w,
        new_h=new_h
    )
    
    return canvas, info


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.4) -> np.ndarray:
    """
    Non-Maximum Suppression (NMS) to remove overlapping detections.
    
    Args:
        boxes: Bounding boxes (N, 4) in xyxy format
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Indices of boxes to keep
    """
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    boxes = boxes[order]
    scores = scores[order]
    
    keep = []
    while len(order) > 0:
        # Keep the box with highest score
        keep.append(order[0])
        
        if len(order) == 1:
            break
        
        # Calculate IoU with remaining boxes
        ious = _compute_iou(boxes[0:1], boxes[1:])
        
        # Remove boxes with IoU > threshold
        mask = ious < iou_threshold
        order = order[1:][mask]
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
    
    return np.array(keep, dtype=np.int32)


def _compute_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Compute IoU between boxes1 and boxes2.
    
    Args:
        boxes1: (M, 4) in xyxy format
        boxes2: (N, 4) in xyxy format
    
    Returns:
        IoU matrix (M, N)
    """
    # Expand dimensions for broadcasting
    boxes1 = boxes1[:, None, :]  # (M, 1, 4)
    boxes2 = boxes2[None, :, :]   # (1, N, 4)
    
    # Compute intersection
    x1 = np.maximum(boxes1[:, :, 0], boxes2[:, :, 0])
    y1 = np.maximum(boxes1[:, :, 1], boxes2[:, :, 1])
    x2 = np.minimum(boxes1[:, :, 2], boxes2[:, :, 2])
    y2 = np.minimum(boxes1[:, :, 3], boxes2[:, :, 3])
    
    inter_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    
    # Compute union
    area1 = (boxes1[:, :, 2] - boxes1[:, :, 0]) * (boxes1[:, :, 3] - boxes1[:, :, 1])
    area2 = (boxes2[:, :, 2] - boxes2[:, :, 0]) * (boxes2[:, :, 3] - boxes2[:, :, 1])
    union_area = area1 + area2 - inter_area
    
    # Compute IoU
    iou = inter_area / (union_area + 1e-7)
    
    return iou.squeeze(0) if boxes1.shape[0] == 1 else iou

