"""
Face Quality Evaluation Module

This module provides quality assessment for face images used in enrollment,
verification, and search operations. Each operation has different quality
requirements:
- ENROLL: Strictest - requires high quality for identity registration
- VERIFY: Moderate - requires decent quality for identity verification  
- SEARCH: Lenient - allows lower quality for general face search

Quality metrics include:
- Blur detection (Laplacian variance)
- Face size validation
- Pose angle estimation (yaw, pitch, roll)
- Brightness/contrast checks
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np
import cv2


@dataclass
class FaceQualityConfig:
    """Configuration for face quality evaluation thresholds."""
    
    # Minimum face size (width and height) in pixels
    min_face_size: int = 80
    
    # Minimum Laplacian variance for blur detection
    # Higher = sharper image required
    min_blur_variance: float = 120.0
    
    # Minimum overall quality score (0.0 - 1.0)
    min_overall_quality: float = 0.7
    
    # Maximum pose angles in degrees
    max_yaw: float = 45.0
    max_pitch: float = 30.0
    max_roll: float = 30.0
    
    # Brightness thresholds (0-255)
    min_brightness: float = 30.0
    max_brightness: float = 225.0
    
    # Whether to require a single face (vs allowing multi-face selection)
    require_single_face: bool = False
    
    # Label for this config (for logging)
    label: str = "default"


@dataclass  
class FaceQualityResult:
    """Result of face quality evaluation."""
    
    # Whether the face passes all quality checks
    is_usable: bool = False
    
    # Overall quality score (0.0 - 1.0)
    score: float = 0.0
    
    # List of reasons why quality check failed (empty if passed)
    reasons: List[str] = field(default_factory=list)
    
    # Individual metric values
    blur_variance: float = 0.0
    face_width: int = 0
    face_height: int = 0
    brightness: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    
    # Raw metrics dict for debugging
    metrics: Dict[str, Any] = field(default_factory=dict)


# ----- Preset Quality Configurations -----

# Strictest quality for enrollment - we need high-quality reference images
ENROLL_QUALITY = FaceQualityConfig(
    min_face_size=112,
    min_blur_variance=150.0,
    min_overall_quality=0.75,
    max_yaw=30.0,
    max_pitch=25.0,
    max_roll=20.0,
    min_brightness=40.0,
    max_brightness=220.0,
    require_single_face=True,
    label="enroll",
)

# Moderate quality for verification - balance between security and usability
VERIFY_QUALITY = FaceQualityConfig(
    min_face_size=80,
    min_blur_variance=100.0,
    min_overall_quality=0.65,
    max_yaw=40.0,
    max_pitch=30.0,
    max_roll=25.0,
    min_brightness=35.0,
    max_brightness=225.0,
    require_single_face=True,
    label="verify",
)

# Lenient quality for search - allows more faces through for discovery
SEARCH_QUALITY = FaceQualityConfig(
    min_face_size=64,
    min_blur_variance=80.0,
    min_overall_quality=0.50,
    max_yaw=50.0,
    max_pitch=40.0,
    max_roll=35.0,
    min_brightness=25.0,
    max_brightness=235.0,
    require_single_face=False,
    label="search",
)

# Quality config for crawler/ingest - most lenient
CRAWLER_INGEST_QUALITY = FaceQualityConfig(
    min_face_size=48,
    min_blur_variance=60.0,
    min_overall_quality=0.40,
    max_yaw=60.0,
    max_pitch=50.0,
    max_roll=45.0,
    min_brightness=20.0,
    max_brightness=240.0,
    require_single_face=False,
    label="crawler",
)


def laplacian_variance(img_bgr: np.ndarray) -> float:
    """
    Calculate Laplacian variance as a measure of image sharpness.
    
    Higher values indicate sharper images, lower values indicate blur.
    
    Args:
        img_bgr: BGR image as numpy array
        
    Returns:
        Variance of the Laplacian (float)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def calculate_brightness(img_bgr: np.ndarray) -> float:
    """
    Calculate average brightness of an image.
    
    Args:
        img_bgr: BGR image as numpy array
        
    Returns:
        Average brightness (0-255)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def extract_pose_angles(face) -> tuple:
    """
    Extract pose angles from InsightFace face object.
    
    Args:
        face: InsightFace Face object with pose attribute
        
    Returns:
        Tuple of (yaw, pitch, roll) in degrees
    """
    # InsightFace provides pose as [pitch, yaw, roll] in radians
    pose = getattr(face, 'pose', None)
    if pose is not None and len(pose) >= 3:
        # Convert radians to degrees
        pitch = float(np.degrees(pose[0]))
        yaw = float(np.degrees(pose[1]))
        roll = float(np.degrees(pose[2]))
        return yaw, pitch, roll
    return 0.0, 0.0, 0.0


def get_face_crop(img_bgr: np.ndarray, face) -> np.ndarray:
    """
    Extract face region from image using bounding box.
    
    Args:
        img_bgr: Full BGR image
        face: InsightFace Face object with bbox attribute
        
    Returns:
        Cropped face region as numpy array
    """
    bbox = getattr(face, 'bbox', None)
    if bbox is None:
        return img_bgr
    
    x1, y1, x2, y2 = map(int, bbox[:4])
    h, w = img_bgr.shape[:2]
    
    # Clamp to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    if x2 <= x1 or y2 <= y1:
        return img_bgr
        
    return img_bgr[y1:y2, x1:x2]


def evaluate_face_quality(
    img_bgr: np.ndarray,
    face,
    cfg: Optional[FaceQualityConfig] = None,
) -> FaceQualityResult:
    """
    Evaluate the quality of a detected face.
    
    This function checks multiple quality metrics against the provided
    configuration thresholds and returns a comprehensive result.
    
    Args:
        img_bgr: Full BGR image containing the face
        face: InsightFace Face object with bbox, pose, etc.
        cfg: Quality configuration. If None, uses VERIFY_QUALITY defaults.
        
    Returns:
        FaceQualityResult with is_usable, score, and detailed metrics
    """
    if cfg is None:
        cfg = VERIFY_QUALITY
    
    reasons: List[str] = []
    metrics: Dict[str, Any] = {}
    
    # Get face bounding box dimensions
    bbox = getattr(face, 'bbox', None)
    if bbox is None:
        return FaceQualityResult(
            is_usable=False,
            score=0.0,
            reasons=["no_bounding_box"],
        )
    
    x1, y1, x2, y2 = map(int, bbox[:4])
    face_width = x2 - x1
    face_height = y2 - y1
    
    # Check face size
    size_ok = face_width >= cfg.min_face_size and face_height >= cfg.min_face_size
    if not size_ok:
        reasons.append(f"face_too_small:{face_width}x{face_height}<{cfg.min_face_size}")
    metrics["face_size"] = (face_width, face_height)
    
    # Get face crop for quality analysis
    face_crop = get_face_crop(img_bgr, face)
    
    # Check blur
    blur = laplacian_variance(face_crop)
    blur_ok = blur >= cfg.min_blur_variance
    if not blur_ok:
        reasons.append(f"too_blurry:{blur:.1f}<{cfg.min_blur_variance}")
    metrics["blur_variance"] = blur
    
    # Check brightness
    brightness = calculate_brightness(face_crop)
    brightness_ok = cfg.min_brightness <= brightness <= cfg.max_brightness
    if not brightness_ok:
        reasons.append(f"bad_brightness:{brightness:.1f}")
    metrics["brightness"] = brightness
    
    # Check pose angles
    yaw, pitch, roll = extract_pose_angles(face)
    yaw_ok = abs(yaw) <= cfg.max_yaw
    pitch_ok = abs(pitch) <= cfg.max_pitch
    roll_ok = abs(roll) <= cfg.max_roll
    pose_ok = yaw_ok and pitch_ok and roll_ok
    
    if not yaw_ok:
        reasons.append(f"yaw_too_extreme:{yaw:.1f}>{cfg.max_yaw}")
    if not pitch_ok:
        reasons.append(f"pitch_too_extreme:{pitch:.1f}>{cfg.max_pitch}")
    if not roll_ok:
        reasons.append(f"roll_too_extreme:{roll:.1f}>{cfg.max_roll}")
    metrics["pose"] = {"yaw": yaw, "pitch": pitch, "roll": roll}
    
    # Calculate overall quality score (weighted average)
    scores = []
    
    # Size score (0-1)
    min_dim = min(face_width, face_height)
    size_score = min(1.0, min_dim / (cfg.min_face_size * 1.5))
    scores.append(size_score * 0.25)
    
    # Blur score (0-1)  
    blur_score = min(1.0, blur / (cfg.min_blur_variance * 2))
    scores.append(blur_score * 0.35)
    
    # Brightness score (0-1) - penalize extremes
    mid_bright = (cfg.min_brightness + cfg.max_brightness) / 2
    bright_range = (cfg.max_brightness - cfg.min_brightness) / 2
    bright_score = 1.0 - min(1.0, abs(brightness - mid_bright) / bright_range)
    scores.append(bright_score * 0.15)
    
    # Pose score (0-1) - penalize extreme angles
    pose_score = 1.0 - (abs(yaw) / 90 + abs(pitch) / 90 + abs(roll) / 90) / 3
    pose_score = max(0.0, pose_score)
    scores.append(pose_score * 0.25)
    
    overall_score = sum(scores)
    metrics["component_scores"] = {
        "size": size_score,
        "blur": blur_score,
        "brightness": bright_score,
        "pose": pose_score,
    }
    
    # Determine if usable
    quality_ok = overall_score >= cfg.min_overall_quality
    if not quality_ok and "overall_quality_low" not in [r.split(":")[0] for r in reasons]:
        reasons.append(f"overall_quality_low:{overall_score:.2f}<{cfg.min_overall_quality}")
    
    is_usable = size_ok and blur_ok and brightness_ok and pose_ok and quality_ok
    
    return FaceQualityResult(
        is_usable=is_usable,
        score=overall_score,
        reasons=reasons,
        blur_variance=blur,
        face_width=face_width,
        face_height=face_height,
        brightness=brightness,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        metrics=metrics,
    )


# Export all public symbols
__all__ = [
    "FaceQualityConfig",
    "FaceQualityResult",
    "evaluate_face_quality",
    "laplacian_variance",
    "ENROLL_QUALITY",
    "VERIFY_QUALITY", 
    "SEARCH_QUALITY",
    "CRAWLER_INGEST_QUALITY",
]

