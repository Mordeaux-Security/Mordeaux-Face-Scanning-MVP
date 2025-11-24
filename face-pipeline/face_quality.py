"""
Face quality evaluation module.

Provides quality assessment for detected faces including blur, pose, and size checks.
"""

from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import cv2

from config.settings import settings
from pipeline.quality import laplacian_variance


@dataclass
class FaceQualityConfig:
    """Configuration for face quality evaluation."""
    min_size: int = 80
    min_blur_var: float = 120.0
    max_yaw_deg: float = 20.0
    max_pitch_deg: float = 20.0
    min_score: float = 0.7


@dataclass
class FaceQualityResult:
    """Result of face quality evaluation."""
    is_usable: bool
    score: float
    reasons: List[str]
    blur_var: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float


def evaluate_face_quality(
    img_bgr: np.ndarray,
    face,
    cfg: Optional[FaceQualityConfig] = None
) -> FaceQualityResult:
    """
    Evaluate quality of a detected face.
    
    Parameters:
        img_bgr: BGR image array
        face: InsightFace Face object with attributes like bbox, landmarks, yaw, pitch, roll
        cfg: Optional quality configuration. If None, uses defaults.
    
    Returns:
        FaceQualityResult with quality assessment
    """
    if cfg is None:
        cfg = FaceQualityConfig()
    
    reasons = []
    
    # Extract face bounding box
    bbox = getattr(face, 'bbox', None)
    if bbox is None:
        return FaceQualityResult(
            is_usable=False,
            score=0.0,
            reasons=["no_bbox"],
            blur_var=0.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0
        )
    
    # Extract face region
    x1, y1, x2, y2 = bbox[:4]
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
    # Check size
    face_w = x2 - x1
    face_h = y2 - y1
    if face_w < cfg.min_size or face_h < cfg.min_size:
        reasons.append(f"face_too_small({face_w}x{face_h})")
    
    # Extract face crop for blur check
    h, w = img_bgr.shape[:2]
    x1_clip = max(0, x1)
    y1_clip = max(0, y1)
    x2_clip = min(w, x2)
    y2_clip = min(h, y2)
    
    if x2_clip > x1_clip and y2_clip > y1_clip:
        face_crop = img_bgr[y1_clip:y2_clip, x1_clip:x2_clip]
        blur_var = laplacian_variance(face_crop)
        
        if blur_var < cfg.min_blur_var:
            reasons.append(f"blur_too_high({blur_var:.1f})")
    else:
        blur_var = 0.0
        reasons.append("face_out_of_bounds")
    
    # Extract pose angles (if available from InsightFace)
    # Handle None values that may be returned by some face models
    yaw_deg = getattr(face, 'yaw', None)
    pitch_deg = getattr(face, 'pitch', None)
    roll_deg = getattr(face, 'roll', None)
    
    # Default to 0.0 if None (pose not available)
    yaw_deg = yaw_deg if yaw_deg is not None else 0.0
    pitch_deg = pitch_deg if pitch_deg is not None else 0.0
    roll_deg = roll_deg if roll_deg is not None else 0.0
    
    # Check pose
    if abs(yaw_deg) > cfg.max_yaw_deg:
        reasons.append(f"yaw_too_large({yaw_deg:.1f}deg)")
    if abs(pitch_deg) > cfg.max_pitch_deg:
        reasons.append(f"pitch_too_large({pitch_deg:.1f}deg)")
    
    # Calculate quality score (higher is better)
    # Base score from blur (normalized to 0-1 range, assuming max blur_var ~500)
    blur_score = min(1.0, blur_var / 500.0)
    
    # Penalize for pose deviations
    yaw_score = 1.0 - min(1.0, abs(yaw_deg) / cfg.max_yaw_deg)
    pitch_score = 1.0 - min(1.0, abs(pitch_deg) / cfg.max_pitch_deg)
    
    # Size score (normalized)
    size_score = min(1.0, min(face_w, face_h) / max(cfg.min_size, 112))
    
    # Combined score (weighted average)
    score = (blur_score * 0.4 + yaw_score * 0.2 + pitch_score * 0.2 + size_score * 0.2)
    
    # Determine if usable
    is_usable = len(reasons) == 0 and score >= cfg.min_score
    
    if is_usable:
        reasons = ["ok"]
    
    return FaceQualityResult(
        is_usable=is_usable,
        score=score,
        reasons=reasons,
        blur_var=blur_var,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg
    )


# Predefined quality configurations for different use cases
# Note: blur_var thresholds lowered for real-world phone camera photos
ENROLL_QUALITY = FaceQualityConfig(
    min_size=80,  # Reasonable minimum face size
    min_blur_var=30.0,  # Lowered for typical phone photos (was 150)
    max_yaw_deg=25.0,  # Allow some head turn
    max_pitch_deg=25.0,
    min_score=0.5
)

VERIFY_QUALITY = FaceQualityConfig(
    min_size=64,
    min_blur_var=25.0,  # Lowered for typical phone photos (was 120)
    max_yaw_deg=30.0,
    max_pitch_deg=30.0,
    min_score=0.45
)

SEARCH_QUALITY = FaceQualityConfig(
    min_size=48,  # More lenient for search
    min_blur_var=20.0,  # Lowered for typical phone photos (was 100)
    max_yaw_deg=40.0,  # More lenient pose
    max_pitch_deg=40.0,
    min_score=0.4
)


