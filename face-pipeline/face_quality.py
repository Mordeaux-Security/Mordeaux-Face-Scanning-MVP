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
    """Configuration for face quality evaluation.
    
    Tuned for production face search with millions of faces.
    Higher thresholds = better embedding quality = more accurate matches.
    """
    min_size: int = 80           # Minimum face size in pixels
    min_blur_var: float = 80.0   # Laplacian variance threshold (higher = sharper)
    max_yaw_deg: float = 25.0    # Max horizontal head rotation (tighter than before)
    max_pitch_deg: float = 25.0  # Max vertical head rotation (tighter than before)
    min_score: float = 0.65      # Minimum overall quality score


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


# =============================================================================
# Predefined Quality Configurations for Different Use Cases
# =============================================================================
# These are tuned for production accuracy. Higher thresholds = better embeddings
# = more accurate search results across millions of faces.

# CRAWLER_INGEST_QUALITY: Used when indexing faces from web crawls
# Stricter quality ensures only high-quality faces enter the database
CRAWLER_INGEST_QUALITY = FaceQualityConfig(
    min_size=80,           # Minimum 80px face for good embedding quality
    min_blur_var=80.0,     # Reasonably sharp faces only
    max_yaw_deg=25.0,      # Near-frontal faces for better recognition
    max_pitch_deg=25.0,    # Not looking too far up/down
    min_score=0.60         # Good overall quality
)

# ENROLL_QUALITY: Used when a user enrolls their identity (high quality needed)
# Further relaxed for real-world usage - allow more photos to pass
ENROLL_QUALITY = FaceQualityConfig(
    min_size=64,           # Face should be at least 64px (more lenient)
    min_blur_var=30.0,     # Much more tolerant blur threshold (was 80.0)
    max_yaw_deg=30.0,      # Allow more angle variation (was 25.0)
    max_pitch_deg=30.0,    # Allow more up/down angle (was 25.0)
    min_score=0.40         # Much more lenient quality bar (was 0.60)
)

# VERIFY_QUALITY: Used for 1:1 verification against enrolled identity
VERIFY_QUALITY = FaceQualityConfig(
    min_size=80,           # Slightly smaller than enrollment OK
    min_blur_var=60.0,     # Some blur tolerance for real-world selfies
    max_yaw_deg=25.0,      # Some pose variation OK
    max_pitch_deg=25.0,
    min_score=0.55         # Moderate quality bar
)

# SEARCH_QUALITY: Used for 1:N search queries (user searching database)
# Very lenient to match what crawler accepts - we want to find matches even with imperfect photos
SEARCH_QUALITY = FaceQualityConfig(
    min_size=32,           # Allow very small faces (crawler accepts small faces)
    min_blur_var=30.0,     # More blur tolerance (match ENROLL_QUALITY)
    max_yaw_deg=45.0,      # Very flexible pose (allow side profiles)
    max_pitch_deg=45.0,
    min_score=0.30         # Very low bar - just need a detectable face for search
)


