"""
Helper functions for face detection with quality evaluation.

This module provides utilities to decode images, detect faces, evaluate quality,
and select the best usable face for embedding generation.
"""

from typing import List, Tuple, Optional
import numpy as np
from fastapi import HTTPException, status

from pipeline.image_utils import decode_image_b64
from pipeline.detector import detect_faces_raw
import sys
from pathlib import Path

# Import face_quality from parent directory (face-pipeline root)
sys.path.insert(0, str(Path(__file__).parent.parent))
from face_quality import evaluate_face_quality, FaceQualityConfig, FaceQualityResult


class DetectedFaceWithQuality:
    """
    Wrapper that combines an InsightFace Face object with its quality evaluation result.
    """
    def __init__(self, face, quality: FaceQualityResult):
        self.face = face
        self.quality = quality


def analyze_faces_with_quality(
    image_b64: str,
    quality_cfg: Optional[FaceQualityConfig] = None,
) -> Tuple[np.ndarray, List[DetectedFaceWithQuality]]:
    """
    Decode an image and run InsightFace detection + quality evaluation
    for each detected face.

    Parameters:
        image_b64: Base64-encoded image string (data URL or raw base64)
        quality_cfg: Optional quality configuration. If None, uses defaults.

    Returns:
        Tuple of:
        - img: numpy array (BGR format, as returned by decode_image_b64)
        - faces_with_quality: list of DetectedFaceWithQuality objects

    Raises:
        ValueError: If image cannot be decoded
    """
    img = decode_image_b64(image_b64)
    if img is None:
        raise ValueError("Failed to decode image from base64 data")

    # Run detection using the wrapped detector helper
    faces = detect_faces_raw(img)

    # Evaluate quality for each detected face
    faces_with_quality: List[DetectedFaceWithQuality] = []
    for f in faces:
        q = evaluate_face_quality(img, f, cfg=quality_cfg)
        faces_with_quality.append(DetectedFaceWithQuality(f, q))

    return img, faces_with_quality


def pick_best_usable_face(
    faces_with_quality: List[DetectedFaceWithQuality],
) -> Optional[DetectedFaceWithQuality]:
    """
    Pick the 'best' usable face by:
      - filtering to is_usable faces
      - choosing the one with highest quality.score

    Parameters:
        faces_with_quality: List of DetectedFaceWithQuality objects

    Returns:
        The best usable face, or None if no usable faces found
    """
    usable = [fwq for fwq in faces_with_quality if fwq.quality.is_usable]
    if not usable:
        return None
    return max(usable, key=lambda fwq: fwq.quality.score)


def embed_one_b64_strict(
    image_b64: str,
    require_single_face: bool = False,
    quality_cfg: Optional[FaceQualityConfig] = None,
) -> Tuple[np.ndarray, DetectedFaceWithQuality]:
    """
    Embed faces with quality gating.
    - If require_single_face=True, we enforce exactly one *usable* face.
    - If False, we just pick the best usable face.
    
    Parameters:
        image_b64: Base64-encoded image string (data URL or raw base64)
        require_single_face: If True, requires exactly one usable face (for identity endpoints)
        quality_cfg: Optional quality configuration. If None, uses defaults.
    
    Returns:
        Tuple of:
        - embedding: numpy array (512-dim float32, L2 normalized)
        - face_with_quality: DetectedFaceWithQuality object for the selected face
    
    Raises:
        HTTPException: With structured error details if no usable faces or multiple faces when required
    """
    img, faces_with_quality = analyze_faces_with_quality(image_b64, quality_cfg)

    usable_faces = [fwq for fwq in faces_with_quality if fwq.quality.is_usable]

    if not usable_faces:
        # No usable faces -> return a clean error with reasons from all faces
        all_reasons = []
        for fwq in faces_with_quality:
            if fwq.quality.reasons:
                all_reasons.extend(fwq.quality.reasons)
        detail = {
            "error": "no_usable_faces",
            "reasons": list(set(all_reasons)) if all_reasons else ["no_faces_detected"],
        }
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

    if require_single_face:
        if len(usable_faces) > 1:
            detail = {
                "error": "multiple_faces_detected",
                "message": "Please upload a photo with only your face visible.",
                "num_usable_faces": len(usable_faces),
            }
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=detail)

        fwq = usable_faces[0]
    else:
        fwq = max(usable_faces, key=lambda fwq: fwq.quality.score)

    face = fwq.face

    # Embed
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        raw_emb = getattr(face, "embedding", None)
        if raw_emb is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"error": "recognition_model_not_loaded"},
            )
        raw_emb = np.array(raw_emb, dtype=np.float32)
        norm = np.linalg.norm(raw_emb)
        if norm == 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"error": "zero_norm_embedding"},
            )
        emb = raw_emb / norm
    else:
        emb = np.array(emb, dtype=np.float32)

    return emb.astype(np.float32), fwq

