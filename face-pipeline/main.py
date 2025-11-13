import os
from typing import List, Optional, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScoredPoint, SearchParams

from pipeline.ensure import ensure_all
from pipeline.insight import get_app
from pipeline.image_utils import decode_image_b64
from pipeline.storage import presign
from pipeline.quality import laplacian_variance
from config.settings import settings
import numpy as np
import cv2
import math

app = FastAPI(title="face-pipeline")

# ----- Config helpers -----
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "faces_v1")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "512"))

# ----- Models -----
class SearchReq(BaseModel):
    tenant_id: str
    vector: Optional[List[float]] = None
    image_b64: Optional[str] = None
    top_k: int = Field(default=50, ge=1, le=200)
    threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional threshold override; defaults to 0.75 with adaptive adjustments")
    mode: str = Field(default="standard")  # kept for compatibility

# ----- Health -----
@app.get("/api/v1/health")
def health():
    return {"status": "healthy", "service": "face-pipeline-search-api"}

@app.get("/healthz")
def healthz():
    return {"status": "healthy", "service": "face-pipeline"}

@app.on_event("startup")
def startup():
    ensure_all()  # ensure MinIO buckets + Qdrant collection

# ----- Quality assessment -----
def estimate_yaw_angle(landmarks: np.ndarray) -> float:
    """
    Estimate yaw angle (head rotation left/right) from facial landmarks.
    Uses the eye positions to estimate horizontal rotation.
    
    Args:
        landmarks: Array of shape (5, 2) with [left_eye, right_eye, nose, left_mouth, right_mouth]
    
    Returns:
        Yaw angle in degrees (positive = turned right, negative = turned left, 0 = frontal)
    """
    if landmarks is None or landmarks.shape[0] < 2:
        return 0.0
    
    try:
        # Get eye positions (indices 0 and 1 for left and right eye)
        left_eye = landmarks[0] if landmarks.shape[0] > 0 else None
        right_eye = landmarks[1] if landmarks.shape[0] > 1 else None
        
        if left_eye is None or right_eye is None:
            return 0.0
        
        # Calculate angle between eyes
        eye_vector = right_eye - left_eye
        eye_distance = np.linalg.norm(eye_vector)
        
        if eye_distance < 1e-6:
            return 0.0
        
        # Estimate yaw: if eyes are not horizontal, face is rotated
        # Normal eye line should be roughly horizontal
        angle_rad = math.asin(np.clip(eye_vector[1] / eye_distance, -1.0, 1.0))
        angle_deg = math.degrees(angle_rad)
        
        # Also check if one eye is significantly further from center than the other
        # This indicates rotation
        nose_x = landmarks[2][0] if landmarks.shape[0] > 2 else (left_eye[0] + right_eye[0]) / 2
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        offset = abs(face_center_x - nose_x)
        eye_span = abs(right_eye[0] - left_eye[0])
        
        if eye_span > 1e-6:
            # Additional yaw estimate based on eye-to-nose offset
            yaw_from_offset = math.degrees(math.atan2(offset, eye_span / 2))
            # Combine both estimates (weighted average)
            angle_deg = (angle_deg * 0.3 + yaw_from_offset * 0.7)
        
        return float(angle_deg)
    except Exception:
        return 0.0

def assess_image_quality(img: np.ndarray, face_result) -> dict:
    """
    Assess image quality metrics for adaptive threshold adjustment.
    Assesses blur on the face crop region, not the full image.
    
    Returns:
        dict with keys: blur (float), yaw (float), is_low_quality (bool)
    """
    # Get landmarks from face result
    yaw = 0.0
    blur_var = 0.0
    
    # Extract face region for blur assessment
    if hasattr(face_result, 'bbox') and face_result.bbox is not None:
        bbox = face_result.bbox
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        # Ensure coordinates are within image bounds
        h, w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 > x1 and y2 > y1:
            face_crop = img[y1:y2, x1:x2]
            blur_var = laplacian_variance(face_crop)
        else:
            # Fallback to full image if bbox is invalid
            blur_var = laplacian_variance(img)
    else:
        # Fallback to full image if no bbox
        blur_var = laplacian_variance(img)
    
    # Calculate yaw angle from landmarks
    if hasattr(face_result, 'kps') and face_result.kps is not None:
        landmarks = np.array(face_result.kps)
        yaw = estimate_yaw_angle(landmarks)
    
    # Determine if low quality (blur or excessive yaw)
    # Low blur threshold: < 120.0 (standard threshold from quality.py)
    # High yaw threshold: > 20 degrees
    is_low_quality = blur_var < 120.0 or abs(yaw) > 20.0
    
    return {
        "blur": float(blur_var),
        "yaw": float(yaw),
        "is_low_quality": is_low_quality,
    }

# ----- Embedding -----
def embed_from_image_b64(data_url: str) -> tuple[np.ndarray, dict]:
    """
    1) Decode image
    2) Detect faces
    3) Pick best face by det_score
    4) Use FaceAnalysis recognition model for 512-D normed embedding
    5) Assess quality metrics (blur, yaw)
    
    Returns:
        tuple of (vector, quality_info) where:
        - vector: float32 np.ndarray of shape (512,) (L2-normalized)
        - quality_info: dict with blur, yaw, is_low_quality
    """
    img = decode_image_b64(data_url)
    if img is None:
        raise HTTPException(400, "invalid_image_data")

    app_ = get_app()
    faces = app_.get(img)  # detection + alignment + recognition (if pack includes it)

    if not faces:
        raise HTTPException(422, "no_face_detected")

    # pick the face with highest detection score
    best = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))

    # Assess quality
    quality_info = assess_image_quality(img, best)

    # prefer normed_embedding if available; else embedding and then normalize
    vec = getattr(best, "normed_embedding", None)
    if vec is None:
        raw = getattr(best, "embedding", None)
        if raw is None:
            raise HTTPException(500, "recognition_model_not_loaded")
        arr = np.array(raw, dtype=np.float32)
        n = np.linalg.norm(arr) + 1e-9
        vec = arr / n
    else:
        vec = np.array(vec, dtype=np.float32)

    if vec.shape[0] != VECTOR_DIM:
        raise HTTPException(500, f"vector_dim_mismatch: got {vec.shape[0]}, expected {VECTOR_DIM}")

    return vec, quality_info

# ----- Presigned URL helpers -----
def _derive_keys_from_face_id(face_id: str, tenant_id: str) -> tuple[Optional[str], Optional[str]]:
    """
    Derive crop_key and thumb_key from face_id.
    Format: {image_sha256}:face_{i} -> {tenant_id}/{image_sha256}_face_{i}.jpg
    """
    try:
        if ":" in str(face_id):
            sha256, suffix = str(face_id).split(":", 1)
            base = f"{sha256}_{suffix}"
            crop_key = f"{tenant_id}/{base}.jpg"
            thumb_key = f"{tenant_id}/{base}_thumb.jpg"
            return crop_key, thumb_key
    except Exception:
        pass
    return None, None

def _derive_keys_from_payload(payload: Dict[str, Any], tenant_id: str) -> tuple[Optional[str], Optional[str]]:
    """
    Derive crop_key and thumb_key from payload.
    Prefers stored keys, falls back to face_id, then image_sha256.
    """
    # Try stored keys first
    crop_key = payload.get("crop_key")
    thumb_key = payload.get("thumb_key")
    if crop_key and thumb_key:
        return crop_key, thumb_key
    
    # Try deriving from face_id in payload
    face_id = payload.get("face_id")
    if face_id:
        return _derive_keys_from_face_id(face_id, tenant_id)
    
    # Fallback: derive from image_sha256 (assumes face_0 for single-face images)
    image_sha256 = payload.get("image_sha256")
    if image_sha256:
        base = f"{image_sha256}_face_0"
        crop_key = f"{tenant_id}/{base}.jpg"
        thumb_key = f"{tenant_id}/{base}_thumb.jpg"
        return crop_key, thumb_key
    
    return None, None

def _get_presigned_urls(hit: ScoredPoint, tenant_id: str) -> Dict[str, Optional[str]]:
    """Generate presigned URLs for thumb and crop from hit payload."""
    payload = hit.payload or {}
    
    # Derive keys from payload (tries stored keys, then face_id, then image_sha256)
    crop_key, thumb_key = _derive_keys_from_payload(payload, tenant_id)
    
    urls: Dict[str, Optional[str]] = {"thumb_url": None, "crop_url": None}
    
    try:
        if thumb_key:
            urls["thumb_url"] = presign(settings.MINIO_BUCKET_THUMBS, thumb_key)
    except Exception:
        pass
    
    try:
        if crop_key:
            urls["crop_url"] = presign(settings.MINIO_BUCKET_CROPS, crop_key)
    except Exception:
        pass
    
    return urls

# ----- Qdrant search -----
def qdrant_search(vec: np.ndarray, tenant_id: str, top_k: int, threshold: float, hnsw_ef: int = 128) -> List[ScoredPoint]:
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    # Safety: ensure unit norm for cosine
    v = vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    flt = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])
    
    # Create SearchParams with hnsw_ef for calibration (128-256)
    search_params = SearchParams(hnsw_ef=hnsw_ef)
    
    hits = qc.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=v.tolist(),
        limit=top_k * 3,  # Fetch more to allow grouping, then limit
        score_threshold=threshold,
        query_filter=flt,
        search_params=search_params,  # HNSW ef parameter for search quality/speed tradeoff
        with_payload=True,
        with_vectors=False,
    )
    return hits

# ----- Adaptive threshold calculation -----
def calculate_adaptive_threshold(base_threshold: float, quality_info: Optional[dict] = None) -> float:
    """
    Calculate adaptive threshold based on quality metrics.
    
    Args:
        base_threshold: Base threshold (default 0.75)
        quality_info: Optional quality info dict with is_low_quality flag
    
    Returns:
        Adjusted threshold
    """
    threshold = base_threshold
    
    # If quality info is available and indicates low quality, lower threshold
    if quality_info and quality_info.get("is_low_quality", False):
        threshold = 0.70  # Lower threshold for low-quality queries
    
    return threshold

# ----- API: /api/v1/search -----
@app.post("/api/v1/search")
def search(req: SearchReq):
    if not req.vector and not req.image_b64:
        raise HTTPException(422, "Provide vector or image_b64")

    # Default threshold is 0.75 (calibration quick-win)
    base_threshold = req.threshold if req.threshold is not None else 0.75
    quality_info = None

    if req.image_b64:
        vec, quality_info = embed_from_image_b64(req.image_b64)
    else:
        # accept direct vector; enforce length + norm
        if len(req.vector) != VECTOR_DIM:
            raise HTTPException(422, f"vector length must be {VECTOR_DIM}")
        arr = np.array(req.vector, dtype=np.float32)
        vec = arr / (np.linalg.norm(arr) + 1e-9)

    # Calculate adaptive threshold
    adaptive_threshold = calculate_adaptive_threshold(base_threshold, quality_info)
    
    # Determine hnsw_ef based on top_k (calibration: 128-256)
    # Use higher ef for larger top_k to maintain quality
    hnsw_ef = 128 if req.top_k <= 50 else 256
    
    hits = qdrant_search(vec, tenant_id=req.tenant_id, top_k=req.top_k, threshold=adaptive_threshold, hnsw_ef=hnsw_ef)

    # Group by p_hash_prefix to collapse near-duplicates
    groups: Dict[str, List[ScoredPoint]] = defaultdict(list)
    for hit in hits:
        payload = hit.payload or {}
        prefix = payload.get("p_hash_prefix", "unknown")
        groups[prefix].append(hit)
    
    # For each group, pick the highest-scoring hit as representative
    grouped_hits: List[Dict[str, Any]] = []
    for prefix, group_hits in groups.items():
        # Sort by score descending, take best
        best = max(group_hits, key=lambda h: h.score)
        
        # Generate presigned URLs
        urls = _get_presigned_urls(best, req.tenant_id)
        
        grouped_hits.append({
            "id": str(best.id),
            "score": float(best.score),
            "payload": best.payload or {},
            "p_hash_prefix": prefix,
            "group_count": len(group_hits),  # Number of near-duplicates in this group
            "thumb_url": urls["thumb_url"],
            "crop_url": urls["crop_url"],
        })
    
    # Sort by score and limit to top_k
    grouped_hits.sort(key=lambda x: x["score"], reverse=True)
    grouped_hits = grouped_hits[:req.top_k]
    
    return {
        "query": {
            "tenant_id": req.tenant_id,
            "search_mode": "image" if req.image_b64 else "vector",
            "mode": req.mode,
            "top_k": req.top_k,
            "threshold": adaptive_threshold,  # Return the actually used threshold
            "base_threshold": base_threshold,
            "quality": quality_info,  # Include quality info if available
        },
        "hits": grouped_hits,
        "count": len(grouped_hits),
    }
