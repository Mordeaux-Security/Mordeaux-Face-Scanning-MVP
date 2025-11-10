import os
from typing import List, Optional, Dict, Any
from collections import defaultdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, ScoredPoint

from pipeline.ensure import ensure_all
from pipeline.insight import get_app
from pipeline.image_utils import decode_image_b64
from pipeline.storage import presign
from config.settings import settings
import numpy as np

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
    threshold: float = Field(default=0.70, ge=0.0, le=1.0)
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

# ----- Embedding -----
def embed_from_image_b64(data_url: str) -> np.ndarray:
    """
    1) Decode image
    2) Detect faces
    3) Pick best face by det_score
    4) Use FaceAnalysis recognition model for 512-D normed embedding
    Returns float32 np.ndarray of shape (512,) (L2-normalized).
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

    return vec

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
def qdrant_search(vec: np.ndarray, tenant_id: str, top_k: int, threshold: float) -> List[ScoredPoint]:
    qc = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)
    # Safety: ensure unit norm for cosine
    v = vec.astype(np.float32)
    v = v / (np.linalg.norm(v) + 1e-9)
    flt = Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])
    hits = qc.search(
        collection_name=QDRANT_COLLECTION,
        query_vector=v.tolist(),
        limit=top_k * 3,  # Fetch more to allow grouping, then limit
        score_threshold=threshold,
        query_filter=flt,
        with_payload=True,
        with_vectors=False,
    )
    return hits

# ----- API: /api/v1/search -----
@app.post("/api/v1/search")
def search(req: SearchReq):
    if not req.vector and not req.image_b64:
        raise HTTPException(422, "Provide vector or image_b64")

    if req.image_b64:
        vec = embed_from_image_b64(req.image_b64)
    else:
        # accept direct vector; enforce length + norm
        if len(req.vector) != VECTOR_DIM:
            raise HTTPException(422, f"vector length must be {VECTOR_DIM}")
        arr = np.array(req.vector, dtype=np.float32)
        vec = arr / (np.linalg.norm(arr) + 1e-9)

    hits = qdrant_search(vec, tenant_id=req.tenant_id, top_k=req.top_k, threshold=req.threshold)

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
            "threshold": req.threshold,
        },
        "hits": grouped_hits,
        "count": len(grouped_hits),
    }
