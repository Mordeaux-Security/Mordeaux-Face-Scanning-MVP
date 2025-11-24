import os
import logging
from typing import List, Optional, Dict, Any
import hashlib
import uuid

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Filter, FieldCondition, MatchValue, ScoredPoint, PointStruct, SearchParams, Range
)

from logging_utils import setup_logging, log_event
from pipeline.ensure import ensure_all
from face_quality import (
    ENROLL_QUALITY,
    VERIFY_QUALITY,
    SEARCH_QUALITY,
)
from pipeline.face_helpers import embed_one_b64_strict

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="face-pipeline")

# ----- Config -----
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
FACES_COLLECTION = os.getenv("QDRANT_COLLECTION", "faces_v1")
IDENTITY_COLLECTION = os.getenv("IDENTITY_COLLECTION", "identities_v1")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "512"))

# Verification threshold for identity-safe search (configurable via env)
# Supports both VERIFY_THRESHOLD and VERIFY_HI_THRESHOLD for backward compatibility
VERIFY_THRESHOLD_DEFAULT = float(os.getenv("VERIFY_THRESHOLD") or os.getenv("VERIFY_HI_THRESHOLD", "0.78"))

# Minimum quality score threshold for identity search (optional, defaults to 0.0 = no minimum)
IDENTITY_SEARCH_MIN_QUALITY = float(os.getenv("IDENTITY_SEARCH_MIN_QUALITY", "0.0"))

# ----- Models -----
class SearchReq(BaseModel):
    tenant_id: str
    vector: Optional[List[float]] = None
    image_b64: Optional[str] = None
    top_k: int = Field(default=50, ge=1, le=200)
    threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    mode: str = Field(default="standard")

class EnrollReq(BaseModel):
    tenant_id: str
    identity_id: str
    images_b64: List[str] = Field(..., min_items=2, max_items=10)  # recommend 3–5
    overwrite: bool = Field(default=True)

class VerifyReq(BaseModel):
    tenant_id: str
    identity_id: str
    image_b64: str
    # thresholds: hi = strong accept, lo = borderline accept if corroborated (not used here but exposed)
    hi_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=1, le=200)

class IdentitySafeSearchReq(BaseModel):
    tenant_id: str
    identity_id: str
    image_b64: str

    top_k: int = 50
    min_score: float = 0.0


class IdentitySafeSearchResult(BaseModel):
    """
    Result type for identity_safe_search. This MUST be compatible with the
    existing search result model used in /api/v1/search, so reuse or
    mirror its fields exactly.
    """
    id: str
    score: float
    image_id: Optional[str] = None
    tenant_id: Optional[str] = None
    identity_id: Optional[str] = None
    payload: Optional[dict] = None


class IdentitySafeSearchResp(BaseModel):
    verified: bool
    similarity: float
    threshold: float
    reason: Optional[dict] = None
    results: List[IdentitySafeSearchResult] = []

# ----- Health -----
@app.get("/api/v1/health")
def health():
    return {"status": "healthy", "service": "face-pipeline-search-api"}

@app.get("/healthz")
def healthz():
    return {"status": "healthy", "service": "face-pipeline"}

@app.on_event("startup")
def startup():
    ensure_all()

# ----- Embedding -----
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))

# ----- Qdrant helpers -----
def _qc() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)

def _tenant_filter(tenant_id: str) -> Filter:
    return Filter(must=[FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id))])

def _tenant_identity_filter(tenant_id: str, identity_id: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
        FieldCondition(key="identity_id", match=MatchValue(value=identity_id)),
    ])

def _tenant_identity_quality_filter(tenant_id: str, identity_id: str) -> Filter:
    """
    Create a filter for tenant_id, identity_id, quality_is_usable == True,
    and optionally quality_score >= IDENTITY_SEARCH_MIN_QUALITY.
    
    This ensures identity-safe search only returns high-quality faces.
    """
    conditions = [
        FieldCondition(key="tenant_id", match=MatchValue(value=tenant_id)),
        FieldCondition(key="identity_id", match=MatchValue(value=identity_id)),
        FieldCondition(
            key="quality_is_usable",
            match=MatchValue(value=True),
        ),
    ]
    
    # Optionally add quality score threshold if configured
    if IDENTITY_SEARCH_MIN_QUALITY > 0.0:
        conditions.append(
            FieldCondition(
                key="quality_score",
                range=Range(gte=IDENTITY_SEARCH_MIN_QUALITY),
            )
        )
    
    return Filter(must=conditions)

# ----- Internal helpers for identity-safe search -----
def _fetch_identity_centroid(tenant_id: str, identity_id: str) -> Optional[np.ndarray]:
    """
    Fetch the centroid vector for (tenant_id, identity_id) from identities_v1.

    Reuse the logic that /api/v1/verify currently uses to fetch the
    identity centroid from Qdrant. This helper should:

    - Construct the same point ID format used during enrollment
      (e.g., f"{tenant_id}:{identity_id}").
    - Query the 'identities_v1' collection.
    - Return the vector as a 1D numpy array or list[float], or None
      if the identity is not found.
    
    Implementation reuses the exact logic from /api/v1/verify endpoint
    (search + scroll fallback pattern).
    """
    qc = _qc()
    try:
        # Reuse exact logic from /api/v1/verify: Try search first (more reliable with proper vector)
        ids = qc.search(
            collection_name=IDENTITY_COLLECTION,
            query_vector=[0.0]*VECTOR_DIM,  # dummy; we use filtering + scroll alternative below if needed
            limit=1,
            score_threshold=0.0,
            query_filter=_tenant_identity_filter(tenant_id, identity_id),
            with_payload=True,
            with_vectors=True,  # need the vector
            search_params=SearchParams(hnsw_ef=64, exact=True),
        )
        
        # If search returns nothing (because exact=True with dummy vector), fallback to scroll
        # Same pattern as /api/v1/verify endpoint
        if not ids:
            res = qc.scroll(
                collection_name=IDENTITY_COLLECTION,
                scroll_filter=_tenant_identity_filter(tenant_id, identity_id),
                with_payload=True,
                with_vectors=True,
                limit=1
            )
            id_hits = res[0]
            if not id_hits:
                return None  # Helper returns None instead of raising HTTPException
            centroid_vec = np.array(id_hits[0].vector, dtype=np.float32)
        else:
            centroid_vec = np.array(ids[0].vector, dtype=np.float32)
        
        # Normalize centroid (exact same normalization as /verify)
        centroid_vec = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-9)
        return centroid_vec
    except Exception:
        return None


def _search_faces_for_identity(
    query_vector: List[float],
    tenant_id: str,
    identity_id: str,
    top_k: int,
    min_score: float,
) -> List[ScoredPoint]:
    """
    Run a Qdrant search on faces_v1 restricted to this (tenant_id, identity_id).

    Reuse the existing /api/v1/search implementation:
        - same vector search parameters,
        - same client usage,
        - same mapping from Qdrant results to Python objects.

    Differences vs generic search:
        - Always filter by tenant_id == tenant_id
        - Always filter by identity_id == identity_id
        - Optionally filter by quality fields, e.g.:
            quality_is_usable == True
            quality_score >= some threshold (optional, can be added later)

    This function should return the raw match objects used in /search
    so we can convert them into IdentitySafeSearchResult later.
    
    Args:
        query_vector: Embedding vector (512-dim float list) - already normalized
        tenant_id: Tenant identifier
        identity_id: Identity identifier
        top_k: Maximum number of results
        min_score: Minimum similarity score threshold (same as /search's threshold)
    
    Returns:
        List of ScoredPoint objects from Qdrant search (same format as /search)
    """
    qc = _qc()
    # Reuse exact Qdrant search logic from /api/v1/search endpoint
    # Same parameters, but with identity filter added (tenant_id + identity_id + quality_is_usable)
    # Use quality filter to ensure only usable faces are returned
    hits = qc.search(
        collection_name=FACES_COLLECTION,
        query_vector=query_vector,  # Already normalized list from caller
        limit=top_k,  # Same as req.top_k in /search
        score_threshold=min_score,  # Same as req.threshold in /search
        query_filter=_tenant_identity_quality_filter(tenant_id, identity_id),  # Filters by tenant_id, identity_id, and quality_is_usable == True
        with_payload=True,  # Same as /search
        with_vectors=False,  # Same as /search
        search_params=SearchParams(hnsw_ef=128, exact=False),  # Exact same params as /search
    )
    return hits

# ----- Public: search (existing) -----
@app.post("/api/v1/search")
def search(req: SearchReq):
    """
    Search for similar faces by vector or image.
    
    For image input, uses quality evaluation to ensure we search on usable faces only.
    Picks the best usable face when multiple faces are detected.
    """
    qc = _qc()
    
    if req.vector is not None:
        query_vec = np.asarray(req.vector, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        if len(query_vec) != VECTOR_DIM:
            raise HTTPException(422, f"vector length must be {VECTOR_DIM}")
    
    elif req.image_b64 is not None:
        # Allow multiple faces but pick best usable
        query_vec, fwq = embed_one_b64_strict(
            req.image_b64,
            require_single_face=False,
            quality_cfg=SEARCH_QUALITY,
        )
    
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "missing_vector_or_image"},
        )

    # ... existing Qdrant search logic here (unchanged)
    hits = qc.search(
        collection_name=FACES_COLLECTION,
        query_vector=query_vec.tolist(),
        limit=req.top_k,
        score_threshold=req.threshold,
        query_filter=_tenant_filter(req.tenant_id),
        with_payload=True,
        with_vectors=False,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )
    
    # Generate URLs for thumbnails/crops/original images
    from pipeline.storage import presign
    out = []
    for h in hits:
        payload = h.payload or {}
        result = {
            "id": str(h.id),
            "score": float(h.score),
            "similarity_pct": round(float(h.score) * 100, 1),
            "payload": payload,
        }
        
        # Try to get thumbnail URL
        if payload.get("thumb_key"):
            try:
                result["thumb_url"] = presign("face-crops", payload["thumb_key"])
            except Exception:
                result["thumb_url"] = None
        
        # Try to get crop URL
        if payload.get("crop_key"):
            try:
                result["crop_url"] = presign("face-crops", payload["crop_key"])
            except Exception:
                result["crop_url"] = None
        
        # Always provide original image URL as fallback
        # Original URL is stored as s3://bucket/key format
        original_url = payload.get("url", "")
        if original_url.startswith("s3://"):
            # Parse s3://bucket/key format
            parts = original_url[5:].split("/", 1)
            if len(parts) == 2:
                bucket, key = parts
                try:
                    result["image_url"] = presign(bucket, key)
                except Exception:
                    result["image_url"] = None
        
        out.append(result)
    
    return {
        "query": {"tenant_id": req.tenant_id, "search_mode": "image" if req.image_b64 else "vector",
                  "mode": req.mode, "top_k": req.top_k, "threshold": req.threshold},
        "hits": out, "count": len(out)
    }

# ----- Public: enroll identity -----
@app.post("/api/v1/enroll_identity")
def enroll_identity(req: EnrollReq):
    """
    Enroll an identity with multiple high-quality face images.
    
    Requires at least 3 valid images that pass quality checks.
    Uses stricter quality thresholds for enrollment to ensure better identity representation.
    """
    embeddings = []
    failed = {}

    for idx, img_b64 in enumerate(req.images_b64):
        try:
            vec, fwq = embed_one_b64_strict(
                img_b64,
                require_single_face=True,
                quality_cfg=ENROLL_QUALITY,
            )
            embeddings.append(vec)
        except HTTPException as e:
            failed[idx] = e.detail

    # Require at least 3 good enrollment images (tunable)
    if len(embeddings) < 3:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "not_enough_high_quality_images",
                "num_valid": len(embeddings),
                "min_required": 3,
                "failed_images": failed,
            },
        )

    # Compute centroid from valid embeddings
    emb_array = np.stack(embeddings, axis=0).astype(np.float32)
    centroid = emb_array.mean(axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    # Write centroid to Qdrant identities_v1 collection
    qc = _qc()
    # Generate UUID from tenant_id:identity_id string (for Qdrant compatibility)
    identity_key = f"{req.tenant_id}:{req.identity_id}"
    hash_obj = hashlib.sha256(identity_key.encode())
    hex_dig = hash_obj.hexdigest()
    point_id = uuid.UUID(hex_dig[:32])
    
    payload = {"tenant_id": req.tenant_id, "identity_id": req.identity_id}
    pts = [PointStruct(id=str(point_id), vector=centroid.tolist(), payload=payload)]
    qc.upsert(collection_name=IDENTITY_COLLECTION, points=pts, wait=True)

    return {
        "ok": True,
        "identity": {"tenant_id": req.tenant_id, "identity_id": req.identity_id},
        "vector_dim": VECTOR_DIM,
        "num_images_used": len(embeddings),
        "num_images_submitted": len(req.images_b64),
    }

# ----- Public: verify probe belongs to identity; return ONLY that identity's faces -----
@app.post("/api/v1/verify")
def verify(req: VerifyReq):
    """
    Verify that a probe image belongs to a specific identity.
    
    Requires a single high-quality face in the probe image.
    Returns verified=false with clear reason if image quality is insufficient,
    prompting user to try a better selfie.
    """
    # 1) Embed probe with quality gating - handle errors gracefully
    try:
        probe_vec, fwq = embed_one_b64_strict(
            req.image_b64,
            require_single_face=True,
            quality_cfg=VERIFY_QUALITY,
        )
    except HTTPException as e:
        # Instead of blowing up the request, return verified=false with a reason
        return {
            "verified": False,
            "similarity": 0.0,
            "reason": e.detail,
            "results": [],
        }

    qc = _qc()
    # 2) Fetch identity centroid from Qdrant identities_v1
    ids = qc.search(
        collection_name=IDENTITY_COLLECTION,
        query_vector=[0.0]*VECTOR_DIM,  # dummy; we use filtering + scroll alternative below if needed
        limit=1,
        score_threshold=0.0,
        query_filter=_tenant_identity_filter(req.tenant_id, req.identity_id),
        with_payload=True,
        with_vectors=True,  # need the vector
        search_params=SearchParams(hnsw_ef=64, exact=True),
    )
    # If search returns nothing (because exact=True with dummy vector), fallback to scroll
    if not ids:
        res = qc.scroll(
            collection_name=IDENTITY_COLLECTION,
            scroll_filter=_tenant_identity_filter(req.tenant_id, req.identity_id),
            with_payload=True, with_vectors=True, limit=1
        )
        id_hits = res[0]
        if not id_hits:
            raise HTTPException(404, "identity_not_enrolled")
        centroid_vec = np.array(id_hits[0].vector, dtype=np.float32)
    else:
        centroid_vec = np.array(ids[0].vector, dtype=np.float32)

    centroid_vec = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-9)

    # 3) Compute cosine similarity
    sim = float(np.dot(probe_vec, centroid_vec))

    # 4) Enforce similarity threshold (default 0.78)
    if sim < req.hi_threshold:
        return {
            "verified": False,
            "similarity": sim,
            "threshold": req.hi_threshold,
            "tenant_id": req.tenant_id,
            "identity_id": req.identity_id,
            "reason": {"error": "low_similarity", "threshold": req.hi_threshold},
            "results": [],
            "count": 0,
        }

    # 4) If verified, run restricted search to fetch ONLY this identity's faces from faces_v1
    hits = qc.search(
        collection_name=FACES_COLLECTION,
        query_vector=probe_vec.tolist(),  # or centroid_vec.tolist(); probe_vec is fine
        limit=req.top_k,
        score_threshold=0.0,  # we'll filter post by identity
        query_filter=_tenant_identity_filter(req.tenant_id, req.identity_id),
        with_payload=True,
        with_vectors=False,
        search_params=SearchParams(hnsw_ef=128, exact=False),
    )
    faces = [{"id": str(h.id), "score": float(h.score), "payload": h.payload or {}} for h in hits]

    return {
        "verified": True,
        "similarity": sim,
        "threshold": req.hi_threshold,
        "tenant_id": req.tenant_id,
        "identity_id": req.identity_id,
        "reason": None,
        "results": faces,
        "count": len(faces),
    }

# ----- Public: identity-safe search -----
@app.post("/api/v1/identity_safe_search", response_model=IdentitySafeSearchResp)
def identity_safe_search(req: IdentitySafeSearchReq) -> IdentitySafeSearchResp:
    """
    Verify probe image against a known identity and, if verified, search faces_v1
    restricted to that identity. No search is ever executed if verification fails.
    """
    # Log request received
    log_event(
        "identity_safe_search_request",
        tenant_id=req.tenant_id,
        identity_id=req.identity_id,
        top_k=req.top_k,
        min_score=req.min_score,
    )
    
    # 1) Embed probe with strict quality + single face
    try:
        probe_vec, fwq = embed_one_b64_strict(
            req.image_b64,
            require_single_face=True,
            quality_cfg=VERIFY_QUALITY,
        )
    except HTTPException as e:
        # Treat quality/face errors as "verification failed" with detailed reason
        detail = e.detail if isinstance(e.detail, dict) else {"error": str(e.detail)}
        
        # Log quality failure
        log_event(
            "identity_safe_search_quality_fail",
            tenant_id=req.tenant_id,
            identity_id=req.identity_id,
            reason=detail,
        )
        
        return IdentitySafeSearchResp(
            verified=False,
            similarity=0.0,
            threshold=VERIFY_THRESHOLD_DEFAULT,
            reason=detail,
            results=[],
        )

    # 2) Fetch identity centroid from identities_v1
    centroid = _fetch_identity_centroid(
        tenant_id=req.tenant_id,
        identity_id=req.identity_id,
    )

    if centroid is None:
        # Log identity not found
        log_event(
            "identity_safe_search_identity_not_found",
            tenant_id=req.tenant_id,
            identity_id=req.identity_id,
        )
        
        return IdentitySafeSearchResp(
            verified=False,
            similarity=0.0,
            threshold=VERIFY_THRESHOLD_DEFAULT,
            reason={"error": "identity_not_found"},
            results=[],
        )

    # 3) Cosine similarity between probe and centroid
    probe_vec = np.asarray(probe_vec, dtype=np.float32)
    centroid_vec = np.asarray(centroid, dtype=np.float32)

    # Normalize defensively
    probe_vec = probe_vec / (np.linalg.norm(probe_vec) + 1e-8)
    centroid_vec = centroid_vec / (np.linalg.norm(centroid_vec) + 1e-8)

    similarity = float(np.dot(probe_vec, centroid_vec))

    # Log verification decision (before returning)
    verified = similarity >= VERIFY_THRESHOLD_DEFAULT
    logger.info(
        "identity_safe_search_decision",
        extra={
            "tenant_id": req.tenant_id,
            "identity_id": req.identity_id,
            "similarity": similarity,
            "threshold": VERIFY_THRESHOLD_DEFAULT,
            "verified": verified,
        },
    )

    if similarity < VERIFY_THRESHOLD_DEFAULT:
        # DO NOT RUN SEARCH IF NOT VERIFIED
        # Log verification failure
        log_event(
            "identity_safe_search_verification_fail",
            tenant_id=req.tenant_id,
            identity_id=req.identity_id,
            similarity=similarity,
            threshold=VERIFY_THRESHOLD_DEFAULT,
        )
        
        return IdentitySafeSearchResp(
            verified=False,
            similarity=similarity,
            threshold=VERIFY_THRESHOLD_DEFAULT,
            reason={"error": "low_similarity"},
            results=[],
        )

    # 4) If verified, run identity-restricted search on faces_v1
    matches = _search_faces_for_identity(
        query_vector=probe_vec.tolist(),
        tenant_id=req.tenant_id,
        identity_id=req.identity_id,
        top_k=req.top_k,
        min_score=req.min_score,
    )

    # 5) Convert matches to IdentitySafeSearchResult models
    results: List[IdentitySafeSearchResult] = []

    for m in matches:
        # Adapt this based on the actual type of 'm'.
        # Reuse the same mapping as the /search endpoint uses.
        payload = getattr(m, "payload", None)
        match_id = getattr(m, "id", None)
        score = getattr(m, "score", None)

        image_id = None
        tenant_val = None
        identity_val = None
        if isinstance(payload, dict):
            image_id = payload.get("image_sha256")  # Use image_sha256 to match existing pattern
            tenant_val = payload.get("tenant_id")
            identity_val = payload.get("identity_id")

        results.append(
            IdentitySafeSearchResult(
                id=str(match_id),
                score=float(score) if score is not None else 0.0,
                image_id=image_id,
                tenant_id=tenant_val,
                identity_id=identity_val,
                payload=payload,
            )
        )

    # Log success
    log_event(
        "identity_safe_search_success",
        tenant_id=req.tenant_id,
        identity_id=req.identity_id,
        similarity=similarity,
        threshold=VERIFY_THRESHOLD_DEFAULT,
        num_results=len(results),
    )

    return IdentitySafeSearchResp(
        verified=True,
        similarity=similarity,
        threshold=VERIFY_THRESHOLD_DEFAULT,
        reason=None,
        results=results,
    )
