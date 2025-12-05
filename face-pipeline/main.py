import os
import logging
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse
import hashlib
import uuid

from fastapi import FastAPI, HTTPException, status, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import base64

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
from config.settings import settings

setup_logging()

logger = logging.getLogger(__name__)

app = FastAPI(title="face-pipeline")

# Add CORS middleware to allow frontend access
cors_origins = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS", "*") != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    threshold: float = Field(default=0.10, ge=0.0, le=1.0)  # Lowered to 0.10 - typical similar faces score 0.16-0.22
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
        search_params=SearchParams(exact=True),  # Exact search for maximum accuracy
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
    processing_details = None  # Will be populated for image searches
    
    if req.vector is not None:
        query_vec = np.asarray(req.vector, dtype=np.float32)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-9)
        if len(query_vec) != VECTOR_DIM:
            raise HTTPException(422, f"vector length must be {VECTOR_DIM}")
    
    elif req.image_b64 is not None:
        # Allow multiple faces but pick best usable
        # First, analyze all faces to get complete processing details
        from pipeline.face_helpers import analyze_faces_with_quality
        
        processing_details = {
            "num_faces_detected": 0,
            "num_usable_faces": 0,
            "selected_face_index": None,
            "all_faces": [],
            "quality_config": {
                "min_size": SEARCH_QUALITY.min_size,
                "min_blur_var": SEARCH_QUALITY.min_blur_var,
                "max_yaw_deg": SEARCH_QUALITY.max_yaw_deg,
                "max_pitch_deg": SEARCH_QUALITY.max_pitch_deg,
                "min_score": SEARCH_QUALITY.min_score,
            },
            "detection_threshold": settings.DET_SCORE_THRESH,
        }
        
        try:
            # Get all faces with quality evaluation
            try:
                img, all_faces_with_quality = analyze_faces_with_quality(
                    req.image_b64,
                    quality_cfg=SEARCH_QUALITY,
                )
            except ValueError as ve:
                # Image decode failed
                processing_details["error"] = f"Image decode failed: {str(ve)}"
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail={"error": "invalid_image", "message": str(ve), "processing_details": processing_details}
                )
            
            # Add image info
            if img is not None:
                processing_details["image"] = {
                    "height": int(img.shape[0]),
                    "width": int(img.shape[1]),
                    "channels": int(img.shape[2]) if len(img.shape) > 2 else 1,
                }
            
            processing_details["num_faces_detected"] = len(all_faces_with_quality)
            
            # Collect detailed info for each face
            for idx, fwq in enumerate(all_faces_with_quality):
                face = fwq.face
                quality = fwq.quality
                
                # Get detection score
                det_score = float(getattr(face, "det_score", 0.0))
                
                # Get bounding box
                bbox = face.bbox.tolist() if hasattr(face, "bbox") and face.bbox is not None else []
                
                # Check if embedding is available
                has_embedding = hasattr(face, 'embedding') and face.embedding is not None
                has_normed_embedding = hasattr(face, 'normed_embedding') and face.normed_embedding is not None
                
                face_info = {
                    "index": idx,
                    "detection_score": det_score,
                    "bbox": bbox,
                    "quality": {
                        "is_usable": quality.is_usable,
                        "score": float(quality.score),
                        "reasons": quality.reasons,
                        "blur_var": float(quality.blur_var),
                        "yaw_deg": float(quality.yaw_deg),
                        "pitch_deg": float(quality.pitch_deg),
                        "roll_deg": float(quality.roll_deg),
                    },
                    "embedding_available": has_embedding,
                    "normed_embedding_available": has_normed_embedding,
                }
                
                processing_details["all_faces"].append(face_info)
            
            # Count usable faces
            usable_faces = [fwq for fwq in all_faces_with_quality if fwq.quality.is_usable]
            processing_details["num_usable_faces"] = len(usable_faces)
            
            # Now get the embedding using the strict function
            query_vec, selected_fwq = embed_one_b64_strict(
                req.image_b64,
                require_single_face=False,
                quality_cfg=SEARCH_QUALITY,
            )
            
            # Find which face was selected
            selected_face = selected_fwq.face
            for idx, fwq in enumerate(all_faces_with_quality):
                if fwq.face is selected_face:
                    processing_details["selected_face_index"] = idx
                    break
            
            # Add embedding info
            emb_norm = np.linalg.norm(query_vec)
            processing_details["embedding"] = {
                "norm": float(emb_norm),
                "dimension": len(query_vec),
                "source": "precomputed" if hasattr(selected_face, 'embedding') and selected_face.embedding is not None else "computed",
            }
            
            logger.info(f"Search: Generated embedding, quality_usable={selected_fwq.quality.is_usable}, quality_score={selected_fwq.quality.score:.4f}, norm={emb_norm:.6f}")
        except HTTPException as e:
            # Include processing details in error response if available
            error_detail = e.detail
            if isinstance(error_detail, dict):
                error_detail["processing_details"] = processing_details
            else:
                error_detail = {"error": str(error_detail), "processing_details": processing_details}
            logger.error(f"Search: Failed to generate embedding: {error_detail}")
            raise HTTPException(status_code=e.status_code, detail=error_detail)
        except Exception as e:
            logger.error(f"Search: Unexpected error generating embedding: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail={"error": "embedding_generation_failed", "message": str(e), "processing_details": processing_details})
    
    else:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"error": "missing_vector_or_image"},
        )

    # Use exact search for maximum accuracy (trades speed for precision)
    # Exact search ensures accurate cosine similarity scores, especially important
    # for same-person matches where we expect high similarity (0.75-0.95+)
    hits = qc.search(
        collection_name=FACES_COLLECTION,
        query_vector=query_vec.tolist(),
        limit=req.top_k,
        score_threshold=req.threshold,
        query_filter=_tenant_filter(req.tenant_id),
        with_payload=True,
        with_vectors=False,
        search_params=SearchParams(exact=True),  # Exact cosine similarity for maximum accuracy
    )
    
    # Generate URLs for thumbnails/crops/original images
    from pipeline.storage import presign
    
    def extract_site_from_payload(payload: dict) -> str:
        """Extract site/domain information from payload metadata."""
        # Try site_id first
        site_id = payload.get("site_id")
        if site_id:
            return str(site_id)
        
        # Try site field (legacy)
        site = payload.get("site")
        if site:
            return str(site)
        
        # Extract domain from source_url or page_url
        for url_field in ["source_url", "page_url", "url"]:
            url = payload.get(url_field, "")
            if url and isinstance(url, str):
                try:
                    parsed = urlparse(url)
                    domain = parsed.netloc or parsed.path.split('/')[0]
                    if domain and '.' in domain:
                        # Remove www. prefix if present
                        domain = domain.replace('www.', '')
                        return domain
                except Exception:
                    continue
        
        return "unknown"
    
    out = []
    for h in hits:
        payload = h.payload or {}
        
        # Extract site information for display
        site = extract_site_from_payload(payload)
        
        result = {
            "id": str(h.id),
            "face_id": str(h.id),  # For compatibility
            "score": float(h.score),
            "similarity_pct": round(float(h.score) * 100, 1),
            "payload": {
                **payload,
                "site": site,  # Add normalized site field for frontend
            },
        }
        
        # Try to get thumbnail URL (use correct bucket: thumbnails)
        thumb_key = payload.get("thumb_key")
        if thumb_key:
            try:
                result["thumb_url"] = presign(settings.MINIO_BUCKET_THUMBS, thumb_key)
            except Exception as e:
                logger.debug(f"Failed to presign thumbnail {thumb_key}: {e}")
                result["thumb_url"] = None
        else:
            result["thumb_url"] = None
        
        # Try to get crop URL (use correct bucket: face-crops)
        crop_key = payload.get("crop_key")
        if crop_key:
            try:
                result["crop_url"] = presign(settings.MINIO_BUCKET_CROPS, crop_key)
            except Exception as e:
                logger.debug(f"Failed to presign crop {crop_key}: {e}")
                result["crop_url"] = None
        else:
            result["crop_url"] = None
        
        # Try to get raw image URL from raw_key or url
        raw_key = payload.get("raw_key")
        if raw_key:
            try:
                result["image_url"] = presign(settings.MINIO_BUCKET_RAW, raw_key)
            except Exception as e:
                logger.debug(f"Failed to presign raw image {raw_key}: {e}")
        
        # Fallback: try original URL if image_url not set
        if not result.get("image_url"):
            original_url = payload.get("url", "") or payload.get("source_url", "")
            if original_url and isinstance(original_url, str):
                if original_url.startswith("s3://"):
                    # Parse s3://bucket/key format
                    parts = original_url[5:].split("/", 1)
                    if len(parts) == 2:
                        bucket, key = parts
                        try:
                            result["image_url"] = presign(bucket, key)
                        except Exception as e:
                            logger.debug(f"Failed to presign s3 URL {original_url}: {e}")
                elif original_url.startswith(("http://", "https://")):
                    # External URL, use directly
                    result["image_url"] = original_url
        
        # Set image_url to thumb_url or crop_url if no raw image available
        if not result.get("image_url"):
            result["image_url"] = result.get("thumb_url") or result.get("crop_url")
        
        out.append(result)
    
    response_data = {
        "query": {"tenant_id": req.tenant_id, "search_mode": "image" if req.image_b64 else "vector",
                  "mode": req.mode, "top_k": req.top_k, "threshold": req.threshold},
        "hits": out, 
        "count": len(out)
    }
    
    # Add processing details for image-based searches
    if processing_details is not None:
        response_data["processing_details"] = processing_details
    
    return response_data

# ----- Public: search by file upload -----
@app.post("/api/v1/search/file")
async def search_by_file(
    file: UploadFile = File(...),
    tenant_id: str = Query(..., description="Tenant ID for multi-tenant filtering"),
    top_k: int = Query(default=10, ge=1, le=100, description="Maximum number of results to return"),
    threshold: float = Query(default=0.10, ge=0.0, le=1.0, description="Minimum similarity score threshold")
):
    """
    Search for similar faces by uploading an image file.
    
    This endpoint accepts multipart/form-data file uploads and converts them
    to base64 for processing by the main search endpoint.
    """
    try:
        # Validate file type
        if file.content_type and not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=422, 
                detail={"error": "invalid_file_type", "message": f"Expected image file, got {file.content_type}"}
            )
        
        # Read file bytes
        file_bytes = await file.read()
        logger.info(f"File search: Received file '{file.filename}', size={len(file_bytes)} bytes, content_type={file.content_type}")
        
        if len(file_bytes) == 0:
            raise HTTPException(status_code=422, detail={"error": "empty_file", "message": "Uploaded file is empty"})
        
        # Basic image format validation (check magic bytes)
        valid_image_signatures = [
            b'\xFF\xD8\xFF',  # JPEG
            b'\x89PNG\r\n\x1a\n',  # PNG
            b'GIF87a',  # GIF
            b'GIF89a',  # GIF
        ]
        is_valid_image = any(file_bytes.startswith(sig) for sig in valid_image_signatures)
        if not is_valid_image:
            logger.warning(f"File search: File doesn't appear to be a valid image (magic bytes: {file_bytes[:10]})")
            # Don't fail here - let face detection handle it
        
        # Convert to base64
        image_b64 = base64.b64encode(file_bytes).decode('utf-8')
        logger.info(f"File search: Converted to base64, length={len(image_b64)} chars")
        
        # Create SearchReq and call main search function
        req = SearchReq(
            tenant_id=tenant_id,
            image_b64=image_b64,
            top_k=top_k,
            threshold=threshold
        )
        
        return search(req)
        
    except HTTPException:
        # Re-raise HTTPException to preserve status code and detail
        raise
    except Exception as e:
        logger.error(f"Error in file search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": "file_search_failed", "message": str(e)})

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

    # Require at least 2 good enrollment images (lowered from 3 for better UX)
    # 3-5 images recommended for best results, but 2 is acceptable
    if len(embeddings) < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "error": "not_enough_high_quality_images",
                "num_valid": len(embeddings),
                "min_required": 2,
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
        search_params=SearchParams(exact=True),  # Exact search for maximum accuracy
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
