import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query, Path, HTTPException, status, File, UploadFile, Response, Depends, Request
from pydantic import BaseModel, Field

import numpy as np
import io
from PIL import Image

from pipeline.detector import detect_faces
from pipeline.embedder import embed
from pipeline.indexer import get_client
from pipeline.storage import presign
from config.settings import settings

"""
Search API Service

Face similarity search endpoints for the face processing pipeline.

This module provides REST API endpoints for:
- Face similarity search (by image bytes or vector)
- Face metadata retrieval by ID
- Pipeline statistics

All endpoints currently return stub responses with TODO markers for DEV2 implementation.
"""

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/api/v1", tags=["search"])


# ============================================================================
# API Version Header Dependency
# ============================================================================

def add_version_header(response: Response):
    """Add X-API-Version header to all responses."""
    response.headers["X-API-Version"] = "v0.1"
    return response


# ============================================================================
# Request/Response Models - DEV2 Ready Contracts
# ============================================================================

class SearchRequest(BaseModel):
    """
    Request model for face similarity search.

    Supports two search modes:
    1. By image bytes (upload image for face extraction + search)
    2. By vector (direct embedding vector search)

    At least one of `image` or `vector` must be provided.
    """
    image: Optional[bytes] = Field(
        None,
        description="Image bytes (multipart/form-data) for face extraction and search"
    )
    vector: Optional[List[float]] = Field(
        None,
        description="Pre-computed embedding vector (512-dim float array)"
    )
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    tenant_id: str = Field(
        ...,
        description="Tenant ID for multi-tenant filtering"
    )
    threshold: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0-1)"
    )


class SearchHit(BaseModel):
    """
    Single search result/hit.

    Contains face ID, similarity score, Qdrant payload, and presigned thumbnail URL.
    """
    face_id: str = Field(
        ...,
        description="Unique face identifier (Qdrant point ID)"
    )
    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Similarity score (cosine similarity, 0-1)"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Qdrant payload with tenant_id, site, url, bbox, quality, etc."
    )
    thumb_url: Optional[str] = Field(
        None,
        description="Presigned URL for face thumbnail (TTL: 10 minutes max, 256px longest side)"
    )
    crop_url: Optional[str] = Field(
        None,
        description="Presigned URL for face crop image (112x112 typical)"
    )
    image_url: Optional[str] = Field(
        None,
        description="Presigned URL for a displayable image (thumbnail or crop)"
    )


class SearchResponse(BaseModel):
    """
    Response model for face similarity search.

    Returns query metadata, list of hits, and total count.
    """
    query: Dict[str, Any] = Field(
        default_factory=dict,
        description="Query metadata (tenant_id, threshold, top_k, search_mode)"
    )
    hits: List[SearchHit] = Field(
        default_factory=list,
        description="Ranked list of matching faces"
    )
    count: int = Field(
        default=0,
        description="Number of hits returned"
    )


class FaceDetailResponse(BaseModel):
    """
    Response model for GET /faces/{face_id}.

    Returns detailed information about a specific face.
    """
    face_id: str = Field(
        ...,
        description="Unique face identifier"
    )
    payload: Dict[str, Any] = Field(
        default_factory=dict,
        description="Qdrant payload with all metadata"
    )
    thumb_url: Optional[str] = Field(
        None,
        description="Presigned URL for face thumbnail (TTL: 10 minutes max, 256px longest side)"
    )


class StatsResponse(BaseModel):
    """
    Response model for GET /stats.

    Returns pipeline processing statistics including counters and timing metrics.
    """
    processed: int = Field(
        default=0,
        description="Total faces processed and indexed"
    )


# ==========================================================================
# Helpers to derive/display image URLs for hits
# ==========================================================================

def _derive_keys_from_id(face_id: str, tenant_id: str) -> tuple[str | None, str | None]:
    """Derive crop/thumb keys from a face id like "{image_sha256}:face_{i}".
    Returns (crop_key, thumb_key) or (None, None) on failure.
    """
    try:
        image_sha256, face_tag = face_id.split(":", 1)
        if not face_tag.startswith("face_"):
            return None, None
        idx = int(face_tag.split("_")[1])
        base = f"{image_sha256}_face_{idx}"
        crop_key = f"{tenant_id}/{base}.jpg"
        thumb_key = f"{tenant_id}/{base}_thumb.jpg"
        return crop_key, thumb_key
    except Exception:
        return None, None


def _urls_for_hit(hit) -> dict:
    """Build presigned URLs for a Qdrant ScoredPoint hit.
    Prefers payload-provided keys; falls back to deriving from id and tenant_id.
    """
    payload = getattr(hit, "payload", {}) or {}
    tenant_id = payload.get("tenant_id")
    crop_key = payload.get("crop_key")
    thumb_key = payload.get("thumb_key")

    if (not crop_key or not thumb_key) and tenant_id:
        dk, tk = _derive_keys_from_id(str(getattr(hit, "id", "")), tenant_id)
        crop_key = crop_key or dk
        thumb_key = thumb_key or tk

    urls: dict = {}
    try:
        if crop_key:
            urls["crop_url"] = presign(getattr(settings, "MINIO_BUCKET_CROPS", "face-crops"), crop_key)
    except Exception:
        pass
    try:
        if thumb_key:
            urls["thumb_url"] = presign(getattr(settings, "MINIO_BUCKET_THUMBS", "thumbnails"), thumb_key)
    except Exception:
        pass
    return urls
    rejected: int = Field(
        default=0,
        description="Total faces rejected (quality checks failed)"
    )
    dup_skipped: int = Field(
        default=0,
        description="Total faces skipped as duplicates"
    )
    counts: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Detailed counter metrics (images_total, faces_detected, etc.)"
    )
    timings_ms: Optional[Dict[str, float]] = Field(
        default=None,
        description="Accumulated timing metrics in milliseconds"
    )


# ============================================================================
# API Endpoints - DEV2 Phase Stubs
# ============================================================================

@router.post(
    "/search",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK
)
async def search_faces(request: SearchRequest) -> SearchResponse:
    """
    Search for similar faces by image or embedding vector.

    **Search Modes**:
    1. **By Image**: Upload image bytes, extract face embedding, search Qdrant
    2. **By Vector**: Provide pre-computed 512-dim embedding, search directly

    **TODO - DEV2 Implementation Steps**:
    1. Validate request (must have `image` OR `vector`, not both)
    2. If `image` provided:
       - Decode image bytes to PIL/numpy
       - Detect faces using pipeline.detector.detect_faces()
       - Extract first/best face
       - Generate embedding using pipeline.embedder.embed()
    3. If `vector` provided:
       - Validate vector dimension (must be 512)
       - L2 normalize if needed
    4. Query Qdrant:
       - Use pipeline.indexer.search(
           vector, top_k, filters={'tenant_id': request.tenant_id}
         )
       - Apply threshold filtering (score >= request.threshold)
    5. For each hit:
       - Generate presigned thumbnail URL using pipeline.storage.presign()
       - Extract face_id from Qdrant point ID
       - Return payload from Qdrant
    6. Return SearchResponse with query metadata, hits, and count

    Args:
        request: SearchRequest with image/vector, tenant_id, top_k, threshold

    Returns:
        SearchResponse with query metadata, hits (empty list for now), and count=0

    Raises:
        HTTPException: 400 if validation fails (both or neither image/vector provided)
    """
    logger.info(
        f"Search request: tenant_id={request.tenant_id}, "
        f"top_k={request.top_k}, threshold={request.threshold}, "
        f"has_image={request.image is not None}, "
        f"has_vector={request.vector is not None}"
    )

    try:
        # Validate that exactly one of image or vector is provided
        if (request.image is None) == (request.vector is None):
            raise HTTPException(400, "Must provide either 'image' or 'vector', not both or neither")

        # Extract embedding
        if request.image:
            # Decode image bytes and detect faces
            import cv2
            
            # Convert PIL to numpy array for OpenCV
            image = Image.open(io.BytesIO(request.image))
            img_array = np.array(image)
            
            # Handle different image formats (RGB, RGBA, Grayscale)
            if len(img_array.shape) == 3:
                if img_array.shape[2] == 4:  # RGBA
                    # Convert RGBA to RGB
                    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                elif img_array.shape[2] == 3:  # RGB
                    img_rgb = img_array
                else:
                    raise HTTPException(400, f"Unsupported image format with {img_array.shape[2]} channels")
                
                # Convert RGB to BGR for OpenCV
                img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                # Grayscale image
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            
            faces = detect_faces(img_bgr)
            if not faces:
                return SearchResponse(
                    query={
                        "tenant_id": request.tenant_id,
                        "search_mode": "image",
                        "top_k": request.top_k,
                        "threshold": request.threshold
                    },
                    hits=[],
                    count=0
                )
            
            # Use first detected face - align and crop
            from pipeline.detector import align_and_crop
            face_data = faces[0]
            face_crop = align_and_crop(img_bgr, face_data["landmarks"])
            embedding = embed(face_crop)
        else:
            # Validate vector dimension (must be 512)
            if len(request.vector) != 512:
                raise HTTPException(400, f"Vector must be 512 dimensions, got {len(request.vector)}")
            embedding = np.array(request.vector, dtype=np.float32)
            # L2 normalize if needed
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

        # Query Qdrant using our indexer module
        from pipeline.indexer import search

        search_results = search(
            vector=embedding.tolist(),
            top_k=request.top_k,
            tenant_id=request.tenant_id,
            threshold=request.threshold,
        )
        
        # Filter by threshold
        # search_results are Qdrant ScoredPoint objects
        search_results = [r for r in search_results if float(getattr(r, "score", 0.0)) >= request.threshold]

        # Process results with presigned URLs and filtered metadata
        hits = []
        for result in search_results:
            original_payload = getattr(result, "payload", {}) or {}

            # Filter metadata to only include allowed fields
            filtered_payload = {}
            allowed_fields = ["site", "url", "ts", "bbox", "p_hash", "quality"]
            for field in allowed_fields:
                if field in original_payload:
                    filtered_payload[field] = original_payload[field]

            # Generate presigned URLs from payload or derived keys
            url_fields = _urls_for_hit(result)
            thumb_url = url_fields.get("thumb_url")
            crop_url = url_fields.get("crop_url")
            # Prefer thumbnail for display; fallback to crop
            image_url = thumb_url or crop_url

            # Extract id/score from ScoredPoint
            res_id = getattr(result, "id", None)
            res_score = getattr(result, "score", 0.0)

            hits.append(SearchHit(
                face_id=str(res_id),
                score=float(res_score),
                payload=filtered_payload,
                thumb_url=thumb_url,
                crop_url=crop_url,
                image_url=image_url
            ))

        return SearchResponse(
            query={
                "tenant_id": request.tenant_id,
                "search_mode": "image" if request.image else "vector",
                "top_k": request.top_k,
                "threshold": request.threshold
            },
            hits=hits,
            count=len(hits)
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in search: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")


@router.post(
    "/search/file",
    response_model=SearchResponse,
    status_code=status.HTTP_200_OK
)
async def search_faces_by_file(
    file: UploadFile = File(...),
    tenant_id: str = Query(..., description="Tenant ID for multi-tenant filtering"),
    top_k: int = Query(default=10, ge=1, le=100, description="Maximum number of results to return"),
    threshold: float = Query(default=0.75, ge=0.0, le=1.0, description="Minimum similarity score threshold")
) -> SearchResponse:
    """
    Search for similar faces by uploading an image file.
    
    This endpoint accepts multipart/form-data file uploads and processes them
    for face detection and similarity search.
    """
    logger.info(
        f"File search request: tenant_id={tenant_id}, "
        f"top_k={top_k}, threshold={threshold}, "
        f"filename={file.filename}, content_type={file.content_type}"
    )
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        # Create SearchRequest with file bytes
        request = SearchRequest(
            image=file_bytes,
            tenant_id=tenant_id,
            top_k=top_k,
            threshold=threshold
        )
        
        # Call the main search function
        return await search_faces(request)
        
    except Exception as e:
        logger.error(f"Error in file search: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")


@router.get(
    "/faces/{face_id}",
    response_model=FaceDetailResponse,
    status_code=status.HTTP_200_OK
)
async def get_face_by_id(
    face_id: str = Path(
        ..., description="Unique face identifier (Qdrant point ID)"
    )
) -> FaceDetailResponse:
    """
    Retrieve detailed information about a specific face by ID.

    Returns face metadata with presigned thumbnail URL (TTL: 10 minutes max).
    Only includes allowed metadata fields: site, url, ts, bbox, p_hash, quality.
    """
    logger.info(f"Get face by ID: {face_id}")

    try:
        client = get_client()
        points = client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[face_id]
        )
        if not points:
            raise HTTPException(404, f"Face not found: {face_id}")

        point = points[0]
        original_payload = point.payload or {}

        # Filter metadata to only include allowed fields: site, url, ts, bbox, p_hash, quality
        filtered_payload = {}
        allowed_fields = ["site", "url", "ts", "bbox", "p_hash", "quality"]
        for field in allowed_fields:
            if field in original_payload:
                filtered_payload[field] = original_payload[field]

        # Generate presigned thumbnail URL (TTL: 10 minutes max)
        thumb_url = None
        thumb_key = original_payload.get('thumb_key')
        if thumb_key:
            thumb_url = presign(
                bucket=settings.minio_bucket_thumbs,
                key=thumb_key,
                ttl_sec=600  # 10 minutes
            )

        return FaceDetailResponse(
            face_id=face_id,
            payload=filtered_payload,
            thumb_url=thumb_url
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving face {face_id}: {e}")
        raise HTTPException(500, f"Internal server error: {str(e)}")


@router.get(
    "/stats",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK
)
async def get_pipeline_stats(
    response: Response,
    tenant_id: Optional[str] = None
) -> StatsResponse:
    """
    Get pipeline processing statistics.

    Returns real-time statistics from Redis counters:
    - processed: Number of faces successfully indexed
    - rejected: Number of faces rejected by quality checks
    - dup_skipped: Number of faces skipped as duplicates

    Args:
        tenant_id: Optional tenant ID to get tenant-specific stats
        response: FastAPI response object for headers

    Returns:
        StatsResponse with current statistics
    """
    try:
        from pipeline.stats import get_stats, snapshot
        
        # Get both legacy stats and new metrics snapshot
        stats_data = get_stats(tenant_id)
        metrics_snapshot = snapshot()
        
        logger.info(f"Retrieved stats: {stats_data}")
        logger.debug(f"Metrics snapshot: {metrics_snapshot}")
        
        # Add version header
        response.headers["X-API-Version"] = "v0.1"
        
        return StatsResponse(
            processed=stats_data["processed"],
            rejected=stats_data["rejected"],
            dup_skipped=stats_data["dup_skipped"],
            counts=metrics_snapshot.get("counts"),
            timings_ms=metrics_snapshot.get("timings_ms")
        )
        
    except Exception as e:
        logger.error(f"Error retrieving statistics: {e}")
        # Return zeros on error to maintain API contract
        return StatsResponse(processed=0, rejected=0, dup_skipped=0, counts={}, timings_ms={})


# ============================================================================
# Health Check Endpoint
# ============================================================================
# Processing Endpoint (Temporary - for testing)
# ============================================================================

@router.post("/process", status_code=status.HTTP_200_OK)
async def process_image(request: Request) -> Dict[str, Any]:
    """
    Process and index an image for face search.
    
    This is a temporary endpoint to add faces to the database for testing.
    """
    try:
        # Get the uploaded file
        form = await request.form()
        file = form.get("file")
        
        if not file:
            raise HTTPException(400, "No file uploaded")
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Process the image
        from pipeline.processor import process_image
        import uuid
        from datetime import datetime
        
        # Create a message for the processor
        import hashlib
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
        
        # Upload image to MinIO first
        from pipeline.storage import put_bytes, ensure_buckets
        ensure_buckets()
        key = f"test/{image_sha256}.jpg"
        put_bytes("raw-images", key, image_bytes, "image/jpeg")
        
        message = {
            "tenant_id": "demo-tenant",
            "site": "test-site",
            "url": "http://test.com/image.jpg",
            "image_sha256": image_sha256,
            "bucket": "raw-images",
            "key": key,
            "image_phash": "test-phash-12345678",  # Placeholder - would be computed from image
            "face_hints": []
        }
        
        # Process the image
        result = process_image(message)
        
        # Check if faces were found and processed
        if result.get("faces_accepted", 0) > 0:
            return {
                "status": "success",
                "message": f"Processed {result['faces_accepted']} faces",
                "faces_accepted": result["faces_accepted"],
                "faces_rejected": result.get("faces_rejected", 0)
            }
        else:
            return {
                "status": "success",
                "message": "No faces detected in image",
                "faces_accepted": 0,
                "faces_rejected": result.get("faces_rejected", 0)
            }
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(500, f"Error processing image: {str(e)}")

# ============================================================================

@router.get("/health", status_code=status.HTTP_200_OK)
async def api_health() -> Dict[str, str]:
    """
    API health check endpoint.

    Returns basic service health status. Does not check dependencies.

    Returns:
        Dict with status, service name, and version
    """
    return {
        "status": "healthy",
        "service": "face-pipeline-search-api",
        "version": "0.1.0-dev2",
        "api_version": "v0.1",
        "note": "API v0.1 contract frozen - all endpoints stable"
    }
