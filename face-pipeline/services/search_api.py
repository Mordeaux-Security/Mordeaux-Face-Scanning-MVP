import logging
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, Query, Path, HTTPException, status, File, UploadFile, Response, Depends
from pydantic import BaseModel, Field

import numpy as np
import io
from PIL import Image

from pipeline.detector import detect_faces
from pipeline.embedder import embed
from pipeline.indexer import get_qdrant_client
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

    Returns pipeline processing statistics.
    """
    processed: int = Field(
        default=0,
        description="Total faces processed and indexed"
    )
    rejected: int = Field(
        default=0,
        description="Total faces rejected (quality checks failed)"
    )
    dup_skipped: int = Field(
        default=0,
        description="Total faces skipped as duplicates"
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
            image = Image.open(io.BytesIO(request.image))
            faces = detect_faces(image)
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
            # Use first detected face
            face_crop = faces[0]
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

        # Query Qdrant
        client = get_qdrant_client()
        search_results = client.search(
            collection_name=settings.qdrant_collection,
            query_vector=embedding.tolist(),
            limit=request.top_k,
            score_threshold=request.threshold,
            query_filter={"must": [{"key": "tenant_id", "match": {"value": request.tenant_id}}]}
        )

        # Process results with presigned URLs and filtered metadata
        hits = []
        for result in search_results:
            original_payload = result.payload or {}

            # Filter metadata to only include allowed fields
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

            hits.append(SearchHit(
                face_id=str(result.id),
                score=float(result.score),
                payload=filtered_payload,
                thumb_url=thumb_url
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
        client = get_qdrant_client()
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
async def get_pipeline_stats() -> StatsResponse:
    """
    Get pipeline processing statistics.

    **TODO - DEV2 Implementation Steps**:
    1. Query Qdrant for total face count:
       - Use qdrant_client.count(collection_name)
       - This gives us 'processed' count
    2. Calculate rejected count:
       - Option A: Track in separate database/cache (Redis counter)
       - Option B: Query application logs/metrics
       - Option C: Return 0 for now, implement tracking in processor.py
    3. Calculate dup_skipped count:
       - Same as rejected - track during pipeline processing
       - Access from metrics or dedicated counter
    4. Return StatsResponse with all counts

    **Implementation Notes**:
    - For now, returns placeholder values (all zeros)
    - In DEV2, add counters to pipeline.processor.process_image()
    - Consider using Prometheus metrics or Redis for real-time stats

    Returns:
        StatsResponse with processed=0, rejected=0, dup_skipped=0 (placeholders)
    """
    logger.info("[STUB] Get pipeline stats")

    # TODO: Implement statistics collection (see docstring above)
    # from pipeline.indexer import get_qdrant_client
    # client = get_qdrant_client()
    # count_result = client.count(collection_name=settings.qdrant_collection)
    # processed = count_result.count
    #
    # # TODO: Get rejected and dup_skipped from metrics/database
    # rejected = 0  # Track in processor.py
    # dup_skipped = 0  # Track in processor.py
    #
    # return StatsResponse(processed=processed, rejected=rejected, dup_skipped=dup_skipped)

    # Placeholder response
    return StatsResponse(
        processed=0,  # TODO: Query Qdrant for actual count
        rejected=0,  # TODO: Track during pipeline processing
        dup_skipped=0,  # TODO: Track during deduplication
    )


# ============================================================================
# Health Check Endpoint
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
