from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uuid
import io
import asyncio
import time

from ..services.face import get_face_service
from ..services.storage import save_raw_and_thumb_async, get_object_from_storage, get_presigned_url
from ..services.vector import get_vector_client
from ..services.cache import get_cache_service
from ..services.batch import get_batch_processor
from ..services.webhook import get_webhook_service, WebhookEvent
from ..core.audit import get_audit_logger
from ..core.config import get_settings
from ..services.cleanup import run_cleanup_jobs
from ..core.errors import (
    ValidationError,
    AuthenticationError,
    AuthorizationError,
    RateLimitError,
    ResourceNotFoundError,
    StorageError,
    VectorDBError,
    FaceProcessingError,
    BatchProcessingError,
    CacheError,
    handle_generic_error,
    create_http_exception
)
router = APIRouter()

# Batch processing models
class BatchIndexRequest(BaseModel):
    """Request model for batch indexing."""
    image_urls: List[str]
    metadata: Optional[dict] = None

class BatchIndexResponse(BaseModel):
    """Response model for batch indexing."""
    batch_id: str
    total_images: int
    status: str
    message: str

class BatchStatusResponse(BaseModel):
    """Response model for batch status."""
    batch_id: str
    status: str
    total_images: int
    processed_images: int
    successful_images: int
    failed_images: int
    errors: List[dict]
    created_at: float
    updated_at: float

# Webhook models
class WebhookRegistrationRequest(BaseModel):
    """Request model for webhook registration."""
    url: str
    events: List[str]
    secret: Optional[str] = None
    timeout: int = 30
    retry_count: int = 3

class WebhookRegistrationResponse(BaseModel):
    """Response model for webhook registration."""
    webhook_id: str
    url: str
    events: List[str]
    message: str

class WebhookTestRequest(BaseModel):
    """Request model for webhook testing."""
    url: str

def _require_image(file: UploadFile, content: bytes, request_id: str = None):
    if not content:
        raise create_http_exception("EMPTY_FILE", request_id=request_id)

    ctype = (file.content_type or "").lower()
    if ctype not in {"image/jpeg", "image/jpg", "image/png"}:
        raise create_http_exception("INVALID_IMAGE_FORMAT", request_id=request_id)

    # Check file size (10MB limit)
    if len(content) > 10 * 1024 * 1024:
        raise create_http_exception("IMAGE_TOO_LARGE", request_id=request_id)

def _validate_top_k(top_k: int, request_id: str = None) -> int:
    """Validate and clamp top_k parameter to enforce ≤ 50 limit."""
    if top_k is None:
        return top_k

    if top_k < 1:
        raise create_http_exception("INVALID_TOP_K", request_id=request_id)

    # Clamp to maximum of 50
    if top_k > 50:
        return 50

    return top_k

@router.post(
    "/index_face",
    tags=["Face Operations"],
    summary="Index Face",
    description="""
    Upload an image, extract face embeddings, and store them in the vector database.

    This endpoint:
    - Detects faces in the uploaded image
    - Generates face embeddings for each detected face
    - Stores the embeddings in the vector database
    - Saves the original image and thumbnail to storage
    - Returns metadata about the indexed faces

    **Note**: This endpoint does not perform similarity search, only indexing.
    """,
    responses={
        200: {
            "description": "Face successfully indexed",
            "content": {
                "application/json": {
                        "example": {
                            "indexed": 2,
                            "phash": "a1b2c3d4e5f6g7h8",
                            "thumb_url": "https://minio.example.com/thumbnails/tenant123/abc123_thumb.jpg?X-Amz-Algorithm=...",
                            "vector_backend": "qdrant"
                        }
                }
            }
        },
        400: {"description": "Invalid image format or size"},
        413: {"description": "Image too large"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def index_face(request: Request, file: UploadFile = File(...)):
    """Upload image, extract embeddings, and upsert to vector DB (no search)."""
    content = await file.read()
    request_id = getattr(request.state, "request_id", None)
    _require_image(file, content, request_id)
    
    vec = get_vector_client()

    # Get tenant_id from request state (set by middleware)
    tenant_id = request.state.tenant_id

    face = get_face_service()
    # Run face processing operations in parallel for better performance
    phash, faces = await asyncio.gather(
        face.compute_phash_async(content),
        face.detect_and_embed_async(content)
    )

    raw_key, raw_url, thumb_key, thumb_url = await save_raw_and_thumb_async(content, tenant_id)

    vec = get_vector_client()
    items = []
    for f in faces:
        items.append({
            "id": str(uuid.uuid4()),
            "embedding": f["embedding"],
            "metadata": {
                "raw_key": raw_key,
                "thumb_key": thumb_key,
                "raw_url": raw_url,
                "thumb_url": thumb_url,
                "bbox": f["bbox"],
                "det_score": f["det_score"],
                "phash": phash,
            },
        })
    if items:
        vec.upsert_embeddings(items, tenant_id)

    # Log search operation for audit
    audit_logger = get_audit_logger()
    await audit_logger.log_search_operation(
        tenant_id=tenant_id,
        operation_type="index",
        face_count=len(faces),
        result_count=len(items),
        vector_backend="pinecone" if vec.using_pinecone() else "qdrant",
        request_id=getattr(request.state, "request_id", None)
    )

    # Send webhook notification
    webhook_service = get_webhook_service()
    webhook_event = WebhookEvent(
        event_type="face.indexed",
        tenant_id=tenant_id,
        data={
            "indexed_count": len(items),
            "face_count": len(faces),
            "phash": phash,
            "thumb_url": thumb_url,
            "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
            "request_id": getattr(request.state, "request_id", None)
        },
        request_id=getattr(request.state, "request_id", None)
    )
    # Send webhook asynchronously (don't wait for completion)
    asyncio.create_task(webhook_service.send_webhook(webhook_event))

    return {
        "indexed": len(items),
        "phash": phash,
        "thumb_url": thumb_url,
        "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
    }

@router.post(
    "/search_face",
    tags=["Face Operations"],
    summary="Search Similar Faces",
    description="""
    Upload an image, extract face embeddings, and find similar faces in the database.

    This endpoint:
    - Detects faces in the uploaded image
    - Generates face embeddings for each detected face
    - Searches for similar faces in the vector database
    - Stores the new embeddings for future searches
    - Returns similarity results with confidence scores

    **Note**: This endpoint both indexes the face and performs similarity search.
    """,
    responses={
        200: {
            "description": "Similar faces found",
            "content": {
                "application/json": {
                        "example": {
                            "faces_found": 1,
                            "phash": "a1b2c3d4e5f6g7h8",
                            "thumb_url": "https://minio.example.com/thumbnails/tenant123/abc123_thumb.jpg?X-Amz-Algorithm=...",
                            "results": [
                                {
                                    "id": "face-123",
                                    "score": 0.95,
                                    "metadata": {
                                        "thumb_url": "https://minio.example.com/thumbnails/tenant123/def456_thumb.jpg?X-Amz-Algorithm=...",
                                        "bbox": [100, 150, 200, 250],
                                        "site": "example.com",
                                        "url": "https://example.com/image.jpg",
                                        "ts": "2024-01-01T12:00:00Z",
                                        "p_hash": "b2c3d4e5f6g7h8i9",
                                        "quality": 0.92
                                    }
                                }
                            ],
                            "vector_backend": "qdrant"
                        }
                }
            }
        },
        400: {"description": "Invalid image format or size"},
        413: {"description": "Image too large"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def search_face(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(
        None,
        description="Number of similar faces to return (1-50, clamped to 50 if exceeded)",
        ge=1,
        le=50
    ),
    threshold: float = Query(None, description="Similarity threshold (0.0-1.0)", ge=0.0, le=1.0)
):
    """Upload image, embed, and query top matches (also upserts so future queries match)."""
    content = await file.read()
    request_id = getattr(request.state, "request_id", None)
    _require_image(file, content, request_id)
    
    vec = get_vector_client()

    # Get tenant_id from request state (set by middleware)
    tenant_id = request.state.tenant_id

    # Use default values from config if not provided
    settings = get_settings()

    # Validate and clamp top_k parameter
    top_k = _validate_top_k(top_k, request_id) or settings.default_top_k
    threshold = threshold or settings.default_similarity_threshold

    face = get_face_service()
    # Run face processing operations in parallel for better performance
    phash, faces = await asyncio.gather(
        face.compute_phash_async(content),
        face.detect_and_embed_async(content)
    )

    raw_key, raw_url, thumb_key, thumb_url = await save_raw_and_thumb_async(content, tenant_id)

    vec = get_vector_client()
    items = []
    for f in faces:
        items.append({
            "id": str(uuid.uuid4()),
            "embedding": f["embedding"],
            "metadata": {
                "raw_key": raw_key,
                "thumb_key": thumb_key,
                "raw_url": raw_url,
                "thumb_url": thumb_url,
                "bbox": f["bbox"],
                "det_score": f["det_score"],
                "phash": phash,
            },
        })
    if items:
        vec.upsert_embeddings(items, tenant_id)

    results = []
    if faces:
        # Get search results from vector database
        search_results = vec.search_similar_embeddings(
            embeddings=[f["embedding"] for f in faces],
            top_k=top_k,
            threshold=threshold,
            tenant_id=tenant_id
        )

        # Filter metadata to only include allowed fields: site, url, ts, bbox, p_hash, quality
        for result in search_results:
            filtered_metadata = {}
            original_metadata = result.get("metadata", {})

            # Only include allowed metadata fields
            allowed_fields = ["site", "url", "ts", "bbox", "p_hash", "quality"]
            for field in allowed_fields:
                if field in original_metadata:
                    filtered_metadata[field] = original_metadata[field]

            # Always include thumb_url as presigned URL (never raw URLs)
            filtered_metadata["thumb_url"] = get_presigned_url(
                settings.s3_bucket_thumbs,
                original_metadata.get("thumb_key", ""),
                "GET"
            )

            results.append({
                "id": result["id"],
                "score": result["score"],
                "metadata": filtered_metadata
            })

    # Log search operation for audit
    audit_logger = get_audit_logger()
    await audit_logger.log_search_operation(
        tenant_id=tenant_id,
        operation_type="search",
        face_count=len(faces),
        result_count=len(results),
        vector_backend="pinecone" if vec.using_pinecone() else "qdrant",
        request_id=getattr(request.state, "request_id", None)
    )

    # Send webhook notification
    webhook_service = get_webhook_service()
    webhook_event = WebhookEvent(
        event_type="face.searched",
        tenant_id=tenant_id,
        data={
            "faces_found": len(faces),
            "results_count": len(results),
            "phash": phash,
            "thumb_url": thumb_url,
            "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
            "request_id": getattr(request.state, "request_id", None)
        },
        request_id=getattr(request.state, "request_id", None)
    )
    # Send webhook asynchronously (don't wait for completion)
    asyncio.create_task(webhook_service.send_webhook(webhook_event))

    return {
        "faces_found": len(faces),
        "phash": phash,
        "thumb_url": thumb_url,
        "results": results,
        "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
    }


@router.post(
    "/compare_face",
    tags=["Face Operations"],
    summary="Compare Face (Search Only)",
    description="""
    Compare an uploaded image against existing faces in the database without storing the image.

    This endpoint:
    - Detects faces in the uploaded image
    - Generates face embeddings for each detected face
    - Searches for similar faces in the vector database
    - Returns similarity results with confidence scores
    - **Does NOT store the image or embeddings** (search-only operation)

    **Use case**: When you want to find similar faces without adding new data to the system.
    """,
    responses={
        200: {
            "description": "Face comparison completed",
            "content": {
                "application/json": {
                        "example": {
                            "phash": "a1b2c3d4e5f6g7h8",
                            "faces_found": 1,
                            "results": [
                                {
                                    "id": "face-123",
                                    "score": 0.95,
                                    "metadata": {
                                        "thumb_url": "https://minio.example.com/thumbnails/tenant123/def456_thumb.jpg?X-Amz-Algorithm=...",
                                        "bbox": [100, 150, 200, 250],
                                        "site": "example.com",
                                        "url": "https://example.com/image.jpg",
                                        "ts": "2024-01-01T12:00:00Z",
                                        "p_hash": "b2c3d4e5f6g7h8i9",
                                        "quality": 0.92
                                    }
                                }
                            ],
                            "vector_backend": "qdrant",
                            "message": "Found 1 face(s) and 1 similar matches"
                        }
                }
            }
        },
        400: {"description": "Invalid image format, size, or no faces detected"},
        413: {"description": "Image too large"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def compare_face(
    request: Request,
    file: UploadFile = File(...),
    top_k: int = Query(
        None,
        description="Number of similar faces to return (1-50, clamped to 50 if exceeded)",
        ge=1,
        le=50
    ),
    threshold: float = Query(None, description="Similarity threshold (0.0-1.0)", ge=0.0, le=1.0)
):
    """
    Search-only endpoint: compares uploaded image against existing faces in database
    without uploading the image to storage or database.
    """
    content = await file.read()
    request_id = getattr(request.state, "request_id", None)
    _require_image(file, content, request_id)
    
    vec = get_vector_client()

    # Get tenant_id from request state (set by middleware)
    tenant_id = request.state.tenant_id

    # Use default values from config if not provided
    settings = get_settings()

    # Validate and clamp top_k parameter
    top_k = _validate_top_k(top_k, request_id) or settings.default_top_k
    threshold = threshold or settings.default_similarity_threshold

    # Check cache first
    cache_service = get_cache_service()
    cached_phash = await cache_service.get_cached_perceptual_hash(content, tenant_id)
    cached_faces = await cache_service.get_cached_face_detection(content, tenant_id)

    if cached_phash and cached_faces:
        # Use cached results
        phash = cached_phash
        faces = cached_faces
    else:
        # Process face detection and cache results
        svc = get_face_service()
        phash, faces = await asyncio.gather(
            svc.compute_phash_async(content),
            svc.detect_and_embed_async(content)
        )

        # Cache the results
        await asyncio.gather(
            cache_service.cache_perceptual_hash(content, tenant_id, phash),
            cache_service.cache_face_detection(content, tenant_id, faces)
        )

    if not faces:
        # No faces detected
        return {
            "faces_found": 0,
            "phash": phash,
            "results": [],
            "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
            "message": "No faces detected in uploaded image"
        }

    # Get search results from vector database
    search_results = vec.search_similar_embeddings(
        embeddings=[f["embedding"] for f in faces],
        top_k=top_k,
        threshold=threshold,
        tenant_id=tenant_id
    )

    # Filter metadata to only include allowed fields: site, url, ts, bbox, p_hash, quality
    results = []
    for result in search_results:
        filtered_metadata = {}
        original_metadata = result.get("metadata", {})

        # Only include allowed metadata fields
        allowed_fields = ["site", "url", "ts", "bbox", "p_hash", "quality"]
        for field in allowed_fields:
            if field in original_metadata:
                filtered_metadata[field] = original_metadata[field]

        # Always include thumb_url as presigned URL (never raw URLs)
        filtered_metadata["thumb_url"] = get_presigned_url(
            settings.s3_bucket_thumbs,
            original_metadata.get("thumb_key", ""),
            "GET"
        )

        results.append({
            "id": result["id"],
            "score": result["score"],
            "metadata": filtered_metadata
        })

    # Log search operation for audit
    audit_logger = get_audit_logger()
    await audit_logger.log_search_operation(
        tenant_id=tenant_id,
        operation_type="compare",
        face_count=0,
        result_count=0,
        vector_backend="pinecone" if get_vector_client().using_pinecone() else "qdrant",
        request_id=getattr(request.state, "request_id", None)
    )

    return {
        "phash": phash,
        "faces_found": 0,
        "results": [],
        "vector_backend": "pinecone" if get_vector_client().using_pinecone() else "qdrant",
        "message": "No faces detected in the uploaded image"
    }

    # Query similar faces for the first detected face
    vec = get_vector_client()

    # Check cache for search results
    cached_results = await cache_service.get_cached_search_results(content, tenant_id, topk=top_k)

    if cached_results:
        results = cached_results
    else:
        results = vec.search_similar(faces[0]["embedding"], tenant_id, topk=top_k)
        # Cache the search results
        await cache_service.cache_search_results(content, tenant_id, topk=top_k, results=results)

    # Log search operation for audit
    audit_logger = get_audit_logger()
    await audit_logger.log_search_operation(
        tenant_id=tenant_id,
        operation_type="compare",
        face_count=len(faces),
        result_count=len(results),
        vector_backend="pinecone" if vec.using_pinecone() else "qdrant",
        request_id=getattr(request.state, "request_id", None)
    )

    # Send webhook notification
    webhook_service = get_webhook_service()
    webhook_event = WebhookEvent(
        event_type="face.compared",
        tenant_id=tenant_id,
        data={
            "faces_found": len(faces),
            "results_count": len(results),
            "phash": phash,
            "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
            "request_id": getattr(request.state, "request_id", None)
        },
        request_id=getattr(request.state, "request_id", None)
    )
    # Send webhook asynchronously (don't wait for completion)
    asyncio.create_task(webhook_service.send_webhook(webhook_event))

    return {
        "phash": phash,
        "faces_found": len(faces),
        "results": results,
        "vector_backend": "pinecone" if vec.using_pinecone() else "qdrant",
        "message": f"Found {len(faces)} face(s) and {len(results)} similar matches"
    }


@router.get("/images/{bucket}/{key:path}")
async def serve_image(bucket: str, key: str):
    """Proxy endpoint to serve images from storage."""
    try:
        image_data = get_object_from_storage(bucket, key)
        return StreamingResponse(io.BytesIO(image_data), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Image not found: {str(e)}")


@router.post("/admin/cleanup")
async def run_cleanup(request: Request):
    """Admin endpoint to run cleanup jobs manually."""
    tenant_id = request.state.tenant_id

    # Log cleanup operation
    audit_logger = get_audit_logger()
    await audit_logger.log_search_operation(
        tenant_id=tenant_id,
        operation_type="cleanup",
        face_count=0,
        result_count=0,
        vector_backend="none",
        request_id=getattr(request.state, "request_id", None)
    )

    try:
        results = await run_cleanup_jobs()
        return {
            "status": "success",
            "results": results,
            "message": "Cleanup jobs completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")


@router.post(
    "/batch/index",
    response_model=BatchIndexResponse,
    tags=["Batch Processing"],
    summary="Create Batch Indexing Job",
    description="""
    Create a new batch job to process multiple images for face indexing.

    This endpoint:
    - Accepts a list of image URLs to process
    - Creates a batch job with a unique ID
    - Starts processing images in the background
    - Returns job status and tracking information

    **Limitations**:
    - Maximum 100 images per batch
    - Images must be accessible via HTTP/HTTPS URLs
    - Processing happens asynchronously in the background
    """,
    responses={
        200: {
            "description": "Batch job created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "batch_id": "batch-123e4567-e89b-12d3-a456-426614174000",
                        "total_images": 10,
                        "status": "created",
                        "message": "Batch job created with 10 images"
                    }
                }
            }
        },
        400: {"description": "Invalid request (no URLs, batch too large)"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def create_batch_index_job(
    request: Request,
    batch_request: BatchIndexRequest,
    background_tasks: BackgroundTasks
):
    """Create a new batch indexing job."""
    tenant_id = request.state.tenant_id

    # Validate request
    if not batch_request.image_urls:
        raise create_http_exception("NO_IMAGE_URLS")

    if len(batch_request.image_urls) > 100:  # Limit batch size
        raise create_http_exception("INVALID_BATCH_SIZE")

    # Create batch job
    batch_processor = get_batch_processor()
    batch_id = await batch_processor.create_batch_job(
        tenant_id=tenant_id,
        image_urls=batch_request.image_urls,
        metadata=batch_request.metadata
    )

    # Start processing in background
    background_tasks.add_task(batch_processor.process_batch_job, batch_id)

    return BatchIndexResponse(
        batch_id=batch_id,
        total_images=len(batch_request.image_urls),
        status="created",
        message=f"Batch job created with {len(batch_request.image_urls)} images"
    )


@router.get("/batch/{batch_id}/status", response_model=BatchStatusResponse)
async def get_batch_status(request: Request, batch_id: str):
    """Get the status of a batch job."""
    tenant_id = request.state.tenant_id
    batch_processor = get_batch_processor()

    batch_info = batch_processor.get_batch_status(batch_id)
    if not batch_info:
        raise create_http_exception("BATCH_NOT_FOUND")

    # Ensure tenant can only access their own batches
    if batch_info["tenant_id"] != tenant_id:
        raise create_http_exception("BATCH_ACCESS_DENIED")

    return BatchStatusResponse(**batch_info)


@router.get("/batch/list")
async def list_batch_jobs(request: Request, limit: int = 20, offset: int = 0):
    """List batch jobs for the current tenant."""
    tenant_id = request.state.tenant_id
    batch_processor = get_batch_processor()

    all_batches = batch_processor.list_batches(tenant_id=tenant_id)

    # Apply pagination
    total_count = len(all_batches)
    paginated_batches = all_batches[offset:offset + limit]

    return {
        "batches": paginated_batches,
        "total_count": total_count,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total_count
    }


@router.delete("/batch/{batch_id}")
async def cancel_batch_job(request: Request, batch_id: str):
    """Cancel a batch job (only if it's still in progress)."""
    tenant_id = request.state.tenant_id
    batch_processor = get_batch_processor()

    batch_info = batch_processor.get_batch_status(batch_id)
    if not batch_info:
        raise create_http_exception("BATCH_NOT_FOUND")

    # Ensure tenant can only access their own batches
    if batch_info["tenant_id"] != tenant_id:
        raise create_http_exception("BATCH_ACCESS_DENIED")

    if batch_info["status"] not in ["created", "processing"]:
        raise create_http_exception("BATCH_CANCELLATION_FAILED")

    # Mark as cancelled
    batch_info["status"] = "cancelled"
    batch_info["updated_at"] = time.time()

    return {
        "message": f"Batch job {batch_id} cancelled",
        "batch_id": batch_id,
        "status": "cancelled"
    }


@router.post("/batch/cleanup")
async def cleanup_old_batches(request: Request, max_age_hours: int = 24):
    """Clean up old completed batch jobs."""
    tenant_id = request.state.tenant_id
    batch_processor = get_batch_processor()

    # Only allow cleanup of own tenant's batches
    cleaned_count = 0
    current_time = time.time()
    max_age_seconds = max_age_hours * 3600

    for batch_id, batch_info in list(batch_processor.active_batches.items()):
        if (batch_info["tenant_id"] == tenant_id and
            batch_info["status"] in ["completed", "failed", "cancelled"] and
            current_time - batch_info["updated_at"] > max_age_seconds):
            del batch_processor.active_batches[batch_id]
            cleaned_count += 1

    return {
        "message": f"Cleaned up {cleaned_count} old batch jobs",
        "cleaned_count": cleaned_count,
        "max_age_hours": max_age_hours
    }


@router.post(
    "/webhooks/register",
    response_model=WebhookRegistrationResponse,
    tags=["Webhooks"],
    summary="Register Webhook",
    description="""
    Register a new webhook endpoint to receive real-time notifications.

    **Supported Events**:
    - `face.indexed`: When a face is successfully indexed
    - `face.searched`: When a face search is completed
    - `face.compared`: When a face comparison is completed
    - `batch.created`: When a batch job is created
    - `batch.completed`: When a batch job is completed
    - `batch.failed`: When a batch job fails

    **Security**: Use the `secret` parameter to enable HMAC signature verification.
    """,
    responses={
        200: {
            "description": "Webhook registered successfully",
            "content": {
                "application/json": {
                    "example": {
                        "webhook_id": "webhook_1",
                        "url": "https://example.com/webhook",
                        "events": ["face.indexed", "face.searched"],
                        "message": "Webhook registered successfully"
                    }
                }
            }
        },
        400: {"description": "Invalid webhook configuration"},
        500: {"description": "Internal server error"}
    }
)
async def register_webhook(request: Request, webhook_request: WebhookRegistrationRequest):
    """Register a new webhook endpoint."""
    tenant_id = request.state.tenant_id
    webhook_service = get_webhook_service()

    # Validate events
    valid_events = [
        "face.indexed",
        "face.searched",
        "face.compared",
        "batch.created",
        "batch.completed",
        "batch.failed"
    ]
    invalid_events = [event for event in webhook_request.events if event not in valid_events]
    if invalid_events:
        raise create_http_exception("INVALID_WEBHOOK_EVENTS", {"invalid_events": invalid_events})

    webhook_id = await webhook_service.register_webhook(
        tenant_id=tenant_id,
        url=webhook_request.url,
        events=webhook_request.events,
        secret=webhook_request.secret,
        timeout=webhook_request.timeout,
        retry_count=webhook_request.retry_count
    )

    return WebhookRegistrationResponse(
        webhook_id=webhook_id,
        url=webhook_request.url,
        events=webhook_request.events,
        message="Webhook registered successfully"
    )


@router.get(
    "/webhooks/list",
    tags=["Webhooks"],
    summary="List Webhooks",
    description="List all registered webhook endpoints for the current tenant.",
    responses={
        200: {
            "description": "List of webhook endpoints",
            "content": {
                "application/json": {
                    "example": {
                        "webhooks": [
                            {
                                "url": "https://example.com/webhook",
                                "events": ["face.indexed", "face.searched"],
                                "has_secret": True,
                                "timeout": 30,
                                "retry_count": 3,
                                "created_at": 1640995200.0,
                                "last_used": 1640995300.0,
                                "success_count": 15,
                                "failure_count": 2
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def list_webhooks(request: Request):
    """List all webhook endpoints for the current tenant."""
    tenant_id = request.state.tenant_id
    webhook_service = get_webhook_service()

    webhooks = await webhook_service.list_webhooks(tenant_id)
    return {"webhooks": webhooks}


@router.delete(
    "/webhooks/unregister",
    tags=["Webhooks"],
    summary="Unregister Webhook",
    description="Unregister a webhook endpoint by URL.",
    responses={
        200: {
            "description": "Webhook unregistered successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Webhook unregistered successfully",
                        "url": "https://example.com/webhook"
                    }
                }
            }
        },
        404: {"description": "Webhook not found"}
    }
)
async def unregister_webhook(request: Request, url: str):
    """Unregister a webhook endpoint."""
    tenant_id = request.state.tenant_id
    webhook_service = get_webhook_service()

    success = await webhook_service.unregister_webhook(tenant_id, url)
    if not success:
        raise create_http_exception("WEBHOOK_NOT_FOUND")

    return {
        "message": "Webhook unregistered successfully",
        "url": url
    }


@router.post(
    "/webhooks/test",
    tags=["Webhooks"],
    summary="Test Webhook",
    description="Send a test webhook to verify endpoint configuration.",
    responses={
        200: {
            "description": "Webhook test completed",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Test webhook sent successfully",
                        "details": {
                            "url": "https://example.com/webhook",
                            "success": True,
                            "status_code": 200,
                            "attempt": 1
                        }
                    }
                }
            }
        }
    }
)
async def test_webhook(request: Request, test_request: WebhookTestRequest):
    """Test a webhook endpoint."""
    tenant_id = request.state.tenant_id
    webhook_service = get_webhook_service()

    result = await webhook_service.test_webhook(tenant_id, test_request.url)
    return result


@router.get(
    "/webhooks/stats",
    tags=["Webhooks"],
    summary="Webhook Statistics",
    description="Get webhook delivery statistics for the current tenant.",
    responses={
        200: {
            "description": "Webhook statistics",
            "content": {
                "application/json": {
                    "example": {
                        "total_endpoints": 2,
                        "total_success": 45,
                        "total_failures": 3,
                        "success_rate": 0.9375,
                        "endpoints": [
                            {
                                "url": "https://example.com/webhook",
                                "events": ["face.indexed"],
                                "success_count": 25,
                                "failure_count": 1,
                                "last_used": 1640995300.0
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def get_webhook_stats(request: Request):
    """Get webhook statistics for the current tenant."""
    tenant_id = request.state.tenant_id
    webhook_service = get_webhook_service()

    stats = await webhook_service.get_webhook_stats(tenant_id)
    return stats
