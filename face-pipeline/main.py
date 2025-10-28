import asyncio
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, status


# Configure logging
import uvicorn


#!/usr/bin/env python3
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from config.settings import settings
from services.search_api import router as search_router

"""
Face Pipeline Main Entry Point

FastAPI application for the face processing pipeline.
"""

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track startup time for uptime calculation
start_time = time.time()

# ============================================================================
# Lifespan Context Manager
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("🚀 Face Pipeline starting up...")
    logger.info(f"📝 Configuration loaded:")
    logger.info(f"   - MinIO: {settings.MINIO_ENDPOINT}")
    logger.info(f"   - Qdrant: {settings.QDRANT_URL}")
    logger.info(f"   - Collection: {settings.QDRANT_COLLECTION}")
    logger.info(f"   - Max Concurrent: {settings.max_concurrent}")
    logger.info(f"   - Face Min Size: {settings.min_face_size}")
    logger.info(f"   - Blur Min Variance: {settings.blur_min_variance}")

    # TODO: Initialize pipeline components
    # - Connect to vector database
    # - Verify storage connection
    # - Load ML models
    # - Set up async task pools

    yield

    # Shutdown
    logger.info("🛑 Face Pipeline shutting down...")

    # TODO: Cleanup resources
    # - Close database connections
    # - Stop background tasks
    # - Save state if needed


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Face Processing Pipeline API v0.1",
    description=(
        "**STABLE API v0.1** - Face detection, embedding, quality assessment, "
        "and similarity search pipeline.\n\n"
        "## API Contract Status: FROZEN ✅\n\n"
        "This API v0.1 contract is **stable and frozen** for safe integration.\n"
        "No breaking changes will be made to v0.1 endpoints.\n\n"
        "## Features\n"
        "- Face similarity search by image or vector\n"
        "- Face metadata retrieval with presigned URLs\n"
        "- Pipeline statistics and health monitoring\n"
        "- Multi-tenant isolation\n\n"
        "## Security\n"
        "- Presigned URLs with 10-minute TTL\n"
        "- Filtered metadata responses\n"
        "- Tenant-scoped access control"
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {
            "name": "search",
            "description": "Face similarity search operations"
        },
        {
            "name": "faces",
            "description": "Face metadata retrieval"
        },
        {
            "name": "statistics",
            "description": "Pipeline statistics and metrics"
        },
        {
            "name": "health",
            "description": "Health monitoring and status"
        }
    ]
)

# ============================================================================
# CORS Middleware
# ============================================================================

# Parse CORS origins from settings
if settings.cors_origins != "*":
    cors_origins = settings.cors_origins.split(",")
else:
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Include Routers
# ============================================================================

app.include_router(search_router)

# ============================================================================
# Root Endpoints
# ============================================================================

@app.get("/", status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint with API information.

    Returns:
        API information and available endpoints
    """
    return {
        "service": "Face Processing Pipeline",
        "version": "0.1.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "ready": "/ready",
            "info": "/info",
            "docs": "/docs",
            "api": "/api/v1",
            "search": "/api/v1/search",
            "stats": "/api/v1/stats"
        }
    }


@app.get("/health", status_code=status.HTTP_200_OK)
async def health():
    """
    Health check endpoint.

    Basic liveness check - returns OK if the application is running.
    Does not verify dependencies (MinIO, Qdrant, models).
    Use /ready for readiness checks.

    Returns:
        Health status
    """
    return {
        "status": "ok"
    }


@app.get("/ready")
def ready():
    """
    Readiness endpoint for Kubernetes/orchestration health checks.

    Checks if the application is ready to serve requests by verifying:
    - ML models are loaded (detector + embedder)
    - MinIO connectivity (storage)
    - Qdrant connectivity (vector database)
    - Redis connectivity (stats and dedup)

    Returns 503 (Service Unavailable) if not ready, 200 OK when ready.

    Returns:
        Dict with:
        - ready: bool - True if ready, False otherwise
        - reason: str - Explanation if not ready
        - checks: dict - Individual check statuses
        - meta: dict - Optional metadata (version, uptime)
    """
    checks = {}
    all_ready = True
    reasons = []
    
    # Check ML models
    try:
        from pipeline.detector import load_detector
        from pipeline.embedder import load_model
        
        load_detector()
        load_model()
        checks["models"] = True
    except Exception as e:
        checks["models"] = False
        all_ready = False
        reasons.append(f"models_not_ready: {e.__class__.__name__}")
    
    # Check MinIO storage
    try:
        from pipeline.storage import get_minio_client
        client = get_minio_client()
        client.list_buckets()
        checks["storage"] = True
    except Exception as e:
        checks["storage"] = False
        all_ready = False
        reasons.append(f"storage_not_ready: {e.__class__.__name__}")
    
    # Check Qdrant vector database
    try:
        from pipeline.indexer import get_qdrant_client
        client = get_qdrant_client()
        client.get_collections()
        checks["vector_db"] = True
    except Exception as e:
        checks["vector_db"] = False
        all_ready = False
        reasons.append(f"vector_db_not_ready: {e.__class__.__name__}")
    
    # Check Redis
    try:
        from pipeline.stats import get_redis_client
        r = get_redis_client()
        r.ping()
        checks["redis"] = True
    except Exception as e:
        checks["redis"] = False
        all_ready = False
        reasons.append(f"redis_not_ready: {e.__class__.__name__}")
    
    reason = "ok" if all_ready else "; ".join(reasons)
    
    # Optional metadata enrichment
    meta = {}
    try:
        # Read version from VERSION file
        import os
        version_file = os.path.join(os.path.dirname(__file__), "VERSION")
        if os.path.exists(version_file):
            with open(version_file, 'r') as f:
                version_line = f.readline().strip()
                if version_line:
                    meta["version"] = version_line
        
        # Calculate uptime (approximate)
        import time
        meta["uptime_sec"] = int(time.time() - start_time) if 'start_time' in globals() else 0
    except Exception:
        # Don't fail the ready check for metadata errors
        pass
    
    result = {
        "ready": all_ready,
        "reason": reason,
        "checks": checks
    }
    
    # Add meta if we have any data
    if meta:
        result["meta"] = meta
    
    return result


@app.get("/info", status_code=status.HTTP_200_OK)
async def info():
    """
    Detailed application information.

    Returns:
        Configuration and system information
    """
    return {
        "service": "Face Processing Pipeline",
        "version": "0.1.0",
        "configuration": {
            "minio_endpoint": settings.MINIO_ENDPOINT,
            "qdrant_url": settings.QDRANT_URL,
            "qdrant_collection": settings.QDRANT_COLLECTION,
            "max_concurrent": settings.max_concurrent,
            "min_face_size": settings.min_face_size,
            "blur_min_variance": settings.blur_min_variance,
            "presign_ttl_sec": settings.PRESIGN_TTL_SEC,
        },
        "features": {
            "face_detection": "planned",
            "face_embedding": "planned",
            "quality_assessment": "planned",
            "similarity_search": "planned",
            "storage": "planned",
            "vector_indexing": "planned"
        }
    }


# ============================================================================
# Error Handlers
# ============================================================================

@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": (
                f"The requested path '{request.url.path}' was not found"
            ),
            "hint": "Visit /docs for API documentation"
        }
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for the face pipeline."""
    # This is called when running: python main.py
    # For production, use: uvicorn main:app
    logger.info("Starting Face Pipeline in development mode...")

    uvicorn.run(
        "main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
        log_level=settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
