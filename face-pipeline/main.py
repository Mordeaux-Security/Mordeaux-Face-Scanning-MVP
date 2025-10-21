import asyncio
import logging
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
    logger.info(f"   - MinIO: {settings.minio_endpoint}")
    logger.info(f"   - Qdrant: {settings.qdrant_url}")
    logger.info(f"   - Collection: {settings.qdrant_collection}")
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


@app.get("/ready", status_code=status.HTTP_503_SERVICE_UNAVAILABLE)
async def ready():
    """
    Readiness endpoint for Kubernetes/orchestration health checks.

    Checks if the application is ready to serve requests by verifying:
    - ML models are loaded
    - MinIO storage is accessible
    - Qdrant vector database is connected

    Returns 503 (Service Unavailable) if not ready, 200 OK when ready.

    **TODO - DEV2 Implementation Steps**:
    1. Check if ML models are loaded:
       - Try to access pipeline.detector model
       - Try to access pipeline.embedder model
       - Return ready=False if models not initialized

    2. Check MinIO connectivity:
       - Try to list buckets or check bucket exists
       - Use pipeline.storage.exists() with a test key
       - Return ready=False if MinIO unreachable

    3. Check Qdrant connectivity:
       - Try to get collection info
       - Use pipeline.indexer client health check
       - Return ready=False if Qdrant unreachable

    4. Return ready=True only if all checks pass

    Returns:
        Dict with:
        - ready: bool - True if ready, False otherwise
        - reason: str - Explanation if not ready
        - checks: dict (optional) - Individual check statuses

    Example response (not ready):
        {
            "ready": false,
            "reason": "models_not_loaded",
            "checks": {
                "models": false,
                "storage": false,
                "vector_db": false
            }
        }

    Example response (ready):
        {
            "ready": true,
            "reason": "all_systems_operational",
            "checks": {
                "models": true,
                "storage": true,
                "vector_db": true
            }
        }
    """
    # TODO: Implement actual readiness checks
    # For now, always return not ready until models are loaded

    # Placeholder response
    return {
        "ready": False,
        "reason": "models_not_loaded",
        "checks": {
            "models": False,      # TODO: Check if InsightFace models loaded
            "storage": False,     # TODO: Check MinIO connectivity
            "vector_db": False    # TODO: Check Qdrant connectivity
        }
    }


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
            "minio_endpoint": settings.minio_endpoint,
            "qdrant_url": settings.qdrant_url,
            "qdrant_collection": settings.qdrant_collection,
            "max_concurrent": settings.max_concurrent,
            "min_face_size": settings.min_face_size,
            "blur_min_variance": settings.blur_min_variance,
            "presign_ttl_sec": settings.presign_ttl_sec,
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

async def main():
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
    asyncio.run(main())
