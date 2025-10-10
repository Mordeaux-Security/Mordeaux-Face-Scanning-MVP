#!/usr/bin/env python3
"""
Face Pipeline Main Entry Point

FastAPI application for the face processing pipeline.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config.settings import settings
from services.search_api import router as search_router

# Configure logging
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
    title="Face Processing Pipeline",
    description="Modular face detection, embedding, quality assessment, and similarity search pipeline",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# ============================================================================
# CORS Middleware
# ============================================================================

# Parse CORS origins from settings
cors_origins = settings.cors_origins.split(",") if settings.cors_origins != "*" else ["*"]

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
    
    Returns:
        Health status
    """
    return {
        "status": "ok"
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
            "message": f"The requested path '{request.url.path}' was not found",
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
    import uvicorn
    
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
