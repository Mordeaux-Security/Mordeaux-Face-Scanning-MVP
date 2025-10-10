import time
import uuid
from typing import Optional
from fastapi import Request, HTTPException, status
from fastapi.responses import Response
import logging
from .config import get_settings
from .metrics import record_request_metrics

logger = logging.getLogger(__name__)

# Tenant scoping middleware
async def tenant_middleware(request: Request, call_next):
    """Middleware to validate X-Tenant-ID header and add to request state."""
    # Exempt health check endpoints and docs from tenant validation
    exempt_paths = [
        "/healthz",
        "/healthz/detailed",
        "/healthz/database",
        "/healthz/redis",
        "/healthz/storage",
        "/healthz/vector",
        "/healthz/face",
        "/docs",
        "/redoc",
        "/openapi.json"
    ]
    
    # Check if the current path is exempt
    if request.url.path in exempt_paths:
        response = await call_next(request)
        return response
    
    tenant_id = request.headers.get("X-Tenant-ID")
    
    if not tenant_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Tenant-ID header is required"
        )
    
    # Validate tenant ID format (basic validation)
    if not tenant_id.strip() or len(tenant_id.strip()) < 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="X-Tenant-ID must be at least 3 characters long"
        )
    
    # Add tenant_id to request state for use in endpoints
    request.state.tenant_id = tenant_id.strip()
    
    response = await call_next(request)
    return response

# Performance monitoring middleware
async def performance_middleware(request: Request, call_next):
    """Middleware to track request timing and performance metrics."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    # Add request ID to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    # Calculate processing time
    process_time = time.time() - start_time
    
    # Record performance metrics
    endpoint = f"{request.method} {request.url.path}"
    tenant_id = getattr(request.state, "tenant_id", "unknown")
    
    record_request_metrics(
        endpoint=endpoint,
        tenant_id=tenant_id,
        duration=process_time,
        status_code=response.status_code,
        request_id=request_id
    )
    
    # Log performance metrics
    logger.info(
        f"Request completed",
        extra={
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "process_time": process_time,
            "tenant_id": tenant_id
        }
    )
    
    return response

# Request size validation middleware
async def request_size_middleware(request: Request, call_next):
    """Middleware to validate request size limits."""
    settings = get_settings()
    content_length = request.headers.get("content-length")
    
    if content_length:
        size_mb = int(content_length) / (1024 * 1024)
        if size_mb > settings.max_image_size_mb:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Request size exceeds {settings.max_image_size_mb}MB limit"
            )
    
    response = await call_next(request)
    return response
