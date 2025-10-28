import asyncio
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from .api.routes import router
from .core.middleware import (
    tenant_middleware,
    performance_middleware,
    request_size_middleware
)
from .core.config import get_settings, validate_configuration
from .core.metrics import get_metrics
from .core.logging_config import setup_logging, get_logger
from .tasks.cleanup_tasks import start_cleanup_scheduler, stop_cleanup_scheduler
from .core.errors import handle_generic_error
from app.services.health import get_health_service

setup_logging()
logger = get_logger(__name__)

# Initialize settings and validate configuration
settings = get_settings()
validate_configuration()

# Create FastAPI app
app = FastAPI(
    title="Mordeaux Face Scanning API",
    description="Face detection and similarity search API",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(tenant_middleware)
app.add_middleware(performance_middleware)
app.add_middleware(request_size_middleware)

# Include routers
app.include_router(router, prefix="/api/v1")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_service = get_health_service()
        health_status = await health_service.check_health()
        return health_status
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Mordeaux Face Scanning API",
        "version": "0.1.0",
        "status": "running"
    }

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event"""
    logger.info("Starting Mordeaux Face Scanning API...")
    
    # Start cleanup scheduler
    start_cleanup_scheduler()
    
    logger.info("API startup complete")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event"""
    logger.info("Shutting down Mordeaux Face Scanning API...")
    
    # Stop cleanup scheduler
    stop_cleanup_scheduler()
    
    logger.info("API shutdown complete")

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    return handle_generic_error(request, exc)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)