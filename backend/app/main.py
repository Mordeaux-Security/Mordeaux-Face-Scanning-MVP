import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from .api.routes import router
from .core.middleware import tenant_middleware, performance_middleware, request_size_middleware
from .core.rate_limiter import rate_limit_middleware
from .core.audit import audit_middleware
from .core.config import get_settings, validate_configuration
from .core.metrics import get_metrics
from .core.logging_config import setup_logging, get_logger
from .tasks.cleanup_tasks import start_cleanup_scheduler, stop_cleanup_scheduler

# Setup structured logging
setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Mordeaux Face Scanning API",
    description="""
    ## Mordeaux Face Scanning API
    
    A comprehensive face detection and similarity search API with support for:
    
    * **Face Detection & Embedding**: Extract face embeddings from images
    * **Similarity Search**: Find similar faces in the database
    * **Batch Processing**: Process multiple images in bulk
    * **Multi-tenant Support**: Isolated data per tenant
    * **Caching**: Redis-based caching for improved performance
    * **Audit Logging**: Complete audit trail of all operations
    
    ### Authentication
    All endpoints require the `X-Tenant-ID` header for tenant identification.
    
    ### Rate Limiting
    - 60 requests per minute per tenant
    - 1000 requests per hour per tenant
    - 10MB maximum image size
    
    ### Supported Image Formats
    - JPEG (.jpg, .jpeg)
    - PNG (.png)
    
    ### Performance
    - P95 latency target: ≤ 5 seconds
    - Presigned URLs TTL: ≤ 10 minutes
    - Data retention: 90 days for crawled images, 24 hours for user queries
    """,
    version="1.0.0",
    contact={
        "name": "Mordeaux API Support",
        "email": "support@mordeaux.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Development server"
        },
        {
            "url": "https://api.mordeaux.com",
            "description": "Production server"
        }
    ],
    openapi_tags=[
        {
            "name": "Face Operations",
            "description": "Face detection, embedding, and similarity search operations"
        },
        {
            "name": "Batch Processing",
            "description": "Bulk face indexing and processing operations"
        },
        {
            "name": "Health & Monitoring",
            "description": "System health checks and monitoring endpoints"
        },
        {
            "name": "Cache Management",
            "description": "Cache statistics and management operations"
        },
        {
            "name": "Administration",
            "description": "Administrative operations and system management"
        },
        {
            "name": "Webhooks",
            "description": "Webhook management and real-time notifications"
        },
        {
            "name": "Metrics Dashboard",
            "description": "System metrics, analytics, and monitoring dashboards"
        }
    ]
)

# Validate configuration on startup
try:
    config_valid = validate_configuration()
    if not config_valid:
        logger.warning("Configuration validation failed - some features may not work correctly")
except Exception as e:
    logger.error(f"Configuration validation error: {e}")

# Add middleware in order (last added is first executed)
app.middleware("http")(audit_middleware)
app.middleware("http")(request_size_middleware)
app.middleware("http")(rate_limit_middleware)
app.middleware("http")(performance_middleware)
app.middleware("http")(tenant_middleware)

app.include_router(router, prefix="/api")

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Startup event handler."""
    logger.info("Starting Mordeaux Face Scanning API...")
    
    # Start cleanup scheduler
    try:
        asyncio.create_task(start_cleanup_scheduler())
        logger.info("Cleanup scheduler started")
    except Exception as e:
        logger.error(f"Failed to start cleanup scheduler: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler."""
    logger.info("Shutting down Mordeaux Face Scanning API...")
    
    # Stop cleanup scheduler
    try:
        await stop_cleanup_scheduler()
        logger.info("Cleanup scheduler stopped")
    except Exception as e:
        logger.error(f"Failed to stop cleanup scheduler: {e}")

# Global exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with consistent error format."""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.detail if isinstance(exc.detail, dict) else {"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    from .core.errors import handle_generic_error
    
    http_exc = handle_generic_error(exc)
    return JSONResponse(
        status_code=http_exc.status_code,
        content=http_exc.detail
    )

@app.get(
    "/healthz",
    tags=["Health & Monitoring"],
    summary="Basic Health Check",
    description="Simple health check endpoint that returns basic system status.",
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "example": {"ok": True, "status": "healthy"}
                }
            }
        }
    }
)
def healthz():
    """Basic health check endpoint."""
    return {"ok": True, "status": "healthy"}

@app.get(
    "/healthz/detailed",
    tags=["Health & Monitoring"],
    summary="Detailed Health Check",
    description="Comprehensive health check that tests all system components including database, Redis, storage, vector database, and face processing service.",
    responses={
        200: {
            "description": "All services are healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "timestamp": 1640995200.0,
                        "total_check_time_ms": 150.5,
                        "environment": "development",
                        "services": {
                            "database": {"status": "healthy", "response_time_ms": 25.3},
                            "redis": {"status": "healthy", "response_time_ms": 12.1},
                            "storage": {"status": "healthy", "response_time_ms": 45.2},
                            "vector_db": {"status": "healthy", "response_time_ms": 38.7},
                            "face_service": {"status": "healthy", "response_time_ms": 28.2}
                        },
                        "summary": {
                            "total_services": 5,
                            "healthy_services": 5,
                            "degraded_services": 0,
                            "unhealthy_services": 0
                        }
                    }
                }
            }
        },
        503: {"description": "One or more services are unhealthy"}
    }
)
async def detailed_healthz():
    """Detailed health check endpoint with comprehensive service status."""
    from app.services.health import get_health_service
    
    health_service = get_health_service()
    health_data = await health_service.get_comprehensive_health()
    
    # Return 503 if any service is unhealthy
    if health_data["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_data)
    
    return health_data

@app.get("/healthz/database")
async def database_health():
    """Database-specific health check."""
    from app.services.health import get_health_service
    
    health_service = get_health_service()
    health_data = await health_service.check_database_health()
    
    if health_data["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_data)
    
    return health_data

@app.get("/healthz/redis")
async def redis_health():
    """Redis-specific health check."""
    from app.services.health import get_health_service
    
    health_service = get_health_service()
    health_data = await health_service.check_redis_health()
    
    if health_data["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_data)
    
    return health_data

@app.get("/healthz/storage")
async def storage_health():
    """Storage-specific health check."""
    from app.services.health import get_health_service
    
    health_service = get_health_service()
    health_data = await health_service.check_storage_health()
    
    if health_data["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_data)
    
    return health_data

@app.get("/healthz/vector")
async def vector_health():
    """Vector database-specific health check."""
    from app.services.health import get_health_service
    
    health_service = get_health_service()
    health_data = await health_service.check_vector_db_health()
    
    if health_data["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_data)
    
    return health_data

@app.get("/healthz/face")
async def face_health():
    """Face service-specific health check."""
    from app.services.health import get_health_service
    
    health_service = get_health_service()
    health_data = await health_service.check_face_service_health()
    
    if health_data["status"] == "unhealthy":
        raise HTTPException(status_code=503, detail=health_data)
    
    return health_data

@app.get("/config")
def get_config():
    """Get current configuration (without sensitive data)."""
    settings = get_settings()
    return {
        "environment": settings.environment,
        "log_level": settings.log_level,
        "rate_limit_per_minute": settings.rate_limit_per_minute,
        "rate_limit_per_hour": settings.rate_limit_per_hour,
        "presigned_url_ttl": settings.presigned_url_ttl,
        "crawled_thumbs_retention_days": settings.crawled_thumbs_retention_days,
        "user_query_images_retention_hours": settings.user_query_images_retention_hours,
        "max_image_size_mb": settings.max_image_size_mb,
        "p95_latency_threshold_seconds": settings.p95_latency_threshold_seconds,
        "using_pinecone": settings.using_pinecone,
        "using_minio": settings.using_minio,
        "vector_index": settings.vector_index,
        "s3_bucket_raw": settings.s3_bucket_raw,
        "s3_bucket_thumbs": settings.s3_bucket_thumbs,
    }

@app.get("/metrics")
def get_metrics_endpoint():
    """Get performance metrics and monitoring data."""
    metrics = get_metrics()
    return metrics.get_metrics_summary()

@app.get("/metrics/p95")
def get_p95_metrics():
    """Get P95 latency metrics specifically."""
    metrics = get_metrics()
    return {
        "p95_latency": metrics.get_p95_latency(),
        "p99_latency": metrics.get_p99_latency(),
        "avg_latency": metrics.get_avg_latency(),
        "median_latency": metrics.get_median_latency(),
        "threshold_exceeded": metrics.is_p95_threshold_exceeded(),
        "threshold_seconds": get_settings().p95_latency_threshold_seconds
    }

@app.get("/performance/recommendations")
async def get_performance_recommendations():
    """Get performance recommendations based on current metrics."""
    from ..services.performance import get_performance_optimizer
    
    optimizer = get_performance_optimizer()
    recommendations = await optimizer.get_performance_recommendations()
    return recommendations

@app.get("/performance/thresholds")
async def get_performance_thresholds():
    """Monitor performance thresholds and alert if exceeded."""
    from ..services.performance import get_performance_optimizer
    
    optimizer = get_performance_optimizer()
    thresholds = await optimizer.monitor_performance_thresholds()
    return thresholds

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics and performance metrics."""
    from ..services.cache import get_cache_service
    
    cache_service = get_cache_service()
    stats = await cache_service.get_cache_stats()
    return stats

@app.delete("/cache/tenant/{tenant_id}")
async def clear_tenant_cache(tenant_id: str):
    """Clear all cached data for a specific tenant."""
    from ..services.cache import get_cache_service
    
    cache_service = get_cache_service()
    deleted_count = await cache_service.invalidate_tenant_cache(tenant_id)
    return {
        "message": f"Cleared {deleted_count} cache entries for tenant {tenant_id}",
        "deleted_count": deleted_count
    }

@app.delete("/cache/all")
async def clear_all_cache():
    """Clear all cached data (use with caution)."""
    from ..services.cache import get_cache_service
    
    cache_service = get_cache_service()
    success = await cache_service.clear_all_cache()
    return {
        "message": "All cache data cleared" if success else "Failed to clear cache",
        "success": success
    }

@app.get(
    "/dashboard/overview",
    tags=["Metrics Dashboard"],
    summary="System Overview Dashboard",
    description="Get high-level system overview with health, performance, and usage metrics.",
    responses={
        200: {
            "description": "System overview data",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "system_health": {
                            "overall_status": "healthy",
                            "healthy_services": 5,
                            "total_services": 5,
                            "unhealthy_services": 0
                        },
                        "performance": {
                            "p95_latency": 2.5,
                            "p99_latency": 4.2,
                            "avg_latency": 1.8,
                            "total_requests": 1250,
                            "error_rate": 0.02,
                            "threshold_exceeded": False
                        },
                        "cache": {
                            "total_cached_items": 450,
                            "hit_rate": 0.85,
                            "memory_usage": "12.5MB"
                        }
                    }
                }
            }
        }
    }
)
async def get_system_overview():
    """Get system overview dashboard data."""
    from ..services.dashboard import get_dashboard_service
    
    dashboard_service = get_dashboard_service()
    overview = await dashboard_service.get_system_overview()
    return overview

@app.get(
    "/dashboard/performance",
    tags=["Metrics Dashboard"],
    summary="Performance Metrics Dashboard",
    description="Get detailed performance metrics and trends.",
    responses={
        200: {
            "description": "Performance metrics data",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "time_range_hours": 24,
                        "overall_metrics": {
                            "total_requests": 1250,
                            "total_errors": 25,
                            "rate_limit_violations": 3,
                            "p95_latency": 2.5,
                            "p99_latency": 4.2,
                            "avg_latency": 1.8
                        },
                        "endpoint_metrics": {
                            "POST /api/search_face": {
                                "request_count": 450,
                                "p95_latency": 3.2,
                                "avg_latency": 2.1,
                                "error_count": 5,
                                "error_rate": 0.011
                            }
                        }
                    }
                }
            }
        }
    }
)
async def get_performance_metrics(time_range_hours: int = 24):
    """Get detailed performance metrics."""
    from ..services.dashboard import get_dashboard_service
    
    dashboard_service = get_dashboard_service()
    metrics = await dashboard_service.get_performance_metrics(time_range_hours)
    return metrics

@app.get(
    "/dashboard/analytics",
    tags=["Metrics Dashboard"],
    summary="Usage Analytics Dashboard",
    description="Get usage analytics and trends for the system.",
    responses={
        200: {
            "description": "Usage analytics data",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "time_range_hours": 24,
                        "database": {
                            "audit_logs": {
                                "total_requests": 1250,
                                "error_requests": 25,
                                "avg_process_time": 1.8,
                                "unique_tenants": 15
                            }
                        },
                        "cache": {
                            "total_items": 450,
                            "hit_rate": 0.85
                        }
                    }
                }
            }
        }
    }
)
async def get_usage_analytics(time_range_hours: int = 24):
    """Get usage analytics and trends."""
    from ..services.dashboard import get_dashboard_service
    
    dashboard_service = get_dashboard_service()
    analytics = await dashboard_service.get_usage_analytics(time_range_hours)
    return analytics

@app.get(
    "/dashboard/tenant/{tenant_id}",
    tags=["Metrics Dashboard"],
    summary="Tenant Metrics Dashboard",
    description="Get metrics specific to a particular tenant.",
    responses={
        200: {
            "description": "Tenant-specific metrics",
            "content": {
                "application/json": {
                    "example": {
                        "tenant_id": "tenant-123",
                        "timestamp": 1640995200.0,
                        "performance": {
                            "request_count": 150,
                            "p95_latency": 2.1,
                            "rate_limit_violations": 0
                        },
                        "webhooks": {
                            "total_endpoints": 2,
                            "total_success": 45,
                            "total_failures": 2,
                            "success_rate": 0.957
                        }
                    }
                }
            }
        }
    }
)
async def get_tenant_metrics(tenant_id: str):
    """Get metrics for a specific tenant."""
    from ..services.dashboard import get_dashboard_service
    
    dashboard_service = get_dashboard_service()
    metrics = await dashboard_service.get_tenant_metrics(tenant_id)
    return metrics

@app.get(
    "/dashboard/health",
    tags=["Metrics Dashboard"],
    summary="Health Dashboard",
    description="Get comprehensive health dashboard data for all system components.",
    responses={
        200: {
            "description": "Health dashboard data",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "overall_health": {
                            "status": "healthy",
                            "total_check_time_ms": 150.5,
                            "summary": {
                                "total_services": 5,
                                "healthy_services": 5,
                                "degraded_services": 0,
                                "unhealthy_services": 0
                            }
                        },
                        "services": {
                            "database": {
                                "status": "healthy",
                                "response_time_ms": 25.3,
                                "details": {
                                    "database_size_bytes": 104857600,
                                    "active_connections": 5
                                }
                            }
                        }
                    }
                }
            }
        }
    }
)
async def get_health_dashboard():
    """Get comprehensive health dashboard data."""
    from ..services.dashboard import get_dashboard_service
    
    dashboard_service = get_dashboard_service()
    health_data = await dashboard_service.get_health_dashboard()
    return health_data

@app.get(
    "/admin/db/performance",
    tags=["Administration"],
    summary="Database Performance Analysis",
    description="Analyze database query performance and identify optimization opportunities.",
    responses={
        200: {
            "description": "Database performance analysis",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "slow_queries": [
                            {
                                "query": "SELECT * FROM audit_logs WHERE tenant_id = $1 AND created_at > $2",
                                "calls": 150,
                                "mean_time": 250.5,
                                "hit_percent": 95.2
                            }
                        ],
                        "index_usage": [
                            {
                                "table": "audit_logs",
                                "index": "idx_audit_logs_tenant_created",
                                "scans": 1250,
                                "size": "2.5 MB"
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def analyze_database_performance():
    """Analyze database query performance."""
    from ..services.db_optimization import get_db_optimization_service
    
    db_service = get_db_optimization_service()
    analysis = await db_service.analyze_query_performance()
    return analysis

@app.get(
    "/admin/db/health",
    tags=["Administration"],
    summary="Database Health Metrics",
    description="Get comprehensive database health metrics including size, connections, and locks.",
    responses={
        200: {
            "description": "Database health metrics",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "database_size_human": "125.5 MB",
                        "table_sizes": [
                            {
                                "table": "audit_logs",
                                "size": "45.2 MB",
                                "size_bytes": 47448064
                            }
                        ],
                        "connections": {
                            "total": 15,
                            "active": 3,
                            "idle": 12
                        }
                    }
                }
            }
        }
    }
)
async def get_database_health():
    """Get database health metrics."""
    from ..services.db_optimization import get_db_optimization_service
    
    db_service = get_db_optimization_service()
    health = await db_service.get_database_health_metrics()
    return health

@app.post(
    "/admin/db/optimize",
    tags=["Administration"],
    summary="Optimize Database Tables",
    description="Run database optimization tasks including ANALYZE and VACUUM operations.",
    responses={
        200: {
            "description": "Database optimization completed",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "table_stats": {
                            "live_tuples": 15000,
                            "dead_tuples": 500,
                            "last_analyze": "2024-01-01T12:00:00"
                        },
                        "index_stats": [
                            {
                                "index_name": "idx_audit_logs_tenant_created",
                                "scans": 1250,
                                "avg_tuples_per_scan": 12.5
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def optimize_database():
    """Optimize database tables."""
    from ..services.db_optimization import get_db_optimization_service
    
    db_service = get_db_optimization_service()
    optimization = await db_service.optimize_audit_logs_table()
    return optimization

@app.post(
    "/admin/db/cleanup",
    tags=["Administration"],
    summary="Cleanup Old Audit Logs",
    description="Clean up old audit logs based on retention policy.",
    responses={
        200: {
            "description": "Audit logs cleanup completed",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "retention_days": 30,
                        "audit_logs_deleted": 5000,
                        "search_audit_logs_deleted": 2500,
                        "total_deleted": 7500,
                        "vacuum_completed": True
                    }
                }
            }
        }
    }
)
async def cleanup_audit_logs(retention_days: int = 30):
    """Clean up old audit logs."""
    from ..services.db_optimization import get_db_optimization_service
    
    db_service = get_db_optimization_service()
    cleanup = await db_service.cleanup_old_audit_logs(retention_days)
    return cleanup

@app.post(
    "/admin/tenants",
    tags=["Administration"],
    summary="Create New Tenant",
    description="Create a new tenant with specified configuration.",
    responses={
        201: {
            "description": "Tenant created successfully",
            "content": {
                "application/json": {
                    "example": {
                        "tenant_id": "tenant_abc123def456",
                        "name": "Acme Corporation",
                        "description": "Main tenant for Acme Corporation",
                        "status": "active",
                        "created_at": 1640995200.0,
                        "metadata": {
                            "plan": "enterprise",
                            "region": "us-east-1"
                        }
                    }
                }
            }
        },
        400: {"description": "Invalid tenant data"},
        500: {"description": "Internal server error"}
    }
)
async def create_tenant(
    name: str,
    description: str = "",
    metadata: dict = None
):
    """Create a new tenant."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    tenant = await tenant_service.create_tenant(name, description, metadata)
    return tenant.to_dict()

@app.get(
    "/admin/tenants",
    tags=["Administration"],
    summary="List Tenants",
    description="List all tenants with optional filtering by status.",
    responses={
        200: {
            "description": "List of tenants",
            "content": {
                "application/json": {
                    "example": {
                        "tenants": [
                            {
                                "tenant_id": "tenant_abc123def456",
                                "name": "Acme Corporation",
                                "description": "Main tenant for Acme Corporation",
                                "status": "active",
                                "created_at": 1640995200.0,
                                "metadata": {}
                            }
                        ],
                        "total": 1
                    }
                }
            }
        }
    }
)
async def list_tenants(
    status: str = None,
    limit: int = 100,
    offset: int = 0
):
    """List tenants with optional filtering."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    tenants = await tenant_service.list_tenants(status, limit, offset)
    
    return {
        "tenants": [tenant.to_dict() for tenant in tenants],
        "total": len(tenants)
    }

@app.get(
    "/admin/tenants/{tenant_id}",
    tags=["Administration"],
    summary="Get Tenant Details",
    description="Get detailed information about a specific tenant.",
    responses={
        200: {
            "description": "Tenant details",
            "content": {
                "application/json": {
                    "example": {
                        "tenant_id": "tenant_abc123def456",
                        "name": "Acme Corporation",
                        "description": "Main tenant for Acme Corporation",
                        "status": "active",
                        "created_at": 1640995200.0,
                        "updated_at": 1640995200.0,
                        "metadata": {}
                    }
                }
            }
        },
        404: {"description": "Tenant not found"}
    }
)
async def get_tenant(tenant_id: str):
    """Get tenant details."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    tenant = await tenant_service.get_tenant(tenant_id)
    
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return tenant.to_dict()

@app.put(
    "/admin/tenants/{tenant_id}",
    tags=["Administration"],
    summary="Update Tenant",
    description="Update tenant information.",
    responses={
        200: {
            "description": "Tenant updated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "tenant_id": "tenant_abc123def456",
                        "name": "Acme Corporation Updated",
                        "description": "Updated description",
                        "status": "active",
                        "updated_at": 1640995300.0
                    }
                }
            }
        },
        404: {"description": "Tenant not found"}
    }
)
async def update_tenant(
    tenant_id: str,
    name: str = None,
    description: str = None,
    metadata: dict = None
):
    """Update tenant information."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    tenant = await tenant_service.update_tenant(tenant_id, name, description, metadata)
    
    if not tenant:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return tenant.to_dict()

@app.post(
    "/admin/tenants/{tenant_id}/suspend",
    tags=["Administration"],
    summary="Suspend Tenant",
    description="Suspend a tenant to prevent further operations.",
    responses={
        200: {
            "description": "Tenant suspended successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tenant suspended successfully",
                        "tenant_id": "tenant_abc123def456"
                    }
                }
            }
        },
        404: {"description": "Tenant not found"}
    }
)
async def suspend_tenant(tenant_id: str):
    """Suspend a tenant."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    success = await tenant_service.suspend_tenant(tenant_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return {
        "message": "Tenant suspended successfully",
        "tenant_id": tenant_id
    }

@app.post(
    "/admin/tenants/{tenant_id}/activate",
    tags=["Administration"],
    summary="Activate Tenant",
    description="Activate a suspended tenant.",
    responses={
        200: {
            "description": "Tenant activated successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tenant activated successfully",
                        "tenant_id": "tenant_abc123def456"
                    }
                }
            }
        },
        404: {"description": "Tenant not found"}
    }
)
async def activate_tenant(tenant_id: str):
    """Activate a tenant."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    success = await tenant_service.activate_tenant(tenant_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return {
        "message": "Tenant activated successfully",
        "tenant_id": tenant_id
    }

@app.delete(
    "/admin/tenants/{tenant_id}",
    tags=["Administration"],
    summary="Delete Tenant",
    description="Delete a tenant (soft delete by default).",
    responses={
        200: {
            "description": "Tenant deleted successfully",
            "content": {
                "application/json": {
                    "example": {
                        "message": "Tenant deleted successfully",
                        "tenant_id": "tenant_abc123def456",
                        "hard_delete": False
                    }
                }
            }
        },
        404: {"description": "Tenant not found"}
    }
)
async def delete_tenant(tenant_id: str, hard_delete: bool = False):
    """Delete a tenant."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    success = await tenant_service.delete_tenant(tenant_id, hard_delete)
    
    if not success:
        raise HTTPException(status_code=404, detail="Tenant not found")
    
    return {
        "message": "Tenant deleted successfully",
        "tenant_id": tenant_id,
        "hard_delete": hard_delete
    }

@app.get(
    "/admin/tenants/{tenant_id}/stats",
    tags=["Administration"],
    summary="Get Tenant Statistics",
    description="Get usage statistics for a specific tenant.",
    responses={
        200: {
            "description": "Tenant statistics",
            "content": {
                "application/json": {
                    "example": {
                        "tenant_id": "tenant_abc123def456",
                        "timestamp": 1640995200.0,
                        "audit_logs": {
                            "total_requests": 1250,
                            "error_requests": 25,
                            "avg_process_time": 1.8,
                            "first_request": "2024-01-01T00:00:00",
                            "last_request": "2024-01-15T12:00:00"
                        },
                        "search_operations": {
                            "index": {
                                "count": 100,
                                "total_faces": 500,
                                "total_results": 100
                            }
                        }
                    }
                }
            }
        }
    }
)
async def get_tenant_stats(tenant_id: str):
    """Get tenant statistics."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    stats = await tenant_service.get_tenant_stats(tenant_id)
    return stats

@app.get(
    "/admin/tenants/usage/summary",
    tags=["Administration"],
    summary="Get Tenant Usage Summary",
    description="Get usage summary across all tenants.",
    responses={
        200: {
            "description": "Tenant usage summary",
            "content": {
                "application/json": {
                    "example": {
                        "timestamp": 1640995200.0,
                        "tenant_counts": {
                            "active": 15,
                            "suspended": 2,
                            "deleted": 1
                        },
                        "top_tenants_by_usage": [
                            {
                                "tenant_id": "tenant_abc123def456",
                                "name": "Acme Corporation",
                                "status": "active",
                                "total_requests": 1250,
                                "error_requests": 25
                            }
                        ]
                    }
                }
            }
        }
    }
)
async def get_tenant_usage_summary():
    """Get tenant usage summary."""
    from ..services.tenant_management import get_tenant_management_service
    
    tenant_service = get_tenant_management_service()
    summary = await tenant_service.get_tenant_usage_summary()
    return summary

@app.get(
    "/admin/export/audit-logs",
    tags=["Administration"],
    summary="Export Audit Logs",
    description="Export audit logs in JSON or CSV format with optional filtering.",
    responses={
        200: {
            "description": "Audit logs export",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "audit_logs_1640995200.json",
                        "content_type": "application/json",
                        "size_bytes": 1024000,
                        "format": "json",
                        "export_timestamp": "2024-01-01T12:00:00"
                    }
                }
            }
        }
    }
)
async def export_audit_logs(
    tenant_id: str = None,
    start_date: str = None,
    end_date: str = None,
    format: str = "json"
):
    """Export audit logs."""
    from ..services.data_export import get_data_export_service
    from datetime import datetime
    
    # Parse dates
    start_dt = None
    end_dt = None
    
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format.")
    
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format.")
    
    export_service = get_data_export_service()
    result = await export_service.export_audit_logs(tenant_id, start_dt, end_dt, format)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@app.get(
    "/admin/export/search-logs",
    tags=["Administration"],
    summary="Export Search Audit Logs",
    description="Export search audit logs in JSON or CSV format with optional filtering.",
    responses={
        200: {
            "description": "Search audit logs export",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "search_audit_logs_1640995200.csv",
                        "content_type": "text/csv",
                        "size_bytes": 512000,
                        "format": "csv",
                        "export_timestamp": "2024-01-01T12:00:00"
                    }
                }
            }
        }
    }
)
async def export_search_logs(
    tenant_id: str = None,
    start_date: str = None,
    end_date: str = None,
    format: str = "json"
):
    """Export search audit logs."""
    from ..services.data_export import get_data_export_service
    from datetime import datetime
    
    # Parse dates
    start_dt = None
    end_dt = None
    
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format.")
    
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format.")
    
    export_service = get_data_export_service()
    result = await export_service.export_search_audit_logs(tenant_id, start_dt, end_dt, format)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@app.get(
    "/admin/export/tenant/{tenant_id}",
    tags=["Administration"],
    summary="Export Tenant Data",
    description="Export all data for a specific tenant including images, faces, and audit logs.",
    responses={
        200: {
            "description": "Tenant data export",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "tenant_abc123_export_1640995200.json",
                        "content_type": "application/json",
                        "size_bytes": 2048000,
                        "format": "json",
                        "export_timestamp": "2024-01-01T12:00:00"
                    }
                }
            }
        },
        404: {"description": "Tenant not found"}
    }
)
async def export_tenant_data(tenant_id: str, format: str = "json"):
    """Export all data for a specific tenant."""
    from ..services.data_export import get_data_export_service
    
    export_service = get_data_export_service()
    result = await export_service.export_tenant_data(tenant_id, format)
    
    if "error" in result:
        if "not found" in result["error"].lower():
            raise HTTPException(status_code=404, detail=result["error"])
        else:
            raise HTTPException(status_code=500, detail=result["error"])
    
    return result

@app.get(
    "/admin/export/system-metrics",
    tags=["Administration"],
    summary="Export System Metrics",
    description="Export system-wide metrics and statistics in JSON or CSV format.",
    responses={
        200: {
            "description": "System metrics export",
            "content": {
                "application/json": {
                    "example": {
                        "filename": "system_metrics_1640995200.json",
                        "content_type": "application/json",
                        "size_bytes": 256000,
                        "format": "json",
                        "export_timestamp": "2024-01-01T12:00:00"
                    }
                }
            }
        }
    }
)
async def export_system_metrics(
    start_date: str = None,
    end_date: str = None,
    format: str = "json"
):
    """Export system metrics and statistics."""
    from ..services.data_export import get_data_export_service
    from datetime import datetime
    
    # Parse dates
    start_dt = None
    end_dt = None
    
    if start_date:
        try:
            start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use ISO format.")
    
    if end_date:
        try:
            end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use ISO format.")
    
    export_service = get_data_export_service()
    result = await export_service.export_system_metrics(start_dt, end_dt, format)
    
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    
    return result
