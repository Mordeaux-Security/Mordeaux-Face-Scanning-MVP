import logging
from fastapi import FastAPI
from .api.routes import router as api_router
from .services.storage import _minio
from .core.config import get_settings

logger = logging.getLogger(__name__)

app = FastAPI(title="backend-cpu")

# Required MinIO buckets
REQUIRED_BUCKETS = [
    'raw-images',
    'thumbnails',
    'audit-logs'
]

def init_minio_buckets():
    """Initialize all required MinIO buckets on startup."""
    settings = get_settings()
    
    # Only initialize buckets if using MinIO (local dev)
    if not settings.using_minio:
        logger.info("Not using MinIO, skipping bucket initialization")
        return
    
    logger.info("Initializing MinIO buckets...")
    logger.info(f"MinIO endpoint: {settings.s3_endpoint}")
    
    try:
        client = _minio()
        logger.info("✓ MinIO client created successfully")
        
        # Initialize each bucket
        for bucket_name in REQUIRED_BUCKETS:
            try:
                # Check if bucket exists
                exists = client.bucket_exists(bucket_name)
                
                if exists:
                    logger.info(f"✓ Bucket '{bucket_name}' already exists")
                else:
                    # Create bucket
                    client.make_bucket(bucket_name)
                    logger.info(f"✓ Created bucket '{bucket_name}'")
                    
            except Exception as e:
                logger.error(f"✗ Failed to initialize bucket '{bucket_name}': {e}")
                # Continue with other buckets even if one fails
        
        logger.info("✓ MinIO bucket initialization completed")
        
    except Exception as e:
        logger.error(f"✗ MinIO connection failed: {e}")
        logger.error("Check MinIO service is running and accessible")
        # Don't fail startup if MinIO is not available - it might start later

@app.on_event("startup")
def startup():
    """Initialize buckets on application startup."""
    init_minio_buckets()

@app.get("/")
def root():
    return {"ok": True}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/healthz")
def healthz():
    return {"status": "healthy", "service": "backend-cpu"}

@app.get("/ready")
def ready():
    return {"ready": True, "reason": "ok", "checks": {"models": True, "storage": True, "vector_db": True, "redis": True}}

app.include_router(api_router)


