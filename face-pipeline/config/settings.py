from pydantic import BaseModel, Field, field_validator
import os


"""
Face Pipeline Configuration Settings

Loads configuration from environment variables using simple BaseModel approach.
"""

class Settings(BaseModel):
    # ----- MinIO -----
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_EXTERNAL_ENDPOINT: str = os.getenv("MINIO_EXTERNAL_ENDPOINT", "")  # External URL for browser access
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    MINIO_BUCKET_RAW: str = os.getenv("MINIO_BUCKET_RAW", "raw-images")
    MINIO_BUCKET_CROPS: str = os.getenv("MINIO_BUCKET_CROPS", "face-crops")
    MINIO_BUCKET_THUMBS: str = os.getenv("MINIO_BUCKET_THUMBS", "thumbnails")
    MINIO_BUCKET_METADATA: str = os.getenv("MINIO_BUCKET_METADATA", "face-metadata")
    PRESIGN_TTL_SEC: int = int(os.getenv("PRESIGN_TTL_SEC", "600"))

    # ----- Qdrant -----
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    # Backward/alternate attribute used by some modules
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "faces_v1")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "faces_v1")
    IDENTITY_COLLECTION: str = os.getenv("IDENTITY_COLLECTION", "identities_v1")
    VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "512"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

    # ----- Face Detection -----
    DET_SIZE: str = os.getenv("DET_SIZE", "640,640")
    DET_SCORE_THRESH: float = float(os.getenv("DET_SCORE_THRESH", "0.45"))
    ONNX_PROVIDERS_CSV: str = os.getenv("ONNX_PROVIDERS_CSV", "CPUExecutionProvider")
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "112"))
    
    # ----- Logging -----
    log_level: str = os.getenv("LOG_LEVEL", "info")
    
    # ----- CORS -----
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    
    # ----- API Configuration -----
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    
    # ----- Pipeline Configuration -----
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "4"))
    min_face_size: int = int(os.getenv("MIN_FACE_SIZE", "32"))
    blur_min_variance: float = float(os.getenv("BLUR_MIN_VARIANCE", "100.0"))

    # ----- Feature Flags -----
    enable_global_dedup: bool = os.getenv("ENABLE_GLOBAL_DEDUP", "false").lower() == "true"
    enable_queue_worker: bool = os.getenv("ENABLE_QUEUE_WORKER", "false").lower() == "true"

    # ----- Redis Configuration -----
    redis_url: str = os.getenv("REDIS_URL", "redis://redis:6379/0")
    redis_stream_name: str = os.getenv("REDIS_STREAM_NAME", "face-processing-queue")
    redis_group_name: str = os.getenv("REDIS_GROUP_NAME", "pipeline")
    redis_consumer_name: str = os.getenv("REDIS_CONSUMER_NAME", "")  # Auto-generated if empty

    # ----- Worker Configuration -----
    max_worker_concurrency: int = Field(default=5, env="MAX_WORKER_CONCURRENCY")
    worker_batch_size: int = Field(default=10, env="WORKER_BATCH_SIZE")

    # ----- Dedup Configuration -----
    dedup_ttl_seconds: int = Field(default=3600, env="DEDUP_TTL_SECONDS")  # 1h default
    dedup_max_hamming: int = Field(default=8, env="DEDUP_MAX_HAMMING")  # Hamming distance threshold

    # ----- API Configuration -----
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8001"))
    api_reload: bool = os.getenv("API_RELOAD", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")

    # ----- Quality Thresholds -----
    min_face_size: int = int(os.getenv("MIN_FACE_SIZE", "80"))
    blur_min_variance: float = float(os.getenv("BLUR_MIN_VARIANCE", "120.0"))
    min_overall_quality: float = float(os.getenv("MIN_OVERALL_QUALITY", "0.7"))

    # ----- Pipeline Settings -----
    max_faces_per_image: int = int(os.getenv("MAX_FACES_PER_IMAGE", "10"))
    enable_deduplication: bool = os.getenv("ENABLE_DEDUPLICATION", "true").lower() == "true"
    max_concurrent: int = int(os.getenv("MAX_CONCURRENT", "5"))

# Global settings instance
settings = Settings()


# Backward compatibility shim for alternative env var names
def _apply_compat_shim():
    """Support alternative environment variable names for backward compatibility."""
    import os
    
    # Map old names to new names
    compat_map = {
        'FACE_STREAM': 'REDIS_STREAM_NAME',
        'FACE_GROUP': 'REDIS_GROUP_NAME',
        'FACE_CONSUMER': 'REDIS_CONSUMER_NAME',
        'GLOBAL_DEDUP_ENABLED': 'ENABLE_GLOBAL_DEDUP',
        'GLOBAL_DEDUP_TTL_SEC': 'DEDUP_TTL_SECONDS',
        'GLOBAL_DEDUP_MAX_HAMMING': 'DEDUP_MAX_HAMMING',
    }
    
    for old_name, new_name in compat_map.items():
        if old_name in os.environ and new_name not in os.environ:
            os.environ[new_name] = os.environ[old_name]

_apply_compat_shim()
