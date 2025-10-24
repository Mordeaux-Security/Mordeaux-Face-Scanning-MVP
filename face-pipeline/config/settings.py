from pydantic import BaseModel
import os


"""
Face Pipeline Configuration Settings

Loads configuration from environment variables using simple BaseModel approach.
"""

class Settings(BaseModel):
    # ----- MinIO -----
    minio_endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
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
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "faces_v1")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "faces_v1")
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


# Global settings instance
settings = Settings()
