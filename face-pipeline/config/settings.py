from pydantic import BaseModel
import os


"""
Face Pipeline Configuration Settings

Loads configuration from environment variables using simple BaseModel approach.
"""

class Settings(BaseModel):
    # ----- MinIO -----
    MINIO_ENDPOINT: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    MINIO_ACCESS_KEY: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    MINIO_SECRET_KEY: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    MINIO_SECURE: bool = os.getenv("MINIO_SECURE", "false").lower() == "true"
    MINIO_BUCKET_RAW: str = os.getenv("MINIO_BUCKET_RAW", "raw-images")
    MINIO_BUCKET_CROPS: str = os.getenv("MINIO_BUCKET_CROPS", "face-crops")
    MINIO_BUCKET_THUMBS: str = os.getenv("MINIO_BUCKET_THUMBS", "thumbnails")
    MINIO_BUCKET_METADATA: str = os.getenv("MINIO_BUCKET_METADATA", "face-metadata")
    PRESIGN_TTL_SEC: int = int(os.getenv("PRESIGN_TTL_SEC", "600"))

    # ----- Qdrant -----
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
    QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "faces_v1")
    VECTOR_DIM: int = int(os.getenv("VECTOR_DIM", "512"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

    # ----- Face Detection -----
    DET_SIZE: str = os.getenv("DET_SIZE", "640,640")
    DET_SCORE_THRESH: float = float(os.getenv("DET_SCORE_THRESH", "0.45"))
    ONNX_PROVIDERS_CSV: str = os.getenv("ONNX_PROVIDERS_CSV", "CPUExecutionProvider")
    IMAGE_SIZE: int = int(os.getenv("IMAGE_SIZE", "112"))


# Global settings instance
settings = Settings()
