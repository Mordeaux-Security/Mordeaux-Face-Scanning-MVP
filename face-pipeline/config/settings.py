"""
Face Pipeline Configuration Settings

Loads configuration from environment variables using pydantic-settings.
Uses the root-level .env file to avoid duplication with docker-compose.yml
"""

from typing import Optional, Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Pipeline configuration settings."""
    
    # ========================================================================
    # Face Detection Settings
    # ========================================================================
    detector_model: str = Field(default="buffalo_l", description="InsightFace model name")
    detector_ctx_id: int = Field(default=-1, description="-1 for CPU, 0+ for GPU")
    detector_size_width: int = Field(default=640, description="Detection width")
    detector_size_height: int = Field(default=640, description="Detection height")
    
    # ========================================================================
    # Embedding Settings
    # ========================================================================
    embedding_dim: int = Field(default=512, description="Embedding dimension")
    normalize_embeddings: bool = Field(default=True, description="L2 normalize embeddings")
    
    # ========================================================================
    # Quality Thresholds
    # ========================================================================
    min_face_quality: float = Field(default=0.7, description="Minimum detection score")
    min_face_size: int = Field(default=30, description="Minimum face size in pixels")
    blur_min_variance: float = Field(default=120.0, description="Minimum Laplacian variance (higher = sharper)")
    min_sharpness: float = Field(default=100.0, description="Minimum sharpness score")
    min_brightness: float = Field(default=30.0, description="Minimum average brightness")
    max_brightness: float = Field(default=225.0, description="Maximum average brightness")
    max_pose_angle: float = Field(default=45.0, description="Maximum pose angle in degrees")
    min_overall_quality: float = Field(default=0.7, description="Minimum overall quality score")
    
    # ========================================================================
    # Storage Settings (MinIO/S3)
    # ========================================================================
    # Supports both MINIO_* (preferred) and S3_* (legacy) prefixes
    minio_endpoint: str = Field(default="localhost:9000", description="MinIO endpoint")
    minio_access_key: str = Field(default="changeme", description="MinIO access key")
    minio_secret_key: str = Field(default="changeme", description="MinIO secret key")
    minio_secure: bool = Field(default=False, description="Use HTTPS for MinIO")
    
    # Buckets
    minio_bucket_raw: str = Field(default="raw-images", description="Raw images bucket")
    minio_bucket_crops: str = Field(default="face-crops", description="Face crops bucket")
    minio_bucket_thumbs: str = Field(default="thumbnails", description="Thumbnails bucket")
    minio_bucket_metadata: str = Field(default="face-metadata", description="Metadata bucket")
    
    # Legacy S3 aliases (fallback to MINIO_* if not set)
    s3_endpoint: Optional[str] = Field(default=None, description="S3 endpoint (legacy)")
    s3_access_key: Optional[str] = Field(default=None, description="S3 access key (legacy)")
    s3_secret_key: Optional[str] = Field(default=None, description="S3 secret key (legacy)")
    s3_use_ssl: Optional[bool] = Field(default=None, description="Use SSL (legacy)")
    
    # ========================================================================
    # Vector Database Settings
    # ========================================================================
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant URL")
    qdrant_collection: str = Field(default="faces_v1", description="Qdrant collection name")
    qdrant_api_key: Optional[str] = Field(default=None, description="Qdrant API key (optional)")
    
    # Vector index settings
    vector_dim: int = Field(default=512, description="Vector dimension")
    similarity_threshold: float = Field(default=0.6, description="Similarity threshold")
    
    # Alternative: Pinecone
    pinecone_api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    pinecone_environment: Optional[str] = Field(default=None, description="Pinecone environment")
    pinecone_index: str = Field(default="faces_v1", description="Pinecone index name")
    
    # ========================================================================
    # Pipeline Configuration
    # ========================================================================
    max_faces_per_image: int = Field(default=10, description="Max faces to process per image")
    max_concurrent: int = Field(default=4, description="Max concurrent processing tasks")
    job_timeout_sec: int = Field(default=300, description="Job timeout in seconds")
    batch_size: int = Field(default=32, description="Batch size for processing")
    enable_deduplication: bool = Field(default=True, description="Enable duplicate detection")
    presign_ttl_sec: int = Field(default=600, description="Presigned URL TTL in seconds")
    
    # ========================================================================
    # API Configuration
    # ========================================================================
    api_host: str = Field(default="0.0.0.0", description="API host")
    api_port: int = Field(default=8000, description="API port")
    api_workers: int = Field(default=4, description="API worker count")
    api_reload: bool = Field(default=False, description="Enable auto-reload")
    cors_origins: str = Field(default="*", description="CORS allowed origins")
    
    # ========================================================================
    # Logging & Monitoring
    # ========================================================================
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: Literal["json", "text"] = Field(default="json", description="Log format")
    enable_metrics: bool = Field(default=True, description="Enable Prometheus metrics")
    metrics_port: int = Field(default=9090, description="Metrics port")
    
    # ========================================================================
    # Pydantic Settings Configuration
    # ========================================================================
    model_config = SettingsConfigDict(
        # Look for .env in parent directory (project root)
        env_file="../.env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore"
    )
    
    # ========================================================================
    # Property helpers for S3 legacy compatibility
    # ========================================================================
    @property
    def storage_endpoint(self) -> str:
        """Get storage endpoint (prefers MINIO_*, falls back to S3_*)."""
        return self.s3_endpoint or self.minio_endpoint
    
    @property
    def storage_access_key(self) -> str:
        """Get storage access key (prefers MINIO_*, falls back to S3_*)."""
        return self.s3_access_key or self.minio_access_key
    
    @property
    def storage_secret_key(self) -> str:
        """Get storage secret key (prefers MINIO_*, falls back to S3_*)."""
        return self.s3_secret_key or self.minio_secret_key
    
    @property
    def storage_use_ssl(self) -> bool:
        """Get storage SSL setting (prefers MINIO_*, falls back to S3_*)."""
        if self.s3_use_ssl is not None:
            return self.s3_use_ssl
        return self.minio_secure


# Global settings instance
settings = Settings()
