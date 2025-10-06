import os
from typing import Optional
from pydantic import validator
from pydantic_settings import BaseSettings
import logging

logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    """Centralized configuration management with validation and defaults."""
    
    # Environment
    environment: str = "development"
    log_level: str = "info"
    
    # Database
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "mordeaux"
    postgres_user: str = "mordeaux"
    postgres_password: str
    
    # Redis
    redis_url: str = "redis://redis:6379/0"
    
    # Storage (MinIO/S3)
    s3_endpoint: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_bucket_raw: str = "raw-images"
    s3_bucket_thumbs: str = "thumbnails"
    s3_bucket_audit: str = "audit-logs"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_use_ssl: bool = False
    
    # Vector Database
    qdrant_url: str = "http://qdrant:6333"
    vector_index: str = "faces_v1"
    pinecone_api_key: Optional[str] = None
    pinecone_index: str = "faces_v1"
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    
    # Presigned URLs
    presigned_url_ttl: int = 600  # 10 minutes in seconds
    
    # Data Retention
    crawled_thumbs_retention_days: int = 90
    user_query_images_retention_hours: int = 24
    
    # Performance
    max_image_size_mb: int = 10
    p95_latency_threshold_seconds: float = 5.0
    
    # UI Defaults
    default_top_k: int = 10
    default_similarity_threshold: float = 0.25
    
    # Celery
    celery_broker_url: str = "redis://redis:6379/1"
    celery_result_backend: str = "redis://redis:6379/2"
    
    @validator('environment')
    def validate_environment(cls, v):
        allowed = ['development', 'staging', 'production']
        if v.lower() not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v.lower()
    
    @validator('log_level')
    def validate_log_level(cls, v):
        allowed = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in allowed:
            raise ValueError(f'Log level must be one of {allowed}')
        return v.lower()
    
    @validator('presigned_url_ttl')
    def validate_presigned_url_ttl(cls, v):
        if v > 600:  # 10 minutes max
            raise ValueError('Presigned URL TTL cannot exceed 10 minutes (600 seconds)')
        return v
    
    @validator('max_image_size_mb')
    def validate_max_image_size(cls, v):
        if v > 10:
            raise ValueError('Maximum image size cannot exceed 10MB')
        return v
    
    @validator('default_top_k')
    def validate_default_top_k(cls, v):
        if v < 1 or v > 100:
            raise ValueError('Default top_k must be between 1 and 100')
        return v
    
    @validator('default_similarity_threshold')
    def validate_default_similarity_threshold(cls, v):
        if v < 0.0 or v > 1.0:
            raise ValueError('Default similarity threshold must be between 0.0 and 1.0')
        return v
    
    @property
    def using_pinecone(self) -> bool:
        """Determine if using Pinecone based on environment and API key."""
        return bool(self.pinecone_api_key) and self.environment == "production"
    
    @property
    def using_minio(self) -> bool:
        """Determine if using MinIO based on S3 endpoint."""
        return bool(self.s3_endpoint)
    
    @property
    def max_image_size_bytes(self) -> int:
        """Get maximum image size in bytes."""
        return self.max_image_size_mb * 1024 * 1024
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
_settings = None

def get_settings() -> Settings:
    """Get global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        logger.info(f"Configuration loaded for environment: {_settings.environment}")
    return _settings

def validate_configuration():
    """Validate all configuration values and log warnings for missing required fields."""
    settings = get_settings()
    
    warnings = []
    
    # Check required fields for production
    if settings.environment == "production":
        if not settings.pinecone_api_key:
            warnings.append("PINECONE_API_KEY not set for production environment")
        if not settings.s3_access_key or not settings.s3_secret_key:
            warnings.append("S3 credentials not set for production environment")
    
    # Check MinIO configuration for development
    if settings.environment == "development" and not settings.s3_endpoint:
        warnings.append("S3_ENDPOINT not set for development environment")
    
    for warning in warnings:
        logger.warning(warning)
    
    return len(warnings) == 0
