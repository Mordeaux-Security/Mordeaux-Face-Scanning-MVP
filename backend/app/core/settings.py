"""
Application Settings

Configuration management for the face scanning application with GPU support.
"""

import os
from typing import Optional, Dict, Any
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with GPU configuration support."""
    
    # ============================================================================
    # Database Configuration
    # ============================================================================
    database_url: str = Field(
        default="postgresql://user:password@localhost:5432/facescan",
        env="DATABASE_URL",
        description="PostgreSQL database connection URL"
    )
    
    # ============================================================================
    # Redis Configuration
    # ============================================================================
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        env="REDIS_URL",
        description="Redis connection URL for caching"
    )
    
    # ============================================================================
    # Storage Configuration
    # ============================================================================
    minio_endpoint: str = Field(
        default="localhost:9000",
        env="MINIO_ENDPOINT",
        description="MinIO server endpoint"
    )
    minio_access_key: str = Field(
        default="minioadmin",
        env="MINIO_ACCESS_KEY",
        description="MinIO access key"
    )
    minio_secret_key: str = Field(
        default="minioadmin",
        env="MINIO_SECRET_KEY",
        description="MinIO secret key"
    )
    minio_bucket: str = Field(
        default="faces",
        env="MINIO_BUCKET",
        description="MinIO bucket name for face images"
    )
    
    # ============================================================================
    # Vector Database Configuration
    # ============================================================================
    qdrant_url: str = Field(
        default="http://localhost:6333",
        env="QDRANT_URL",
        description="Qdrant vector database URL"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None,
        env="QDRANT_API_KEY",
        description="Qdrant API key (optional)"
    )
    
    # ============================================================================
    # GPU Configuration
    # ============================================================================
    
    # Master GPU control
    all_gpu: bool = Field(
        default=False,
        env="ALL_GPU",
        description="Enable GPU acceleration for all operations"
    )
    
    # Granular GPU controls
    face_detection_gpu: bool = Field(
        default=False,
        env="FACE_DETECTION_GPU",
        description="Enable GPU acceleration for face detection"
    )
    face_embedding_gpu: bool = Field(
        default=False,
        env="FACE_EMBEDDING_GPU",
        description="Enable GPU acceleration for face embedding"
    )
    image_processing_gpu: bool = Field(
        default=False,
        env="IMAGE_PROCESSING_GPU",
        description="Enable GPU acceleration for image processing"
    )
    image_enhancement_gpu: bool = Field(
        default=False,
        env="IMAGE_ENHANCEMENT_GPU",
        description="Enable GPU acceleration for image enhancement"
    )
    quality_checks_gpu: bool = Field(
        default=False,
        env="QUALITY_CHECKS_GPU",
        description="Enable GPU acceleration for quality checks"
    )
    
    # GPU backend configuration
    gpu_backend: str = Field(
        default="auto",
        env="GPU_BACKEND",
        description="Preferred GPU backend (auto|cuda|rocm|mps|cpu)"
    )
    gpu_device_id: int = Field(
        default=0,
        env="GPU_DEVICE_ID",
        description="GPU device ID to use"
    )
    
    # GPU memory management
    gpu_memory_limit_gb: float = Field(
        default=8.0,
        env="GPU_MEMORY_LIMIT_GB",
        description="Maximum GPU memory to use in GB"
    )
    gpu_batch_size: int = Field(
        default=32,
        env="GPU_BATCH_SIZE",
        description="Batch size for GPU operations"
    )
    
    # ============================================================================
    # Face Detection Configuration
    # ============================================================================
    face_model_name: str = Field(
        default="buffalo_l",
        env="FACE_MODEL_NAME",
        description="InsightFace model name"
    )
    face_detection_size: tuple = Field(
        default=(640, 640),
        env="FACE_DETECTION_SIZE",
        description="Face detection input size (width, height)"
    )
    min_face_quality: float = Field(
        default=0.5,
        env="MIN_FACE_QUALITY",
        description="Minimum face detection quality score"
    )
    require_face: bool = Field(
        default=True,
        env="REQUIRE_FACE",
        description="Require at least one face in images"
    )
    crop_faces: bool = Field(
        default=True,
        env="CROP_FACES",
        description="Crop and save face regions as thumbnails"
    )
    face_margin: float = Field(
        default=0.2,
        env="FACE_MARGIN",
        description="Margin around face as fraction of face size"
    )
    
    # ============================================================================
    # Crawler Configuration
    # ============================================================================
    max_images_per_site: int = Field(
        default=20,
        env="MAX_IMAGES_PER_SITE",
        description="Maximum images to collect per site"
    )
    max_pages_per_site: int = Field(
        default=5,
        env="MAX_PAGES_PER_SITE",
        description="Maximum pages to crawl per site"
    )
    concurrent_sites: int = Field(
        default=3,
        env="CONCURRENT_SITES",
        description="Maximum number of sites to crawl concurrently"
    )
    max_concurrent_images: int = Field(
        default=10,
        env="MAX_CONCURRENT_IMAGES",
        description="Maximum number of images to process concurrently per site"
    )
    batch_size: int = Field(
        default=25,
        env="BATCH_SIZE",
        description="Batch size for operations"
    )
    timeout: int = Field(
        default=30,
        env="TIMEOUT",
        description="Request timeout in seconds"
    )
    
    # ============================================================================
    # Performance Configuration
    # ============================================================================
    thread_pool_size: int = Field(
        default=4,
        env="THREAD_POOL_SIZE",
        description="Size of thread pool for CPU-intensive operations"
    )
    memory_limit_gb: float = Field(
        default=8.0,
        env="MEMORY_LIMIT_GB",
        description="Maximum memory usage in GB"
    )
    enable_memory_monitoring: bool = Field(
        default=True,
        env="ENABLE_MEMORY_MONITORING",
        description="Enable memory usage monitoring"
    )
    
    # ============================================================================
    # Logging Configuration
    # ============================================================================
    log_level: str = Field(
        default="INFO",
        env="LOG_LEVEL",
        description="Logging level (DEBUG|INFO|WARNING|ERROR)"
    )
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT",
        description="Log message format"
    )
    
    # ============================================================================
    # Security Configuration
    # ============================================================================
    max_content_length: int = Field(
        default=8 * 1024 * 1024,  # 8MB
        env="MAX_CONTENT_LENGTH",
        description="Maximum content length in bytes"
    )
    allowed_content_types: set = Field(
        default={"image/jpeg", "image/jpg", "image/png", "image/gif", "image/webp", "image/bmp"},
        env="ALLOWED_CONTENT_TYPES",
        description="Allowed content types for images"
    )
    blocked_content_types: set = Field(
        default={"image/svg+xml"},
        env="BLOCKED_CONTENT_TYPES",
        description="Blocked content types"
    )
    
    # ============================================================================
    # Development Configuration
    # ============================================================================
    debug: bool = Field(
        default=False,
        env="DEBUG",
        description="Enable debug mode"
    )
    reload: bool = Field(
        default=False,
        env="RELOAD",
        description="Enable auto-reload in development"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration as dictionary."""
        return {
            'all_gpu': self.all_gpu,
            'face_detection_gpu': self.face_detection_gpu,
            'face_embedding_gpu': self.face_embedding_gpu,
            'image_processing_gpu': self.image_processing_gpu,
            'image_enhancement_gpu': self.image_enhancement_gpu,
            'quality_checks_gpu': self.quality_checks_gpu,
            'gpu_backend': self.gpu_backend,
            'gpu_device_id': self.gpu_device_id,
            'gpu_memory_limit_gb': self.gpu_memory_limit_gb,
            'gpu_batch_size': self.gpu_batch_size,
        }
    
    def is_gpu_enabled_for_operation(self, operation: str) -> bool:
        """
        Check if GPU is enabled for a specific operation.
        
        Args:
            operation: Operation name (face_detection, face_embedding, etc.)
            
        Returns:
            True if GPU is enabled for the operation
        """
        if self.all_gpu:
            return True
        
        operation_key = f"{operation}_gpu"
        return getattr(self, operation_key, False)


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings():
    """Reload settings from environment."""
    global _settings
    _settings = Settings()
    return _settings
