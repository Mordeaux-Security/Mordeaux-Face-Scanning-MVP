"""
Application Settings

Configuration management for the face scanning application with GPU worker support.
"""

import os
from typing import Optional, Dict, Any
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with GPU worker configuration support."""
    
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
        default=1024,
        env="GPU_BATCH_SIZE",
        description="Batch size for GPU operations"
    )
    
    # GPU Worker Configuration
    gpu_worker_enabled: bool = Field(
        default=False,
        env="GPU_WORKER_ENABLED",
        description="Enable Windows GPU worker service"
    )
    gpu_worker_url: str = Field(
        default="http://localhost:8765",
        env="GPU_WORKER_URL",
        description="GPU worker service URL (localhost for native Windows, host.docker.internal for Docker)"
    )
    gpu_worker_timeout: int = Field(
        default=60,
        env="GPU_WORKER_TIMEOUT",
        description="GPU worker request timeout in seconds"
    )
    gpu_worker_max_retries: int = Field(
        default=5,
        env="GPU_WORKER_MAX_RETRIES",
        description="Maximum retries for GPU worker requests"
    )
    gpu_worker_batch_size: int = Field(
        default=1024,
        env="GPU_WORKER_BATCH_SIZE",
        description="Initial batch size for GPU worker requests"
    )
    gpu_worker_health_check_interval: int = Field(
        default=10,
        env="GPU_WORKER_HEALTH_CHECK_INTERVAL",
        description="Health check interval in seconds"
    )
    
    # GPU Resource Monitor Configuration
    gpu_resource_monitor_enabled: bool = Field(
        default=True,
        env="GPU_RESOURCE_MONITOR_ENABLED",
        description="Enable dynamic GPU resource monitoring"
    )
    gpu_resource_monitor_interval: float = Field(
        default=0.3,
        env="GPU_RESOURCE_MONITOR_INTERVAL",
        description="GPU resource monitor polling interval in seconds"
    )
    gpu_target_utilization: float = Field(
        default=0.90,
        env="GPU_TARGET_UTILIZATION",
        description="Target GPU utilization (0.0-1.0)"
    )
    gpu_memory_threshold: float = Field(
        default=0.85,
        env="GPU_MEMORY_THRESHOLD",
        description="Memory threshold for backing off (0.0-1.0)"
    )
    gpu_batch_increment: int = Field(
        default=4,
        env="GPU_BATCH_INCREMENT",
        description="Batch size increment amount"
    )
    gpu_batch_decrement: int = Field(
        default=3,
        env="GPU_BATCH_DECREMENT",
        description="Batch size decrement amount"
    )
    
    # ============================================================================
    # Batch Queue Configuration
    # ============================================================================
    batch_queue_enabled: bool = Field(
        default=True,
        env="BATCH_QUEUE_ENABLED",
        description="Enable batch queue for GPU processing"
    )
    batch_queue_size: int = Field(
        default=64,
        env="BATCH_QUEUE_SIZE",
        description="Number of images per batch"
    )
    batch_queue_max_depth: int = Field(
        default=10,
        env="BATCH_QUEUE_MAX_DEPTH",
        description="Maximum number of batches in queue"
    )
    batch_queue_flush_timeout: float = Field(
        default=2.0,
        env="BATCH_QUEUE_FLUSH_TIMEOUT",
        description="Timeout in seconds before flushing partial batch"
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
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
