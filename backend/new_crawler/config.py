"""
Configuration Management for New Crawler System

Handles environment variable loading, Docker/Windows addressing,
and provides centralized configuration for all crawler components.
"""

import os
import logging
import redis
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# Singleton pattern per process
_config_instance = None


class CrawlerConfig(BaseSettings):
    """Configuration for the new crawler system."""
    
    # Environment
    environment: str = "development"
    log_level: str = "info"
    
    # Redis Configuration
    redis_url: str = "redis://redis:6379/0"
    redis_max_connections: int = 20
    redis_retry_on_timeout: bool = True
    
    # Crawling Limits
    nc_max_pages_per_site: int = 5
    nc_max_images_per_site: int = 10
    
    # Debug Logging
    nc_debug_logging: bool = True
    nc_gpu_worker_logging: bool = True
    
    # Queue Configuration
    nc_batch_size: int = 64
    nc_max_queue_depth: int = 512
    nc_crawler_concurrency: int = 32
    nc_extractor_concurrency: int = 16
    nc_cache_ttl_days: int = 90
    
    # Worker Configuration adds to 7 (8 cores-1 for Orchestrator)
    num_crawlers: int = 2
    num_extractors: int = 4
    num_gpu_processors: int = 1
    
    # GPU Worker Configuration
    gpu_worker_enabled: bool = True
    gpu_worker_url: str = "http://host.docker.internal:8765"
    gpu_worker_timeout: float = 60.0
    gpu_worker_max_retries: int = 3
    
    # HTTP Configuration
    nc_http_timeout: float = 30.0
    nc_js_render_timeout: float = 15.0
    nc_max_redirects: int = 3
    nc_max_retries: int = 3
    
    # Selector Mining Configuration
    nc_use_3x3_mining: bool = True
    nc_max_selector_patterns: int = 30
    
    # Storage Configuration
    s3_endpoint: Optional[str] = None
    s3_region: str = "us-east-1"
    s3_bucket_raw: str = "raw-images"
    s3_bucket_thumbs: str = "thumbnails"
    s3_access_key: Optional[str] = None
    s3_secret_key: Optional[str] = None
    s3_use_ssl: bool = False
    
    # Face Detection Configuration
    min_face_quality: float = 0.5
    min_face_size: int = 80
    face_margin: float = 0.2
    
    # Back-pressure Configuration
    backpressure_threshold: float = 0.75
    backpressure_check_interval: float = 2.0
    
    # Monitoring Configuration
    metrics_interval: float = 30.0
    health_check_interval: float = 10.0
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_prefix": ""
    }
    
    @field_validator('gpu_worker_url')
    @classmethod
    def validate_gpu_worker_url(cls, v):
        """Validate GPU worker URL and handle Docker/Windows addressing."""
        if not v.startswith(('http://', 'https://')):
            raise ValueError('GPU worker URL must start with http:// or https://')
        
        # Parse URL to check if it's valid
        try:
            parsed = urlparse(v)
            if not parsed.hostname:
                raise ValueError('GPU worker URL must have a valid hostname')
        except Exception as e:
            raise ValueError(f'Invalid GPU worker URL: {e}')
        
        return v
    
    @field_validator('redis_url')
    @classmethod
    def validate_redis_url(cls, v):
        """Validate Redis URL."""
        if not v.startswith(('redis://', 'rediss://')):
            raise ValueError('Redis URL must start with redis:// or rediss://')
        return v
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v):
        """Validate log level."""
        allowed = ['debug', 'info', 'warning', 'error', 'critical']
        if v.lower() not in allowed:
            raise ValueError(f'Log level must be one of {allowed}')
        return v.lower()
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v):
        """Validate environment."""
        allowed = ['development', 'staging', 'production']
        if v.lower() not in allowed:
            raise ValueError(f'Environment must be one of {allowed}')
        return v.lower()
    
    @property
    def is_docker(self) -> bool:
        """Check if running in Docker container."""
        return os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
    
    @property
    def is_windows(self) -> bool:
        """Check if running on Windows."""
        return os.name == 'nt'
    
    @property
    def gpu_worker_host_resolved(self) -> str:
        """Resolve GPU worker URL based on environment."""
        parsed = urlparse(self.gpu_worker_url)
        
        # If already using host.docker.internal, keep it as-is
        if 'host.docker.internal' in parsed.netloc:
            logger.debug(f"Using Docker host gateway for GPU worker: {self.gpu_worker_url}")
            return self.gpu_worker_url
        
        # Check if running in Docker
        in_docker = os.path.exists('/.dockerenv') or os.environ.get('DOCKER_CONTAINER') == 'true'
        
        if in_docker and parsed.hostname in ['localhost', '127.0.0.1']:
            # Replace localhost with host.docker.internal for Docker->Windows communication
            new_url = parsed._replace(netloc=f"host.docker.internal:{parsed.port or 8765}")
            resolved_url = new_url.geturl()
            logger.debug(f"Replaced localhost with host.docker.internal: {resolved_url}")
            return resolved_url
        else:
            # Native environment or already properly configured
            logger.debug(f"Using GPU worker URL as-is: {self.gpu_worker_url}")
            return self.gpu_worker_url
    
    @property
    def queue_names(self) -> Dict[str, str]:
        """Get queue names with prefix."""
        return {
            'sites': 'nc:sites',
            'candidates': 'nc:candidates',
            'images': 'nc:images',
            'results': 'nc:results',
            'cpu_fallback': 'nc:cpu_fallback'
        }
    
    @property
    def cache_keys(self) -> Dict[str, str]:
        """Get cache key patterns."""
        return {
            'phash': 'nc:cache:phash:{phash}',
            'site_stats': 'nc:cache:site_stats:{site_id}',
            'processing_stats': 'nc:cache:processing_stats:{site_id}'
        }
    
    def get_queue_name(self, queue_type: str) -> str:
        """Get full queue name for a queue type."""
        if queue_type not in self.queue_names:
            raise ValueError(f'Unknown queue type: {queue_type}')
        return self.queue_names[queue_type]
    
    def get_cache_key(self, cache_type: str, **kwargs) -> str:
        """Get cache key for a cache type."""
        if cache_type not in self.cache_keys:
            raise ValueError(f'Unknown cache type: {cache_type}')
        return self.cache_keys[cache_type].format(**kwargs)
    
    def validate_environment(self):
        """Validate environment configuration."""
        logger.info("Validating crawler configuration...")
        
        # Check Redis connectivity
        try:
            client = redis.from_url(self.redis_url, decode_responses=False)
            client.ping()
            logger.info(f"Redis connection validated: {self.redis_url}")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
        
        # Validate GPU worker URL format
        if not self.gpu_worker_url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid GPU worker URL format: {self.gpu_worker_url}")
        
        logger.info(f"GPU worker URL validated: {self.gpu_worker_url}")
        
        # Log configuration summary
        logger.info(f"Configuration validated: crawlers={self.num_crawlers}, "
                   f"extractors={self.num_extractors}, gpu_processors={self.num_gpu_processors}")
    
    def log_configuration(self):
        """Log current configuration (without sensitive data)."""
        logger.info("=== New Crawler Configuration ===")
        logger.info(f"Environment: {self.environment}")
        logger.info(f"Log Level: {self.log_level}")
        logger.info(f"Docker: {self.is_docker}")
        logger.info(f"Windows: {self.is_windows}")
        logger.info(f"Redis URL: {self.redis_url}")
        logger.info(f"GPU Worker URL: {self.gpu_worker_url}")
        logger.info(f"GPU Worker Host: {self.gpu_worker_host_resolved}")
        logger.info(f"Batch Size: {self.nc_batch_size}")
        logger.info(f"Max Queue Depth: {self.nc_max_queue_depth}")
        logger.info(f"Crawler Concurrency: {self.nc_crawler_concurrency}")
        logger.info(f"Extractor Concurrency: {self.nc_extractor_concurrency}")
        logger.info(f"Use 3x3 Mining: {self.nc_use_3x3_mining}")
        logger.info(f"Max Selector Patterns: {self.nc_max_selector_patterns}")
        logger.info(f"HTTP Timeout: {self.nc_http_timeout}s")
        logger.info(f"GPU Timeout: {self.gpu_worker_timeout}s")
        logger.info(f"Cache TTL: {self.nc_cache_ttl_days} days")
        logger.info("================================")


def get_config() -> CrawlerConfig:
    """Get singleton config instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = CrawlerConfig()
        _config_instance.validate_environment()
    return _config_instance


def reload_config():
    """Reload configuration from environment."""
    global _config_instance
    _config_instance = None
    return get_config()


def validate_configuration() -> bool:
    """Validate configuration and log warnings for missing required fields."""
    config = get_config()
    warnings = []
    
    # Check required fields for production
    if config.environment == "production":
        if not config.s3_access_key or not config.s3_secret_key:
            warnings.append("S3 credentials not set for production environment")
        if not config.s3_endpoint:
            warnings.append("S3 endpoint not set for production environment")
    
    # Check MinIO configuration for development
    if config.environment == "development" and not config.s3_endpoint:
        warnings.append("S3_ENDPOINT not set for development environment")
    
    # Check GPU worker configuration
    if config.gpu_worker_enabled and not config.gpu_worker_url:
        warnings.append("GPU worker enabled but URL not configured")
    
    for warning in warnings:
        logger.warning(warning)
    
    return len(warnings) == 0
