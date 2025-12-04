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
    log_level: str = "error"
    
    # Redis Configuration
    redis_url: str = "redis://redis:6379/0"
    redis_max_connections: int = 500  # Increased for diagnostic logging and high concurrency
    redis_retry_on_timeout: bool = True
    
    # Crawling Limits
    nc_max_pages_per_site: int = -1  # Set to -1 for unlimited crawling
    nc_max_images_per_site: int = 10
    nc_strict_limits: bool = True  # When enabled, stops feeding queues and removes existing items when site limits are reached
    
    # Debug Logging
    nc_debug_logging: bool = True
    nc_gpu_worker_logging: bool = True
    nc_diagnostic_logging: bool = True  # Enable diagnostic bottleneck logging
    nc_diagnostic_log_interval: int = 50  # Interval for periodic throughput logs
    
    # Queue Configuration
    nc_batch_size: int = 256  # DEPRECATED: Old extractor batch threshold (no longer used - extractors push individually to gpu:inbox)
    nc_max_queue_depth: int = 128  # Mac: 128, AMD: 4096
    nc_extractor_concurrency: int = 16  # Mac: 16, AMD: 1024 - Total concurrent downloads across all extractor workers (divided among workers)
    nc_extractor_batch_pop_size: int = 16  # Mac: 16, AMD: 50 - Number of candidates to pop at once from queue
    nc_url_dedup_ttl_hours: int = 24  # TTL for URL deduplication set
    nc_cache_ttl_days: int = 90
    
    # HTTP Performance Configuration
    nc_skip_head_check: bool = True  # Skip HEAD requests when HTML metadata is available
    
    # Crawler Performance Configuration
    nc_max_concurrent_sites_per_worker: int = 2  # Mac: 2, AMD: 32 - Max sites processed concurrently per crawler worker (eliminates site switching delays)
    
    # GPU Performance Configuration
    nc_max_concurrent_batches_per_worker: int = 2  # Max batches processed concurrently per GPU worker (improves GPU utilization) - NOTE: Overridden by GPU scheduler (max 2 inflight)
    nc_batch_flush_timeout: float = 5.0  # DEPRECATED: Max seconds before forcing batch flush (no longer used with GPU scheduler)
    
    # Worker Configuration adds to 7 (8 cores-1 for Orchestrator)
    # Each extractor worker runs nc_extractor_concurrency // num_extractors concurrent download tasks
    # ideal for amd computer (3, 8, 1, 1), Mac: (1, 2, 1, 1)
    num_crawlers: int = 1  # Mac: 1 crawler worker
    num_extractors: int = 2  # Mac: 2 extractor workers
    num_gpu_processors: int = 1  # Mac: 1 GPU processor worker
    num_storage_workers: int = 1  # Number of storage worker processes
    
    # GPU Worker Configuration
    gpu_worker_enabled: bool = True
    gpu_worker_url: str = "http://host.docker.internal:8765"
    gpu_worker_timeout: float = 60.0
    gpu_worker_max_retries: int = 3
    
    # DEPRECATED: Old batching configuration (replaced by GPU scheduler)
    gpu_min_batch_size: int = 64  # DEPRECATED: No longer used with GPU scheduler
    gpu_batch_flush_ms: int = 3000  # DEPRECATED: No longer used with GPU scheduler
    
    # GPU Scheduler Configuration (new centralized batching)
    gpu_target_batch: int = 8  # Mac: 8, AMD: 512 - Target batch size for GPU processing
    gpu_max_wait_ms: int = 12  # Max milliseconds to wait before launching early batch
    gpu_min_launch_ms: int = 100  # Minimum milliseconds between batch launches (Windows/AMD stability)
    gpu_inbox_key: str = "gpu:inbox"  # Redis queue key for GPU input (single FIFO queue)
    
    # GPU Processor Worker Configuration
    image_processing_idle_wait: float = 0.05  # Mac: 0.05, AMD: 0.002 - Sleep duration (seconds) when idle in main loop
    
    # MinIO Connection Pool Configuration
    minio_max_pool_size: int = 50
    minio_pool_timeout: float = 30.0
    
    # HTTP Configuration
    nc_http_timeout: float = 30.0
    nc_js_render_timeout: float = 120.0  # Mac: 120.0, AMD: 20.0
    nc_max_redirects: int = 3
    nc_max_retries: int = 3
    nc_retry_base_delay: float = 1.0  # Base delay for exponential backoff (seconds)
    nc_retry_max_delay: float = 30.0  # Maximum delay for exponential backoff (seconds)
    nc_retry_jitter: float = 0.5  # Random jitter added to retry delay (seconds)
    nc_circuit_breaker_failure_threshold: int = 5  # Number of failures before opening circuit
    nc_circuit_breaker_open_timeout_base: float = 30.0  # Base timeout for circuit breaker (seconds)
    
    # JavaScript Rendering Configuration
    nc_js_wait_strategy: str = "fixed"  # "fixed" | "networkidle" | "both"
    nc_js_wait_timeout: float = 5.0  # seconds to wait for fixed strategy
    nc_js_networkidle_timeout: float = 30.0  # Mac: 30.0, AMD: 3.0 - timeout for network idle strategy
    # First visit strategy: fetch both HTTP and JS, pick best by candidate count
    nc_js_first_visit_compare: bool = True
    # Max concurrent Playwright renders
    nc_js_concurrency: int = 2  # Mac: 2, AMD: 32
    # Browser pool size for JavaScript rendering
    nc_js_browser_pool_size: int = 1  # Mac: 1, AMD: 32
    # Block unnecessary resources (CSS, fonts, media) to speed up rendering
    nc_js_block_resources: bool = True
    # Aggressive HTTP-first strategy (require 3x more images for JS to win)
    nc_js_aggressive_http: bool = True
    
    # Image Extraction Configuration
    nc_extract_background_images: bool = True
    nc_extract_srcset_images: bool = True
    nc_extract_data_attributes: bool = True
    
    # Advanced HTTP Configuration
    nc_same_origin_redirects_only: bool = True
    nc_blocklist_redirect_hosts: list[str] = ['progress-tm.com', 'google.com']
    nc_realistic_headers: bool = True
    
    # Advanced JavaScript Configuration
    nc_js_wait_selectors: str = 'img, picture source, noscript img, [style*="background-image"]'
    nc_capture_network_images: bool = True
    nc_extract_script_images: bool = True
    nc_extract_noscript_images: bool = True
    nc_extract_jsonld_images: bool = True
    
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
    
    # Vector Database Configuration
    vectorization_enabled: bool = True  # Enable/disable vector DB upserts after storage
    default_tenant_id: str = "demo-tenant"  # Default tenant ID for crawled faces (matches website default)
    qdrant_url: str = "http://qdrant:6333"  # Qdrant URL for vector storage
    vector_index: str = "faces_v1"  # Collection name for face vectors
    
    # Face Detection Configuration
    min_face_quality: float = 0.9
    min_face_size: int = 30
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
    
    @field_validator('nc_js_wait_strategy')
    @classmethod
    def validate_js_wait_strategy(cls, v):
        """Validate JavaScript wait strategy."""
        allowed = ['fixed', 'networkidle', 'both']
        if v.lower() not in allowed:
            raise ValueError(f'JS wait strategy must be one of {allowed}')
        return v.lower()
    
    @field_validator('nc_js_wait_timeout')
    @classmethod
    def validate_js_wait_timeout(cls, v):
        """Validate JavaScript wait timeout."""
        if v < 1.0 or v > 30.0:
            raise ValueError('JS wait timeout must be between 1.0 and 30.0 seconds')
        return v
    
    @field_validator('nc_blocklist_redirect_hosts')
    @classmethod
    def validate_blocklist_redirect_hosts(cls, v):
        """Validate redirect host blocklist."""
        if not isinstance(v, list):
            raise ValueError('Blocklist must be a list of strings')
        for host in v:
            if not isinstance(host, str) or not host.strip():
                raise ValueError('All blocklist entries must be non-empty strings')
        return v
    
    @field_validator('nc_js_wait_selectors')
    @classmethod
    def validate_js_wait_selectors(cls, v):
        """Validate JavaScript wait selectors."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError('JS wait selectors must be a non-empty string')
        return v.strip()
    
    @field_validator('nc_max_pages_per_site')
    @classmethod
    def validate_max_pages(cls, v):
        """Validate max pages per site."""
        if v < -1 or v == 0:
            raise ValueError('Max pages must be positive or -1 for unlimited')
        return v
    
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
            'storage': 'nc:storage',
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
        logger.info(f"GPU Min Batch Size: {self.gpu_min_batch_size}")
        logger.info(f"Max Queue Depth: {self.nc_max_queue_depth}")
        logger.info(f"Extractor Concurrency: {self.nc_extractor_concurrency}")
        logger.info(f"Max Concurrent Sites per Worker: {self.nc_max_concurrent_sites_per_worker}")
        logger.info(f"Max Concurrent Batches per Worker: {self.nc_max_concurrent_batches_per_worker}")
        logger.info(f"Batch Flush Timeout: {self.nc_batch_flush_timeout}s")
        logger.info(f"Use 3x3 Mining: {self.nc_use_3x3_mining}")
        logger.info(f"Max Selector Patterns: {self.nc_max_selector_patterns}")
        logger.info(f"HTTP Timeout: {self.nc_http_timeout}s")
        logger.info(f"GPU Timeout: {self.gpu_worker_timeout}s")
        logger.info(f"JS Wait Strategy: {self.nc_js_wait_strategy}")
        logger.info(f"JS Wait Timeout: {self.nc_js_wait_timeout}s")
        logger.info(f"JS Network Idle Timeout: {self.nc_js_networkidle_timeout}s")
        logger.info(f"Extract Background Images: {self.nc_extract_background_images}")
        logger.info(f"Extract Srcset Images: {self.nc_extract_srcset_images}")
        logger.info(f"Extract Data Attributes: {self.nc_extract_data_attributes}")
        logger.info(f"Same Origin Redirects Only: {self.nc_same_origin_redirects_only}")
        logger.info(f"Blocklist Redirect Hosts: {self.nc_blocklist_redirect_hosts}")
        logger.info(f"Realistic Headers: {self.nc_realistic_headers}")
        logger.info(f"JS Wait Selectors: {self.nc_js_wait_selectors}")
        logger.info(f"Capture Network Images: {self.nc_capture_network_images}")
        logger.info(f"Extract Script Images: {self.nc_extract_script_images}")
        logger.info(f"Extract Noscript Images: {self.nc_extract_noscript_images}")
        logger.info(f"Extract JSON-LD Images: {self.nc_extract_jsonld_images}")
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
