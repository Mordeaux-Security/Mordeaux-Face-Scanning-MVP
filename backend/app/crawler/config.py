"""
Unified Configuration System for Crawler

Centralized configuration management with environment variable integration,
type safety, and validation. Replaces scattered constants across multiple files.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Set, Tuple
from pathlib import Path


@dataclass
class CrawlerConfig:
    """Unified configuration for all crawler operations."""
    
    # ============================================================================
    # Core Crawling Settings
    # ============================================================================
    max_pages: int = 20
    max_images: int = 50
    max_total_images: int = 50
    require_faces: bool = False
    crop_faces: bool = True
    max_depth: int = 1
    max_redirects: int = 3
    
    # ============================================================================
    # HTTP Client Settings
    # ============================================================================
    concurrent_downloads: int = 8
    concurrent_processing: int = 3
    per_host_concurrency: int = 6
    timeout_seconds: int = 30
    connect_timeout: int = 5
    read_timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # HTTP Connection Pooling
    max_keepalive_connections: int = 200
    max_connections: int = 500
    keepalive_expiry: float = 30.0
    
    # ============================================================================
    # Memory Management Settings
    # ============================================================================
    memory_pressure_threshold: float = 0.75
    memory_critical_threshold: float = 0.85
    memory_low_threshold: float = 0.60
    batch_size: int = 32
    gc_frequency: int = 100
    cpu_sample_frequency: int = 20
    
    # ============================================================================
    # Face Detection Settings
    # ============================================================================
    face_threads: int = 4
    face_max_workers: int = 12
    face_min_size: int = 80
    face_min_quality: float = 0.5
    face_margin: float = 0.2
    face_strong_detection_threshold: float = 0.8
    face_detection_scales: List[float] = field(default_factory=lambda: [1.0, 2.0])
    
    # Face Deduplication
    face_dup_dist_image: float = 0.35
    face_dup_dist_album: float = 0.38
    
    # ============================================================================
    # Image Processing Settings
    # ============================================================================
    max_image_bytes: int = 10 * 1024 * 1024  # 10 MiB
    max_content_length: int = 8 * 1024 * 1024  # 8MB
    max_image_pixels: int = 50_000_000
    min_image_size: Tuple[int, int] = (100, 100)  # Minimum width x height in pixels
    
    # Image Enhancement
    image_enhancement_low_res_width: int = 500
    image_enhancement_low_res_height: int = 400
    image_enhancement_contrast: float = 1.15
    image_enhancement_sharpness: float = 1.1
    image_jpeg_quality: int = 95
    
    # Thumbnail Settings
    thumbnail_size: tuple = (150, 150)
    thumbnail_quality: int = 95
    
    # ============================================================================
    # Content Filtering Settings
    # ============================================================================
    allowed_content_types: Set[str] = field(default_factory=lambda: {
        'image/jpeg', 'image/jpg', 'image/png', 
        'image/gif', 'image/webp', 'image/bmp'
    })
    blocked_content_types: Set[str] = field(default_factory=lambda: {'image/svg+xml'})
    
    # Image Selection
    preferred_image_width: int = 640
    similarity_threshold: int = 5
    max_preview_images: int = 3
    
    # ============================================================================
    # Album/Gallery Processing Settings
    # ============================================================================
    album_detection_enabled: bool = True
    album_min_images: int = 3
    album_max_images: int = 1000
    album_face_quality_threshold: float = 0.6
    album_save_all_quality_faces: bool = True
    album_face_deduplication: bool = True
    
    # Video Thumbnail Processing
    video_thumbnail_face_extraction: bool = True
    video_face_quality_threshold: float = 0.5
    
    # Album Metadata
    album_metadata_tracking: bool = True
    album_person_deduplication: bool = True
    
    # ============================================================================
    # JavaScript Rendering Settings
    # ============================================================================
    js_rendering_enabled: bool = True
    js_rendering_timeout: float = 15.0
    js_rendering_wait_time: float = 0.5
    js_rendering_max_concurrent: int = 5
    js_rendering_headless: bool = True
    js_rendering_viewport_width: int = 1280
    js_rendering_viewport_height: int = 720
    
    # JavaScript Detection
    js_detection_enabled: bool = True
    js_detection_keywords: List[str] = field(default_factory=lambda: [
        'react', 'vue', 'angular', 'spa', 'single-page',
        'lazy-load', 'infinite-scroll', 'dynamic-content'
    ])
    js_detection_script_threshold: int = 5
    js_detection_fallback_enabled: bool = True
    
    # JavaScript Performance
    js_rendering_memory_limit: int = 512 * 1024 * 1024  # 512MB
    js_rendering_cpu_limit: int = 80
    js_rendering_cache_ttl: int = 300
    
    # ============================================================================
    # Storage Settings
    # ============================================================================
    storage_bucket: str = "crawled-images"
    storage_region: str = "us-east-1"
    storage_timeout: float = 0.5
    
    # ============================================================================
    # List Crawling Settings
    # ============================================================================
    list_crawl_default_sites_file: str = "sites.txt"
    list_crawl_max_pages_per_site: int = 5
    list_crawl_max_images_per_site: int = 20
    list_crawl_auto_selector_mining: bool = True
    list_crawl_skip_existing_recipes: bool = True
    
    # ============================================================================
    # GPU Settings (Future)
    # ============================================================================
    gpu_enabled: bool = False
    gpu_device_id: int = 0
    gpu_memory_fraction: float = 0.8
    
    # ============================================================================
    # Logging and Debug Settings
    # ============================================================================
    log_truncate_max_length: int = 120
    log_hash_suffix_length: int = 8
    
    # ============================================================================
    # Security Settings
    # ============================================================================
    malicious_schemes: Set[str] = field(default_factory=lambda: {
        'javascript', 'data', 'file', 'ftp'
    })
    suspicious_extensions: Set[str] = field(default_factory=lambda: {
        '.exe', '.scr', '.apk', '.msi', '.bat', '.cmd', '.ps1', '.php', '.cgi', '.bin'
    })
    bait_query_keys: Set[str] = field(default_factory=lambda: {
        'download', 'redirect', 'out', 'go'
    })
    blocked_hosts: Set[str] = field(default_factory=set)
    blocked_tlds: Set[str] = field(default_factory=lambda: {
        '.tk', '.ml', '.ga', '.cf'
    })
    
    # ============================================================================
    # Jitter and Timing Settings
    # ============================================================================
    jitter_range: tuple = (100, 400)  # milliseconds
    
    @classmethod
    def from_env(cls) -> 'CrawlerConfig':
        """Load configuration from environment variables."""
        return cls(
            # Core settings
            max_pages=int(os.getenv('CRAWLER_MAX_PAGES', 20)),
            max_images=int(os.getenv('CRAWLER_MAX_IMAGES', 50)),
            max_total_images=int(os.getenv('CRAWLER_MAX_TOTAL_IMAGES', 50)),
            require_faces=os.getenv('CRAWLER_REQUIRE_FACES', 'false').lower() == 'true',
            crop_faces=os.getenv('CRAWLER_CROP_FACES', 'true').lower() == 'true',
            max_depth=int(os.getenv('CRAWLER_MAX_DEPTH', 1)),
            max_redirects=int(os.getenv('CRAWLER_MAX_REDIRECTS', 3)),
            
            # HTTP settings
            concurrent_downloads=int(os.getenv('CRAWLER_CONCURRENT_DOWNLOADS', 8)),
            concurrent_processing=int(os.getenv('CRAWLER_CONCURRENT_PROCESSING', 3)),
            per_host_concurrency=int(os.getenv('CRAWLER_PER_HOST_CONCURRENCY', 6)),
            timeout_seconds=int(os.getenv('CRAWLER_TIMEOUT', 30)),
            connect_timeout=int(os.getenv('CRAWLER_CONNECT_TIMEOUT', 5)),
            read_timeout=int(os.getenv('CRAWLER_READ_TIMEOUT', 10)),
            max_retries=int(os.getenv('CRAWLER_MAX_RETRIES', 3)),
            retry_delay=float(os.getenv('CRAWLER_RETRY_DELAY', 1.0)),
            
            # Memory management
            memory_pressure_threshold=float(os.getenv('CRAWLER_MEMORY_THRESHOLD', 0.75)),
            memory_critical_threshold=float(os.getenv('CRAWLER_MEMORY_CRITICAL', 0.85)),
            memory_low_threshold=float(os.getenv('CRAWLER_MEMORY_LOW', 0.60)),
            batch_size=int(os.getenv('CRAWLER_BATCH_SIZE', 32)),
            gc_frequency=int(os.getenv('CRAWLER_GC_FREQUENCY', 100)),
            
            # Face detection
            face_threads=int(os.getenv('CRAWLER_FACE_THREADS', 4)),
            face_max_workers=int(os.getenv('CRAWLER_FACE_MAX_WORKERS', 12)),
            face_min_size=int(os.getenv('CRAWLER_FACE_MIN_SIZE', 80)),
            face_min_quality=float(os.getenv('CRAWLER_FACE_MIN_QUALITY', 0.5)),
            face_margin=float(os.getenv('CRAWLER_FACE_MARGIN', 0.2)),
            face_strong_detection_threshold=float(os.getenv('CRAWLER_FACE_STRONG_THRESHOLD', 0.8)),
            
            # Face deduplication
            face_dup_dist_image=float(os.getenv('CRAWLER_FACE_DUP_DIST_IMAGE', 0.35)),
            face_dup_dist_album=float(os.getenv('CRAWLER_FACE_DUP_DIST_ALBUM', 0.38)),
            
            # Image processing
            max_image_bytes=int(os.getenv('CRAWLER_MAX_IMAGE_BYTES', 10 * 1024 * 1024)),
            max_content_length=int(os.getenv('CRAWLER_MAX_CONTENT_LENGTH', 8 * 1024 * 1024)),
            max_image_pixels=int(os.getenv('CRAWLER_MAX_IMAGE_PIXELS', 50_000_000)),
            min_image_size=(
                int(os.getenv('CRAWLER_MIN_IMAGE_WIDTH', 100)),
                int(os.getenv('CRAWLER_MIN_IMAGE_HEIGHT', 100))
            ),
            
            # Image enhancement
            image_enhancement_low_res_width=int(os.getenv('CRAWLER_ENHANCE_LOW_RES_WIDTH', 500)),
            image_enhancement_low_res_height=int(os.getenv('CRAWLER_ENHANCE_LOW_RES_HEIGHT', 400)),
            image_enhancement_contrast=float(os.getenv('CRAWLER_ENHANCE_CONTRAST', 1.15)),
            image_enhancement_sharpness=float(os.getenv('CRAWLER_ENHANCE_SHARPNESS', 1.1)),
            image_jpeg_quality=int(os.getenv('CRAWLER_JPEG_QUALITY', 95)),
            
            # Thumbnail settings
            thumbnail_quality=int(os.getenv('CRAWLER_THUMBNAIL_QUALITY', 95)),
            
            # Album processing
            album_detection_enabled=os.getenv('CRAWLER_ALBUM_DETECTION', 'true').lower() == 'true',
            album_min_images=int(os.getenv('CRAWLER_ALBUM_MIN_IMAGES', 3)),
            album_max_images=int(os.getenv('CRAWLER_ALBUM_MAX_IMAGES', 1000)),
            album_face_quality_threshold=float(os.getenv('CRAWLER_ALBUM_FACE_QUALITY', 0.6)),
            album_save_all_quality_faces=os.getenv('CRAWLER_ALBUM_SAVE_ALL_FACES', 'true').lower() == 'true',
            album_face_deduplication=os.getenv('CRAWLER_ALBUM_FACE_DEDUP', 'true').lower() == 'true',
            
            # Video processing
            video_thumbnail_face_extraction=os.getenv('CRAWLER_VIDEO_FACE_EXTRACTION', 'true').lower() == 'true',
            video_face_quality_threshold=float(os.getenv('CRAWLER_VIDEO_FACE_QUALITY', 0.5)),
            
            # JavaScript rendering
            js_rendering_enabled=os.getenv('CRAWLER_JS_RENDERING', 'true').lower() == 'true',
            js_rendering_timeout=float(os.getenv('CRAWLER_JS_TIMEOUT', 15.0)),
            js_rendering_wait_time=float(os.getenv('CRAWLER_JS_WAIT_TIME', 0.5)),
            js_rendering_max_concurrent=int(os.getenv('CRAWLER_JS_MAX_CONCURRENT', 5)),
            js_rendering_headless=os.getenv('CRAWLER_JS_HEADLESS', 'true').lower() == 'true',
            js_rendering_viewport_width=int(os.getenv('CRAWLER_JS_VIEWPORT_WIDTH', 1280)),
            js_rendering_viewport_height=int(os.getenv('CRAWLER_JS_VIEWPORT_HEIGHT', 720)),
            
            # JavaScript detection
            js_detection_enabled=os.getenv('CRAWLER_JS_DETECTION', 'true').lower() == 'true',
            js_detection_script_threshold=int(os.getenv('CRAWLER_JS_SCRIPT_THRESHOLD', 5)),
            js_detection_fallback_enabled=os.getenv('CRAWLER_JS_FALLBACK', 'true').lower() == 'true',
            
            # JavaScript performance
            js_rendering_memory_limit=int(os.getenv('CRAWLER_JS_MEMORY_LIMIT', 512 * 1024 * 1024)),
            js_rendering_cpu_limit=int(os.getenv('CRAWLER_JS_CPU_LIMIT', 80)),
            js_rendering_cache_ttl=int(os.getenv('CRAWLER_JS_CACHE_TTL', 300)),
            
            # Storage settings
            storage_bucket=os.getenv('CRAWLER_STORAGE_BUCKET', 'crawled-images'),
            storage_region=os.getenv('CRAWLER_STORAGE_REGION', 'us-east-1'),
            storage_timeout=float(os.getenv('CRAWLER_STORAGE_TIMEOUT', 0.5)),
            
            # List crawling
            list_crawl_default_sites_file=os.getenv('CRAWLER_SITES_FILE', 'sites.txt'),
            list_crawl_max_pages_per_site=int(os.getenv('CRAWLER_LIST_MAX_PAGES', 5)),
            list_crawl_max_images_per_site=int(os.getenv('CRAWLER_LIST_MAX_IMAGES', 20)),
            list_crawl_auto_selector_mining=os.getenv('CRAWLER_AUTO_MINING', 'true').lower() == 'true',
            list_crawl_skip_existing_recipes=os.getenv('CRAWLER_SKIP_EXISTING_RECIPES', 'true').lower() == 'true',
            
            # GPU settings
            gpu_enabled=os.getenv('CRAWLER_GPU_ENABLED', 'false').lower() == 'true',
            gpu_device_id=int(os.getenv('CRAWLER_GPU_DEVICE', 0)),
            gpu_memory_fraction=float(os.getenv('CRAWLER_GPU_MEMORY_FRACTION', 0.8)),
            
            # Logging
            log_truncate_max_length=int(os.getenv('CRAWLER_LOG_TRUNCATE_LENGTH', 120)),
            log_hash_suffix_length=int(os.getenv('CRAWLER_LOG_HASH_SUFFIX_LENGTH', 8)),
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        if self.max_pages <= 0:
            raise ValueError("max_pages must be positive")
        if self.max_images <= 0:
            raise ValueError("max_images must be positive")
        if self.concurrent_downloads <= 0:
            raise ValueError("concurrent_downloads must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if not 0.0 <= self.memory_pressure_threshold <= 1.0:
            raise ValueError("memory_pressure_threshold must be between 0.0 and 1.0")
        if not 0.0 <= self.face_min_quality <= 1.0:
            raise ValueError("face_min_quality must be between 0.0 and 1.0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.face_threads <= 0:
            raise ValueError("face_threads must be positive")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for key, value in updates.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def get_http_limits(self) -> Dict[str, Any]:
        """Get HTTP client limits configuration."""
        return {
            'max_connections': self.max_connections,
            'max_keepalive_connections': self.max_keepalive_connections,
            'keepalive_expiry': self.keepalive_expiry,
        }
    
    def get_face_detection_config(self) -> Dict[str, Any]:
        """Get face detection configuration."""
        return {
            'threads': self.face_threads,
            'max_workers': self.face_max_workers,
            'min_size': self.face_min_size,
            'min_quality': self.face_min_quality,
            'margin': self.face_margin,
            'strong_detection_threshold': self.face_strong_detection_threshold,
            'detection_scales': self.face_detection_scales,
            'dup_dist_image': self.face_dup_dist_image,
            'dup_dist_album': self.face_dup_dist_album,
        }
    
    def get_memory_config(self) -> Dict[str, Any]:
        """Get memory management configuration."""
        return {
            'pressure_threshold': self.memory_pressure_threshold,
            'critical_threshold': self.memory_critical_threshold,
            'low_threshold': self.memory_low_threshold,
            'batch_size': self.batch_size,
            'gc_frequency': self.gc_frequency,
            'cpu_sample_frequency': self.cpu_sample_frequency,
        }
    
    def get_storage_config(self) -> Dict[str, Any]:
        """Get storage configuration."""
        return {
            'bucket': self.storage_bucket,
            'region': self.storage_region,
            'timeout': self.storage_timeout,
        }
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU configuration."""
        return {
            'enabled': self.gpu_enabled,
            'device_id': self.gpu_device_id,
            'memory_fraction': self.gpu_memory_fraction,
        }


# Global configuration instance
_config: Optional[CrawlerConfig] = None


def get_config() -> CrawlerConfig:
    """Get global configuration instance."""
    global _config
    if _config is None:
        _config = CrawlerConfig.from_env()
        _config.validate()
    return _config


def set_config(config: CrawlerConfig) -> None:
    """Set global configuration instance."""
    global _config
    config.validate()
    _config = config


def reset_config() -> None:
    """Reset global configuration instance."""
    global _config
    _config = None
