"""
Image Crawler Service

A comprehensive web crawler service for extracting and processing images from websites.
Features include intelligent image detection, face recognition, caching, and multi-tenant support.
"""

import asyncio
import hashlib
import logging
import os
import sys
import concurrent.futures
import psutil
import gc
import weakref
import re
import random
import time
import json
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass, field
from .crawler_modules.types import CrawlResult, ImageInfo
from .crawler_modules.resources import ResourceMonitor
from .adaptive_resource_manager import get_adaptive_resource_manager
from .pipeline_orchestrator import get_pipeline_orchestrator
from .crawler_modules.extraction import ImageExtractor, extract_style_bg_url, extract_jsonld_thumbnails, pick_from_srcset
from .crawler_modules.processing import ImageProcessingService
from .crawler_modules.storage_facade import StorageFacade
from .crawler_modules.caching_facade import CachingFacade
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

# Import HTTP service for unified HTTP handling
from .http_service import HttpService, fetch_html_with_redirects
# Import selector mining service for 3x3 depth approach
from .selector_mining import SelectorMiningService

# Import site recipes functionality
from ..config.site_recipes import get_recipe_for_url, get_recipe_for_host

# Image safety configuration - set once on import
from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = 50_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

from .storage import get_storage_cleanup_function
from . import storage
from .face import get_face_service, close_face_service
from . import face
# Cache service is now handled by CachingFacade
from urllib.parse import urljoin, urlparse

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

# URL Security Configuration
MALICIOUS_SCHEMES = {'javascript', 'data', 'file', 'ftp'}
SUSPICIOUS_EXTENSIONS = {'.exe', '.scr', '.apk', '.msi', '.bat', '.cmd', '.ps1', '.php', '.cgi', '.bin'}
BAIT_QUERY_KEYS = {'download', 'redirect', 'out', 'go'}

# Host/TLD Denylist (placeholders - can be expanded)
BLOCKED_HOSTS = {
    'malware.example.com',
    'phishing-site.net',
    'suspicious-domain.org'
}

BLOCKED_TLDS = {
    '.tk', '.ml', '.ga', '.cf'  # Free TLDs often used for malicious purposes
}

# Content Security Configuration
ALLOWED_CONTENT_TYPES = {'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'}
BLOCKED_CONTENT_TYPES = {'image/svg+xml'}  # SVG can contain malicious scripts
MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB

# Crawl Policy Configuration
DEFAULT_MAX_DEPTH = 1
DEFAULT_PER_HOST_CONCURRENCY = 3
DEFAULT_JITTER_RANGE = (100, 400)  # milliseconds

# ============================================================================
# SECURITY GUARDS
# ============================================================================

def validate_url_security(url: str) -> Tuple[bool, str]:
    """
    Validate URL for security threats.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_safe, reason_code)
    """
    try:
        parsed = urlparse(url)
        
        # Check for malicious schemes
        if parsed.scheme.lower() in MALICIOUS_SCHEMES:
            return False, "MALICIOUS_SCHEME"
        
        # Check for suspicious file extensions
        path_lower = parsed.path.lower()
        for ext in SUSPICIOUS_EXTENSIONS:
            if path_lower.endswith(ext):
                return False, "SUSPICIOUS_EXTENSION"
        
        # Check for bait query parameters
        query_params = set(parse_qs(parsed.query).keys())
        if query_params.intersection(BAIT_QUERY_KEYS):
            return False, "BAIT_QUERY_PARAM"
        
        # Check blocked hosts
        if parsed.netloc.lower() in BLOCKED_HOSTS:
            return False, "BLOCKED_HOST"
        
        # Check blocked TLDs
        for tld in BLOCKED_TLDS:
            if parsed.netloc.lower().endswith(tld):
                return False, "BLOCKED_TLD"
        
        return True, "SAFE"
        
    except Exception as e:
        logger.warning(f"URL validation error for {url}: {e}")
        return False, "VALIDATION_ERROR"

def validate_content_security(content_type: str, content_length: Optional[int] = None) -> Tuple[bool, str]:
    """
    Validate content type and size for security.
    
    Args:
        content_type: MIME type of the content
        content_length: Size of the content in bytes
        
    Returns:
        Tuple of (is_safe, reason_code)
    """
    if not content_type:
        return False, "NO_CONTENT_TYPE"
    
    content_type_lower = content_type.lower()
    
    # Explicitly reject SVG
    if 'image/svg+xml' in content_type_lower:
        return False, "SKIP_SVG"
    
    # Check for blocked content types
    if content_type_lower in BLOCKED_CONTENT_TYPES:
        return False, "BLOCKED_CONTENT_TYPE"
    
    # Check if content type is allowed
    if not any(allowed_type in content_type_lower for allowed_type in ALLOWED_CONTENT_TYPES):
        return False, "NOT_IMAGE_CONTENT"
    
    # Check content length if provided
    if content_length is not None and content_length > MAX_CONTENT_LENGTH:
        return False, "CONTENT_TOO_LARGE"
    
    return True, "SAFE"

def validate_redirect_security(redirect_url: str, redirect_count: int, max_redirects: int = 3) -> Tuple[bool, str]:
    """
    Validate redirect URL for security and limits.
    
    Args:
        redirect_url: URL being redirected to
        redirect_count: Current redirect count
        max_redirects: Maximum allowed redirects
        
    Returns:
        Tuple of (is_safe, reason_code)
    """
    # Check redirect limit
    if redirect_count >= max_redirects:
        return False, "TOO_MANY_REDIRECTS"
    
    # Validate the redirect URL
    is_safe, reason = validate_url_security(redirect_url)
    if not is_safe:
        return False, f"REDIRECT_{reason}"
    
    return True, "SAFE"

# ============================================================================
# THUMBNAIL EXTRACTION HELPERS
# ============================================================================


# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

# MemoryMonitor moved to .crawler_modules.memory

logger = logging.getLogger(__name__)

# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def _truncate_log_string(text: str, max_length: int = 120) -> str:
    """
    Truncate long strings for logging with a hash suffix for identification.
    
    Args:
        text: String to truncate
        max_length: Maximum length before truncation
        
    Returns:
        Truncated string with hash suffix if truncated
    """
    if len(text) <= max_length:
        return text
    
    # Create a short hash of the original string for identification
    hash_suffix = hashlib.md5(text.encode()).hexdigest()[:8]
    truncated = text[:max_length - len(hash_suffix) - 3]  # Reserve space for "..." and hash
    return f"{truncated}...{hash_suffix}"

# ============================================================================
# DATA CLASSES
# ============================================================================

# CrawlResult moved to .crawler.types


# ImageInfo moved to .crawler.types


# ============================================================================
# MAIN CRAWLER CLASS
# ============================================================================

class ImageCrawler:
    """
    Image crawler service with intelligent detection, face recognition, and caching capabilities.
    Supports multi-tenant operations and concurrent processing.
    """
    
    def __init__(
        self,
        tenant_id: str = "default",
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        allowed_extensions: Optional[Set[str]] = None,
        timeout: int = 30,
        min_face_quality: float = 0.5,
        require_face: bool = True,
        crop_faces: bool = True,
        face_margin: float = 0.2,
        max_total_images: int = 50,
        max_pages: int = 20,
        same_domain_only: bool = True,
        similarity_threshold: int = 5,
        max_concurrent_images: int = 20,  # Maximum concurrent image processing
        batch_size: int = 50,
        enable_audit_logging: bool = True,
        use_3x3_mining: bool = False,  # Enable 3x3 depth mining for better selector discovery
    ):
        self.tenant_id = tenant_id  # Multi-tenancy support
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or {
            '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp'
        }
        self.timeout = timeout
        self.min_face_quality = min_face_quality  # Minimum detection score for face quality
        self.require_face = require_face  # Whether to require at least one face
        self.crop_faces = crop_faces  # Whether to crop and save only face regions
        self.face_margin = face_margin  # Margin around face as fraction of face size
        self.max_total_images = max_total_images  # Maximum total images to collect
        self.max_pages = max_pages  # Maximum pages to crawl
        self.same_domain_only = same_domain_only  # Only crawl same domain
        self.similarity_threshold = similarity_threshold  # Hamming distance threshold for content similarity
        self.max_concurrent_images = max_concurrent_images
        self.batch_size = batch_size
        self.enable_audit_logging = enable_audit_logging
        self.use_3x3_mining = use_3x3_mining  # Audit logging support
        self._early_exit_count = 0
        
        # Initialize extraction service
        self._extractor = ImageExtractor()
        
        # Security and crawl policy settings
        self.max_depth = DEFAULT_MAX_DEPTH
        self.per_host_concurrency = DEFAULT_PER_HOST_CONCURRENCY
        self.jitter_range = DEFAULT_JITTER_RANGE
        self.respect_robots_txt = False  # Optional robots.txt respect flag
        self.max_redirects = 3
        self._redirect_counts = {}  # Track redirect counts per URL
        
        # Dynamic concurrency control with adaptive resource management
        self._processing_semaphore = asyncio.Semaphore(self.max_concurrent_images)
        self._storage_semaphore = asyncio.Semaphore(self.max_concurrent_images)
        self._download_semaphore = asyncio.Semaphore(min(self.max_concurrent_images * 2, 50))  # Allow more downloads
        self._per_host_semaphores = {}  # Per-host concurrency control
        self._per_host_limits = {}  # Track per-host limits
        
        # Dynamic concurrency tracking
        self._last_concurrency_update = time.time()
        self._concurrency_update_interval = 2.0  # Update every 2 seconds
        
        # Thread pool for CPU-intensive face detection
        self._face_detection_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(psutil.cpu_count() or 4, 8),  # Limit to CPU cores but cap at 8
            thread_name_prefix="face_detection"
        )
        
        # Thread pool for storage operations
        self._storage_thread_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=min(self.max_concurrent_images, 16),  # Match concurrency limit
            thread_name_prefix="storage"
        )
        
        # Dynamic resource management
        self._resource_monitor = ResourceMonitor()
        self._adaptive_manager = get_adaptive_resource_manager()
        self._pipeline_orchestrator = get_pipeline_orchestrator()
        
        # Initialize resource monitoring
        if self._adaptive_manager.settings.enable_dynamic_resources:
            self._resource_monitor.start_monitoring(
                interval_ms=self._adaptive_manager.settings.resource_monitor_interval_ms
            )
        
        # Legacy memory monitoring (for backward compatibility)
        self._memory_monitor = self._resource_monitor  # Alias for compatibility
        self._active_tasks = weakref.WeakSet()  # Track active tasks for memory cleanup
        self._memory_pressure_threshold = 75  # Memory pressure threshold
        self._gc_frequency = 10  # Force GC every N operations
        self._operation_count = 0
        self._cpu_sample_counter = 0  # Counter for CPU sampling frequency
        self._jitter_applied = False  # Track if jitter has been applied
        
        self.session: Optional[httpx.AsyncClient] = None
        self._http_service: Optional[HttpService] = None
        self._selector_mining_service: Optional[SelectorMiningService] = None
        # Cache service is now handled by CachingFacade
        self.pending_cache_entries = []  # Batch cache writes
    
    def _update_dynamic_concurrency(self):
        """Update concurrency limits based on resource utilization."""
        if not self._adaptive_manager.settings.enable_dynamic_resources:
            return
        
        current_time = time.time()
        if current_time - self._last_concurrency_update < self._concurrency_update_interval:
            return
        
        try:
            # Get optimal concurrency from adaptive manager
            optimal_processing = self._adaptive_manager.get_optimal_concurrency(
                'processing', self.max_concurrent_images
            )
            optimal_downloads = self._adaptive_manager.get_optimal_concurrency(
                'downloads', min(self.max_concurrent_images * 2, 50)
            )
            
            # Update semaphores if values changed
            if optimal_processing != self.max_concurrent_images:
                old_processing = self.max_concurrent_images
                self.max_concurrent_images = optimal_processing
                self._processing_semaphore = asyncio.Semaphore(optimal_processing)
                self._storage_semaphore = asyncio.Semaphore(optimal_processing)
                
                logger.debug(f"Dynamic concurrency update: processing {old_processing} → {optimal_processing}")
            
            if optimal_downloads != self._download_semaphore._value:
                self._download_semaphore = asyncio.Semaphore(optimal_downloads)
                logger.debug(f"Dynamic concurrency update: downloads → {optimal_downloads}")
            
            self._last_concurrency_update = current_time
            
        except Exception as e:
            logger.warning(f"Error updating dynamic concurrency: {e}")
        
    async def __aenter__(self):
        """Async context manager entry."""
        # Create HTTP service for unified HTTP handling
        self._http_service = HttpService()
        
        # Create selector mining service for 3x3 depth approach
        self._selector_mining_service = SelectorMiningService()
        
        # Create image processing service for face detection and processing
        self._image_processing_service = ImageProcessingService(
            min_face_quality=self.min_face_quality,
            require_face=self.require_face,
            crop_faces=self.crop_faces,
            face_margin=self.face_margin,
            min_dimension=100
        )
        
        # Create storage facade for clean storage operations
        self._storage_facade = StorageFacade()
        
        # Create caching facade for clean cache operations
        self._caching_facade = CachingFacade()
        
        # Create image extractor for validation in smart fallback
        from .crawler_modules.extraction import ImageExtractor
        self._image_extractor = ImageExtractor()
        
        # Create HTTP client using the service
        self.session = await self._http_service.create_client(
            limits=httpx.Limits(max_keepalive_connections=100, max_connections=200),
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=30.0,    # Read timeout
                write=10.0,   # Write timeout
                pool=5.0      # Pool timeout
            ),
            verify=True,  # TLS verification
            http2=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with comprehensive cleanup."""
        logger.info("Starting crawler cleanup...")
        
        try:
            # Close HTTP session
            if self.session:
                logger.info("Closing HTTP session...")
                await self.session.aclose()
                logger.info("HTTP session closed")
        except Exception as e:
            logger.warning(f"Error closing HTTP session: {e}")
        
        try:
            # Close HTTP service
            if self._http_service:
                logger.info("Closing HTTP service...")
                await self._http_service.close()
                logger.info("HTTP service closed")
        except Exception as e:
            logger.warning(f"Error closing HTTP service: {e}")
        
        try:
            # Clean up crawler thread pools
            logger.info("Shutting down crawler thread pools...")
            self._face_detection_thread_pool.shutdown(wait=True)
            self._storage_thread_pool.shutdown(wait=True)
            logger.info("Crawler thread pools shutdown complete")
        except Exception as e:
            logger.warning(f"Error shutting down crawler thread pools: {e}")
        
        try:
            # Clean up image processing service
            if hasattr(self, '_image_processing_service'):
                self._image_processing_service.cleanup()
        except Exception as e:
            logger.warning(f"Error cleaning up image processing service: {e}")
        
        try:
            # Clean up service resources
            logger.info("Cleaning up service resources...")
            close_storage_resources = get_storage_cleanup_function()
            close_storage_resources()
            logger.info("Storage service resources cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up storage resources: {e}")
        
        try:
            # Clean up face service resources
            logger.info("Cleaning up face service resources...")
            close_face_service()
            logger.info("Face service resources cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up face service resources: {e}")
        
        try:
            # Clean up caching facade resources
            if hasattr(self, '_caching_facade'):
                await self._caching_facade.close()
        except Exception as e:
            logger.warning(f"Error cleaning up caching facade resources: {e}")
        
        try:
            # Final memory cleanup
            logger.info("Performing final memory cleanup...")
            gc.collect()
            logger.info("Crawler cleanup complete")
        except Exception as e:
            logger.warning(f"Error during final memory cleanup: {e}")
    
    # ============================================================================
    # MEMORY MANAGEMENT METHODS
    # ============================================================================
    
    def _adjust_concurrency_dynamically(self):
        """
        Adjust concurrency based on memory pressure and system resources.
        """
        try:
            # Get comprehensive system metrics - only sample CPU every ~20 operations to reduce overhead
            self._cpu_sample_counter += 1
            if self._cpu_sample_counter % 20 == 0:
                cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking call
            else:
                cpu_percent = 50.0  # Default moderate CPU usage
            memory_status = self._memory_monitor.get_memory_status()
            
            # Trigger garbage collection if needed
            if self._memory_monitor.should_trigger_gc():
                logger.info(f"Triggering garbage collection due to memory pressure: {memory_status['pressure_level']}")
                gc.collect()
            
            # Calculate base concurrency from CPU and memory
            cpu_factor = 1.0
            if cpu_percent > 80:
                cpu_factor = 0.5  # Reduce by half if CPU is high
            elif cpu_percent < 30:
                cpu_factor = 1.5  # Increase by 50% if CPU is low
            
            # Get memory-safe concurrency limit
            base_concurrency = int(self.max_concurrent_images * cpu_factor)
            new_concurrency = self._memory_monitor.get_safe_concurrency_limit(base_concurrency)
            
            # Ensure minimum concurrency for progress
            new_concurrency = max(1, new_concurrency)
            
            # Update semaphores if concurrency changed
            if new_concurrency != self.max_concurrent_images:
                old_concurrency = self.max_concurrent_images
                self.max_concurrent_images = new_concurrency
                self._processing_semaphore = asyncio.Semaphore(self.max_concurrent_images)
                self._storage_semaphore = asyncio.Semaphore(self.max_concurrent_images)
                self._download_semaphore = asyncio.Semaphore(min(self.max_concurrent_images * 2, 50))
                
                logger.info(f"Dynamic concurrency adjustment: {old_concurrency} → {new_concurrency} "
                           f"(CPU: {cpu_percent:.1f}%, Memory: {memory_status['pressure_level']}, "
                           f"Available: {memory_status['available_gb']:.1f}GB)")
                
        except Exception as e:
            logger.warning(f"Failed to adjust concurrency dynamically: {e}")
    
    def _manage_memory_pressure(self):
        """Proactive memory management during operations."""
        self._operation_count += 1
        
        # Check for memory pressure and adjust if needed
        memory_status = self._memory_monitor.get_memory_status()
        if memory_status['pressure_level'] in ['high', 'critical']:
            # Force cleanup of completed tasks
            self._cleanup_completed_tasks()
            
            # Trigger immediate GC only under memory pressure
            gc.collect()
            
            logger.debug(f"Memory pressure management: {memory_status['pressure_level']} "
                        f"({memory_status['percent']:.1f}% used)")
    
    def _cleanup_completed_tasks(self):
        """Clean up references to completed tasks."""
        # The WeakSet automatically removes completed tasks
        # Track active tasks for logging and monitoring
        active_count = len(self._active_tasks)
        if active_count > 0:
            logger.debug(f"Active tasks: {active_count}")
    
    async def _async_storage_operation(self, storage_func, *args, **kwargs):
        """
        Run storage operations with memory management.
        """
        # Track this operation for memory management
        task = asyncio.current_task()
        if task:
            self._active_tasks.add(task)
        
        try:
            # Check memory before storage operation
            self._manage_memory_pressure()
            
            # Run storage operation in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._storage_thread_pool, storage_func, *args, **kwargs
            )
            
            # Clean up memory after operation
            del args, kwargs  # Free memory
            # Trigger GC after memory-intensive storage operations
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Storage operation failed: {e}")
            raise
        finally:
            # Remove task from tracking when complete
            if task and task in self._active_tasks:
                self._active_tasks.discard(task)
    
    # ============================================================================
    # IMAGE PROCESSING METHODS
    # ============================================================================
    
    async def _process_images_batch(self, images: List[ImageInfo], all_errors: List[str], cache_hits: int, cache_misses: int, method_used: str, url: str) -> 'CrawlResult':
        """Process images using batch approach for small workloads."""
        saved_raw_keys = []
        saved_thumbnail_keys = []
        
        processing_tasks = []
        for i, image_info in enumerate(images, 1):
            processing_tasks.append(self._process_single_image(image_info, i, len(images)))

        processed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

        # Process results and store to MinIO
        for i, result in enumerate(processed_results):
            image_info = images[i]
            if isinstance(result, Exception):
                logger.error(f"Error processing image {image_info.url}: {result}")
                all_errors.append(f"Processing error for {image_info.url}: {result}")
                continue

            image_bytes, thumbnail_bytes, was_cached, download_errors, faces = result
            all_errors.extend(download_errors)

            if was_cached:
                cache_hits += 1
                continue
            else:
                cache_misses += 1

                if image_bytes:
                    # Debug logging for video URL
                    logger.info(f"Processing image with video URL: {image_info.video_url}")
                    
                    # Use storage facade to save image and thumbnail
                    raw_key, thumb_key = await self._storage_facade.save_raw_and_thumbnail(
                        image_bytes=image_bytes,
                        thumbnail_bytes=thumbnail_bytes,
                        image_info=image_info,
                        page_url=url
                    )
                    
                    # Add keys to results
                    if raw_key:
                        saved_raw_keys.append(raw_key)
                    if thumb_key:
                        saved_thumbnail_keys.append(thumb_key)
        
        # Get detailed cache statistics
        cache_stats = await self._caching_facade.get_cache_statistics()
        
        result = CrawlResult(
            url=url,
            images_found=len(images),
            raw_images_saved=len(saved_raw_keys),
            thumbnails_saved=len(saved_thumbnail_keys),
            pages_crawled=1,
            saved_raw_keys=saved_raw_keys,
            saved_thumbnail_keys=saved_thumbnail_keys,
            errors=all_errors,
            targeting_method=method_used,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            redis_hits=cache_stats['redis_hits'],
            postgres_hits=cache_stats['postgres_hits'],
            tenant_id=self.tenant_id,
            early_exit_count=int(getattr(self, "_early_exit_count", 0))
        )
        
        logger.info(f"Crawl completed - Found: {result.images_found}, Raw Images Saved: {result.raw_images_saved}, Thumbnails Saved: {result.thumbnails_saved}")
        return result
    
    async def _process_images_streaming(self, images: List[ImageInfo], all_errors: List[str], cache_hits: int, cache_misses: int, method_used: str, url: str) -> 'CrawlResult':
        """Process images using memory-efficient streaming pipeline for large workloads."""
        # Memory-efficient streaming pipeline with bounded queues
        download_queue = asyncio.Queue(maxsize=20)      # Small queue for backpressure
        processing_queue = asyncio.Queue(maxsize=10)    # Even smaller processing queue
        storage_queue = asyncio.Queue(maxsize=10)       # Small storage queue
        
        # Results tracking
        streaming_results = {
            'saved_raw_keys': [],
            'saved_thumbnail_keys': [],
            'errors': all_errors.copy(),
            'cache_hits': cache_hits,
            'cache_misses': cache_misses
        }
        
        # Create workers (only 3 workers total to minimize overhead)
        workers = [
            asyncio.create_task(self._streaming_download_worker(download_queue, processing_queue, streaming_results)),
            asyncio.create_task(self._streaming_processing_worker(processing_queue, storage_queue, streaming_results)),
            asyncio.create_task(self._streaming_storage_worker(storage_queue, streaming_results, url))
        ]
        
        # Feed images into download queue
        for i, image_info in enumerate(images, 1):
            await download_queue.put((image_info, i, len(images)))
        
        # Signal completion to all queues
        await download_queue.put(None)  # Sentinel value
        
        # Wait for all workers to complete
        await asyncio.gather(*workers, return_exceptions=True)
        
        # Get detailed cache statistics
        cache_stats = await self._caching_facade.get_cache_statistics()
        
        result = CrawlResult(
            url=url,
            images_found=len(images),
            raw_images_saved=len(streaming_results['saved_raw_keys']),
            thumbnails_saved=len(streaming_results['saved_thumbnail_keys']),
            pages_crawled=1,
            saved_raw_keys=streaming_results['saved_raw_keys'],
            saved_thumbnail_keys=streaming_results['saved_thumbnail_keys'],
            errors=streaming_results['errors'],
            targeting_method=method_used,
            cache_hits=streaming_results['cache_hits'],
            cache_misses=streaming_results['cache_misses'],
            redis_hits=cache_stats['redis_hits'],
            postgres_hits=cache_stats['postgres_hits'],
            tenant_id=self.tenant_id,
            early_exit_count=int(getattr(self, "_early_exit_count", 0))
        )
        
        logger.info(f"Crawl completed - Found: {result.images_found}, Raw Images Saved: {result.raw_images_saved}, Thumbnails Saved: {result.thumbnails_saved}")
        return result
    
    async def _streaming_download_worker(self, download_queue: asyncio.Queue, processing_queue: asyncio.Queue, results: dict):
        """Memory-efficient download worker."""
        while True:
            item = await download_queue.get()
            if item is None:  # Sentinel value
                await processing_queue.put(None)  # Pass sentinel to next stage
                break
            
            image_info, index, total = item
            try:
                # Download image (this includes cache check)
                image_bytes, thumbnail_bytes, was_cached, download_errors, faces = await self._process_single_image(image_info, index, total)
                results['errors'].extend(download_errors)
                
                if was_cached:
                    results['cache_hits'] += 1
                else:
                    results['cache_misses'] += 1
                    if image_bytes:
                        await processing_queue.put((image_bytes, thumbnail_bytes, image_info, faces))
                        
            except Exception as e:
                logger.error(f"Download worker error for {_truncate_log_string(image_info.url)}: {e}")
                results['errors'].append(f"Download error for {_truncate_log_string(image_info.url)}: {e}")
            finally:
                download_queue.task_done()
    
    async def _streaming_processing_worker(self, processing_queue: asyncio.Queue, storage_queue: asyncio.Queue, results: dict):
        """Memory-efficient processing worker for face detection."""
        while True:
            item = await processing_queue.get()
            if item is None:  # Sentinel value
                await storage_queue.put(None)  # Pass sentinel to next stage
                break
            
            image_bytes, thumbnail_bytes, image_info, faces = item
            try:
                # Face detection if needed (reuse existing logic)
                if self.require_face or self.crop_faces:
                    if not faces:  # Only detect if not already detected
                        faces, thumbnail_bytes = await self._async_face_detection(image_bytes, image_info)
                    
                    if self.require_face and not faces:
                        logger.info(f"No faces detected for {_truncate_log_string(image_info.url)}, skipping.")
                        continue
                    
                    if self.crop_faces and not faces:
                        logger.info(f"No faces detected for {_truncate_log_string(image_info.url)}, no thumbnail created.")
                        thumbnail_bytes = None
                
                # Queue for storage
                await storage_queue.put((image_bytes, thumbnail_bytes, image_info))
                        
            except Exception as e:
                logger.error(f"Processing worker error for {_truncate_log_string(image_info.url)}: {e}")
                results['errors'].append(f"Processing error for {_truncate_log_string(image_info.url)}: {e}")
            finally:
                processing_queue.task_done()
    
    async def _streaming_storage_worker(self, storage_queue: asyncio.Queue, results: dict, url: str):
        """Memory-efficient storage worker with micro-batching."""
        storage_batch = []
        batch_size = 5  # Small batch size for memory efficiency
        
        while True:
            try:
                # Wait for item with timeout for micro-batching
                item = await asyncio.wait_for(storage_queue.get(), timeout=0.5)
                
                if item is None:  # Sentinel value
                    # Process any remaining items
                    if storage_batch:
                        await self._process_storage_batch(storage_batch, results, url)
                    break
                
                storage_batch.append(item)
                
                # Process batch when full
                if len(storage_batch) >= batch_size:
                    await self._process_storage_batch(storage_batch, results, url)
                    storage_batch = []
                    
            except asyncio.TimeoutError:
                # Process any pending items on timeout
                if storage_batch:
                    await self._process_storage_batch(storage_batch, results, url)
                    storage_batch = []
                continue
            except Exception as e:
                logger.error(f"Storage worker error: {e}")
                results['errors'].append(f"Storage worker error: {e}")
            finally:
                if 'item' in locals() and item is not None:
                    try:
                        storage_queue.task_done()
                    except ValueError:
                        pass
    
    async def _process_storage_batch(self, batch: List, results: dict, url: str):
        """Process a batch of items for storage."""
        if not batch:
            return
        
        # Process batch concurrently
        batch_tasks = []
        for image_bytes, thumbnail_bytes, image_info in batch:
            batch_tasks.append(self._store_single_item(image_bytes, thumbnail_bytes, image_info, results, url))
        
        await asyncio.gather(*batch_tasks, return_exceptions=True)
    
    async def _store_single_item(self, image_bytes: bytes, thumbnail_bytes: Optional[bytes], image_info: 'ImageInfo', results: dict, url: str):
        """Store a single item and update cache."""
        try:
            async with self._storage_semaphore:
                # Debug logging for video URL
                logger.info(f"Processing image with video URL: {image_info.video_url}")
                
                # Use storage facade to save image and thumbnail
                raw_key, thumb_key = await self._storage_facade.save_raw_and_thumbnail(
                    image_bytes=image_bytes,
                    thumbnail_bytes=thumbnail_bytes,
                    image_info=image_info,
                    page_url=url
                )
                
                # Add keys to results
                if raw_key:
                    results['saved_raw_keys'].append(raw_key)
                if thumb_key:
                    results['saved_thumbnail_keys'].append(thumb_key)
                        
        except Exception as e:
            logger.error(f"Storage error for {_truncate_log_string(image_info.url)}: {e}")
            results['errors'].append(f"Storage error for {_truncate_log_string(image_info.url)}: {e}")
    
    # ============================================================================
    # IMAGE EXTRACTION METHODS
    # ============================================================================
    
    def extract_images_by_method(self, html_content: str, base_url: str, method: str = "smart") -> Tuple[List[ImageInfo], str]:
        """
        Extract images using configurable targeting methods with site recipe support.
        
        Supports flexible CSS selector patterns for different website structures.
        Now integrates with site recipes for per-domain customization.
        
        Args:
            html_content: The HTML content to parse
            base_url: Base URL for resolving relative URLs
            method: Targeting method ('smart', 'data-mediumthumb', 'js-videoThumb', etc.)
            
        Returns:
            Tuple of (images_list, method_used)
        """
        images = []
        method_used = method
        
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Get site recipe for this URL
            recipe = get_recipe_for_url(base_url)
            recipe_method = recipe.get("method", method)
            
            # Use recipe method if different from requested method
            if recipe_method != method and method == "smart":
                logger.info(f"Site recipe overrides method '{method}' with '{recipe_method}' for {base_url}")
                method = recipe_method
                method_used = recipe_method
            
            if method == "smart" or method == recipe_method:
                # Use site recipe selectors if available, otherwise fall back to built-in patterns
                recipe_selectors = recipe.get("selectors")
                
                if recipe_selectors:
                    logger.debug(f"Using site recipe selectors for {base_url}: {len(recipe_selectors)} selectors")
                    images = self._extract_with_selectors(soup, base_url, recipe_selectors)
                    if images:
                        method_used = f"recipe-{recipe_method}"
                        logger.info(f"Site recipe method '{recipe_method}' found {len(images)} images")
                        # Debug: Show first few image URLs
                        for i, img in enumerate(images[:3]):
                            logger.debug(f"DEBUG: Recipe image {i+1}: {img.url}")
                    else:
                        logger.debug(f"Site recipe found no images, falling back to built-in patterns")
                        # Fall back to built-in patterns
                        images, method_used = self._extract_with_builtin_patterns(soup, base_url, "smart")
                else:
                    # No recipe selectors, use built-in patterns
                    images, method_used = self._extract_with_builtin_patterns(soup, base_url, "smart")
            else:
                # Use specific method (either built-in or recipe-based)
                images, method_used = self._extract_with_builtin_patterns(soup, base_url, method)
            
            logger.info(f"Found {len(images)} images using method: {method_used}")
                
        except Exception as e:
            logger.error(f"Error parsing HTML content: {str(e)}")
            
        return images, method_used
    
    def _extract_with_builtin_patterns(self, soup, base_url: str, method: str) -> Tuple[List[ImageInfo], str]:
        """
        Extract images using built-in patterns (fallback when no recipe is available).
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative URLs
            method: Targeting method
            
        Returns:
            Tuple of (images_list, method_used)
        """
        images = []
        method_used = method
        
        if method == "smart":
            # Try multiple extraction patterns in order of preference
            patterns = [
                # PornHub-style patterns
                ("data-mediumthumb", [{"selector": "img[data-mediumthumb]", "description": "data-mediumthumb attribute"}]),
                ("js-videoThumb", [{"selector": "img.js-videoThumb", "description": "js-videoThumb class"}]),
                ("phimage", [{"selector": ".phimage img", "description": "images in .phimage containers"}]),
                ("latestThumb", [{"selector": "a.latestThumb img", "description": "images in .latestThumb links"}]),
                
                # Common video thumbnail patterns
                ("video-thumb", [
                    {"selector": "img[data-video-thumb]", "description": "data-video-thumb attribute"},
                    {"selector": ".video-thumb img", "description": ".video-thumb container images"},
                    {"selector": ".thumbnail img", "description": ".thumbnail container images"},
                    {"selector": ".thumb img", "description": ".thumb container images"}
                ]),
                
                # Size-based patterns (common video thumbnail dimensions)
                ("size-320x180", [{"selector": "img[width='320'][height='180']", "description": "320x180 dimensions"}]),
                ("size-640x360", [{"selector": "img[width='640'][height='360']", "description": "640x360 dimensions"}]),
                ("size-1280x720", [{"selector": "img[width='1280'][height='720']", "description": "1280x720 dimensions"}]),
                
                # Generic patterns
                ("all-images", [{"selector": "img", "description": "all images"}])
            ]
            
            logger.debug(f"DEBUG: Starting built-in smart method extraction for URL: {base_url}")
            for pattern_name, selectors in patterns:
                logger.debug(f"DEBUG: Trying pattern: {pattern_name}")
                images = self._extract_with_selectors(soup, base_url, selectors)
                logger.debug(f"DEBUG: Pattern {pattern_name} found {len(images)} images")
                if images:
                    method_used = pattern_name
                    logger.info(f"Built-in method selected: {pattern_name} (found {len(images)} images)")
                    # Debug: Show first few image URLs
                    for i, img in enumerate(images[:3]):
                        logger.debug(f"DEBUG: Sample image {i+1}: {img.url}")
                    break
                else:
                    logger.debug(f"DEBUG: Pattern {pattern_name} found no images, trying next pattern")
        else:
            # Use specific method
            if method in ["data-mediumthumb", "js-videoThumb", "phimage", "latestThumb", "video-thumb", "size-320x180", "size-640x360", "size-1280x720", "all-images"]:
                # Map method names to their selectors
                method_selectors = {
                    "data-mediumthumb": [{"selector": "img[data-mediumthumb]", "description": "data-mediumthumb attribute"}],
                    "js-videoThumb": [{"selector": "img.js-videoThumb", "description": "js-videoThumb class"}],
                    "phimage": [{"selector": ".phimage img", "description": "images in .phimage containers"}],
                    "latestThumb": [{"selector": "a.latestThumb img", "description": "images in .latestThumb links"}],
                    "video-thumb": [
                        {"selector": "img[data-video-thumb]", "description": "data-video-thumb attribute"},
                        {"selector": ".video-thumb img", "description": ".video-thumb container images"},
                        {"selector": ".thumbnail img", "description": ".thumbnail container images"},
                        {"selector": ".thumb img", "description": ".thumb container images"}
                    ],
                    "size-320x180": [{"selector": "img[width='320'][height='180']", "description": "320x180 dimensions"}],
                    "size-640x360": [{"selector": "img[width='640'][height='360']", "description": "640x360 dimensions"}],
                    "size-1280x720": [{"selector": "img[width='1280'][height='720']", "description": "1280x720 dimensions"}],
                    "all-images": [{"selector": "img", "description": "all images"}]
                }
                images = self._extract_with_selectors(soup, base_url, method_selectors[method])
            else:
                logger.warning(f"Unknown method '{method}', falling back to all images")
                images = self._extract_with_selectors(soup, base_url, [{"selector": "img", "description": "all images"}])
                method_used = "all-images"
        
        return images, method_used
    
    def _extract_with_selectors(self, soup, base_url: str, selectors: List[Dict]) -> List[ImageInfo]:
        """
        Extract images using CSS selectors with flexible matching and expanded sources.
        
        Provides robust selector handling for various HTML structures and additional
        extraction methods for comprehensive thumbnail discovery.
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative URLs
            selectors: List of selector dictionaries with 'selector' and 'description' keys
            
        Returns:
            List of ImageInfo objects
        """
        all_images = []
        seen_urls = set()  # Avoid duplicates
        
        # First, try traditional CSS selector-based extraction
        for selector_config in selectors:
            try:
                selector = selector_config["selector"]
                description = selector_config.get("description", selector)
                
                # Handle different types of selectors
                if selector.startswith("img["):
                    # Attribute-based selector (e.g., "img[data-mediumthumb]")
                    if "width=" in selector and "height=" in selector:
                        # Parse width/height attributes
                        width_match = selector.split("width='")[1].split("'")[0] if "width='" in selector else None
                        height_match = selector.split("height='")[1].split("'")[0] if "height='" in selector else None
                        if width_match and height_match:
                            imgs = soup.find_all('img', attrs={'width': width_match, 'height': height_match})
                        else:
                            imgs = []
                    else:
                        # Other attribute selectors
                        attr_name = selector.split("[")[1].split("]")[0]
                        if "=" in attr_name:
                            # Has value (e.g., "data-mediumthumb=true")
                            attr_parts = attr_name.split("=")
                            attr_key = attr_parts[0]
                            attr_value = attr_parts[1].strip("'\"")
                            imgs = soup.find_all('img', attrs={attr_key: attr_value})
                        else:
                            # Attribute exists without value (e.g., "data-mediumthumb")
                            imgs = soup.find_all('img', attrs={attr_name: True})
                elif selector.startswith("img."):
                    # Class-based selector (e.g., "img.js-videoThumb")
                    class_name = selector.split(".")[1]
                    imgs = soup.find_all('img', class_=class_name)
                elif " img" in selector:
                    # Container-based selector (e.g., ".phimage img", "a.latestThumb img")
                    container_selector = selector.split(" img")[0]
                    containers = soup.select(container_selector)
                    imgs = []
                    for container in containers:
                        imgs.extend(container.find_all('img'))
                else:
                    # Generic CSS selector
                    imgs = soup.select(selector)
                
                # Process found images - keep original img tags to preserve parent structure
                for img in imgs:
                    img_urls = self._extract_img_urls(img, base_url)
                    # If img has URLs, add the original img tag (not a MockImg) so we preserve parent structure
                    if img_urls:
                        # Use the first URL as the primary src
                        primary_url = img_urls[0]
                        if primary_url and primary_url not in seen_urls:
                            seen_urls.add(primary_url)
                            all_images.append(img)  # Add the original BeautifulSoup img tag
                        
            except Exception as e:
                logger.warning(f"Error with selector '{selector_config}': {str(e)}")
                continue
        
        # Then, perform comprehensive extraction for additional sources
        additional_images = self._extract_additional_sources(soup, base_url, seen_urls)
        all_images.extend(additional_images)
        
        logger.debug(f"Found {len(all_images)} unique images using {len(selectors)} selectors + additional sources")
        return self._process_img_tags(all_images, base_url)
    
    def _extract_img_urls(self, img_tag, base_url: str) -> List[str]:
        """
        Extract all possible image URLs from an img tag including srcset.
        
        Args:
            img_tag: BeautifulSoup img tag
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of extracted image URLs
        """
        urls = []
        
        # Standard src attribute
        src = img_tag.get('src')
        if src:
            urls.append(urljoin(base_url, src))
        
        # Data attributes (lazy loading)
        data_src = img_tag.get('data-src')
        if data_src:
            urls.append(urljoin(base_url, data_src))
        
        data_lazy_src = img_tag.get('data-lazy-src')
        if data_lazy_src:
            urls.append(urljoin(base_url, data_lazy_src))
        
        # Srcset attribute
        srcset = img_tag.get('srcset')
        if srcset:
            srcset_url = pick_from_srcset(srcset, base_url)
            if srcset_url:
                urls.append(srcset_url)
        
        # Other common data attributes
        for attr in ['data-original', 'data-large', 'data-medium', 'data-thumb']:
            value = img_tag.get(attr)
            if value:
                urls.append(urljoin(base_url, value))
        
        return urls
    
    def _extract_additional_sources(self, soup, base_url: str, seen_urls: Set[str]) -> List:
        """
        Extract images from additional sources beyond CSS selectors.
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative URLs
            seen_urls: Set of URLs already found to avoid duplicates
            
        Returns:
            List of mock img tags for additional images found
        """
        additional_images = []
        
        try:
            # 1. Meta tags (Open Graph, Twitter Cards, etc.)
            meta_selectors = [
                ('meta[property="og:image"]', 'content'),
                ('meta[name="twitter:image"]', 'content'),
                ('meta[name="twitter:image:src"]', 'content'),
                ('meta[property="og:image:url"]', 'content'),
                ('meta[name="thumbnail"]', 'content'),
                ('meta[name="image"]', 'content'),
                ('meta[property="image"]', 'content'),
                ('link[rel="image_src"]', 'href'),
                ('link[rel="apple-touch-icon"]', 'href'),
                ('link[rel="icon"]', 'href')
            ]
            
            for selector, attr in meta_selectors:
                for element in soup.select(selector):
                    url = element.get(attr)
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        absolute_url = urljoin(base_url, url)
                        class MockImg:
                            def __init__(self, src_url):
                                self.src_url = src_url
                            def get(self, attr, default=''):
                                return self.src_url if attr == 'src' else default
                        mock_img = MockImg(absolute_url)
                        additional_images.append(mock_img)
            
            # 2. Video poster attributes
            for video in soup.find_all('video', poster=True):
                poster_url = video.get('poster')
                if poster_url and poster_url not in seen_urls:
                    seen_urls.add(poster_url)
                    absolute_url = urljoin(base_url, poster_url)
                    class MockImg:
                        def __init__(self, src_url):
                            self.src_url = src_url
                        def get(self, attr, default=''):
                            return self.src_url if attr == 'src' else default
                    mock_img = MockImg(absolute_url)
                    additional_images.append(mock_img)
            
            # 3. Source tags with srcset
            for source in soup.find_all('source', srcset=True):
                srcset = source.get('srcset')
                if srcset:
                    srcset_url = pick_from_srcset(srcset, base_url)
                    if srcset_url and srcset_url not in seen_urls:
                        seen_urls.add(srcset_url)
                        class MockImg:
                            def __init__(self, src_url):
                                self.src_url = src_url
                            def get(self, attr, default=''):
                                return self.src_url if attr == 'src' else default
                        mock_img = MockImg(srcset_url)
                        additional_images.append(mock_img)
                
                # Also check src attribute on source tags
                src = source.get('src')
                if src and src not in seen_urls:
                    seen_urls.add(src)
                    absolute_url = urljoin(base_url, src)
                    class MockImg:
                        def __init__(self, src_url):
                            self.src_url = src_url
                        def get(self, attr, default=''):
                            return self.src_url if attr == 'src' else default
                    mock_img = MockImg(absolute_url)
                    additional_images.append(mock_img)
            
            # 4. Inline background images from style attributes
            for element in soup.find_all(style=True):
                style_attr = element.get('style')
                if style_attr:
                    bg_url = extract_style_bg_url(style_attr, base_url)
                    if bg_url and bg_url not in seen_urls:
                        seen_urls.add(bg_url)
                        class MockImg:
                            def __init__(self, src_url):
                                self.src_url = src_url
                            def get(self, attr, default=''):
                                return self.src_url if attr == 'src' else default
                        mock_img = MockImg(bg_url)
                        additional_images.append(mock_img)
            
            # 5. JSON-LD structured data
            jsonld_urls = extract_jsonld_thumbnails(str(soup), base_url)
            for url in jsonld_urls:
                if url not in seen_urls:
                    seen_urls.add(url)
                    class MockImg:
                        def __init__(self, src_url):
                            self.src_url = src_url
                        def get(self, attr, default=''):
                            return self.src_url if attr == 'src' else default
                    mock_img = MockImg(url)
                    additional_images.append(mock_img)
            
        except Exception as e:
            logger.warning(f"Error extracting additional sources: {e}")
        
        return additional_images
    
    def _process_img_tags(self, img_tags, base_url: str) -> List[ImageInfo]:
        """Process a list of img tags and return ImageInfo objects with security validation and recipe-based attribute extraction."""
        images = []
        
        # Get site recipe for attribute priority
        recipe = get_recipe_for_url(base_url)
        attributes_priority = recipe.get("attributes_priority", ["alt", "title", "data-title", "data-alt"])
        extra_sources = recipe.get("extra_sources", [])
        
        for img in img_tags:
            # Try multiple sources for image URL (including recipe-specific sources)
            src = None
            for source in ["src"] + extra_sources:
                src = img.get(source)
                if src:
                    break
            
            if not src:
                continue
                
            # Resolve relative URLs
            absolute_url = urljoin(base_url, src)
            
            # Validate URL security
            is_safe, reason = validate_url_security(absolute_url)
            if not is_safe:
                logger.info(f"Image URL rejected: {reason} - {_truncate_log_string(absolute_url)}")
                continue
            
            # Explicitly reject SVG URLs
            if absolute_url.lower().endswith('.svg'):
                logger.info(f"Image URL rejected: SKIP_SVG - {_truncate_log_string(absolute_url)}")
                continue
            
            # Extract image metadata using recipe-based attribute priority
            alt_text = ""
            title = ""
            
            # Extract alt text using priority order
            for attr in attributes_priority:
                if attr in ["alt", "title"]:
                    value = img.get(attr, '')
                    if value:
                        if attr == "alt":
                            alt_text = value
                        elif attr == "title":
                            title = value
                        break
                else:
                    # Custom attributes
                    value = img.get(attr, '')
                    if value:
                        # Use first non-empty attribute for alt_text
                        if not alt_text:
                            alt_text = value
                        elif not title:
                            title = value
                        break
            
            # Fallback to standard attributes if recipe attributes didn't provide values
            if not alt_text:
                alt_text = img.get('alt', '')
            if not title:
                title = img.get('title', '')
            
            width = img.get('width')
            height = img.get('height')
            
            # Convert width/height to integers if possible
            try:
                width = int(width) if width else None
            except (ValueError, TypeError):
                width = None
                
            try:
                height = int(height) if height else None
            except (ValueError, TypeError):
                height = None
            
            # Extract video URL from parent elements or data attributes
            video_url = self._extract_video_url_from_context(img, base_url)
            
            # Debug logging for video URL extraction
            if video_url:
                logger.info(f"✅ Extracted video URL: {video_url[:100]}... for image: {absolute_url[:80]}...")
            else:
                logger.info(f"❌ No video URL found for image: {absolute_url[:80]}...")
            
            images.append(ImageInfo(
                url=absolute_url,
                alt_text=alt_text,
                title=title,
                width=width,
                height=height,
                video_url=video_url
            ))
        
        return images
    
    def _extract_video_url_from_context(self, img_tag, base_url: str) -> Optional[str]:
        """
        Extract video URL from the context around an image tag.
        Looks for video URLs in parent <a> tags, data attributes, and common patterns.
        """
        try:
            # Skip video URL extraction for MockImg objects (they don't have parent structure)
            if not hasattr(img_tag, 'find_parent'):
                return None
            
            src = img_tag.get('src', '')
            logger.info(f"🔍 Extracting video URL from context for img tag: {src[:50]}...")
            # Method 1: Check parent <a> tag href
            parent_link = img_tag.find_parent('a')
            logger.info(f"DEBUG: parent_link = {parent_link is not None}")
            if parent_link:
                href = parent_link.get('href')
                logger.info(f"DEBUG: href = {href[:50] if href else None}...")
                if href:
                    absolute_url = urljoin(base_url, href)
                    logger.info(f"DEBUG: absolute_url = {absolute_url[:100]}...")
                    
                    # Validate if this looks like a video URL
                    is_video = self._is_video_url(absolute_url)
                    logger.info(f"DEBUG: is_video = {is_video}")
                    if is_video:
                        logger.info(f"✅ Found video URL in parent <a> tag: {absolute_url[:100]}...")
                        return absolute_url
                    else:
                        logger.info(f"❌ Parent href is not a video URL: {absolute_url[:100]}...")
            else:
                logger.info(f"❌ No parent <a> tag found for img")
            
            # Method 2: Check data attributes on the image tag itself
            video_attrs = ['data-video-url', 'data-video', 'data-href', 'data-link']
            for attr in video_attrs:
                video_url = img_tag.get(attr)
                if video_url:
                    absolute_url = urljoin(base_url, video_url)
                    if self._is_video_url(absolute_url):
                        return absolute_url
            
            # Method 3: Check parent container elements for video links
            for parent in img_tag.parents:
                # Check if parent has video-related attributes
                if parent.name == 'div' or parent.name == 'article':
                    for attr in video_attrs:
                        video_url = parent.get(attr)
                        if video_url:
                            absolute_url = urljoin(base_url, video_url)
                            if self._is_video_url(absolute_url):
                                return absolute_url
                    
                    # Look for child <a> tags that might contain video URLs
                    child_links = parent.find_all('a', href=True)
                    for link in child_links:
                        href = link.get('href')
                        if href:
                            absolute_url = urljoin(base_url, href)
                            if self._is_video_url(absolute_url):
                                return absolute_url
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting video URL from context: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _is_video_url(self, url: str) -> bool:
        """
        Determine if a URL is likely a video URL based on common patterns.
        """
        if not url:
            return False
        
        url_lower = url.lower()
        
        # Common video URL patterns
        video_patterns = [
            '/view_video.php',
            '/videos/',
            '/video/',
            '/watch/',
            '/play/',
            'viewkey=',
            'video_id=',
            'v=',
            '/embed/',
            '.mp4',
            '.avi',
            '.mov',
            '.mkv',
            '.webm',
            '/player/',
            '/stream/',
        ]
        
        # Check if URL contains any video patterns
        for pattern in video_patterns:
            if pattern in url_lower:
                return True
        
        # Check if it's a pornhub-style URL
        if 'pornhub' in url_lower and ('view_video' in url_lower or 'videos' in url_lower):
            return True
        
        # Check if it's an xvideos-style URL
        if 'xvideos' in url_lower and ('video' in url_lower):
            return True
        
        return False
    
    async def download_image(self, image_info: ImageInfo) -> Tuple[Optional[bytes], List[str]]:
        """Download an image from its URL with streaming and early abort capabilities."""
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use 'async with' context manager.")
        
        # Use separate download semaphore for better throughput
        async with self._download_semaphore:
            errors = []
            
            try:
                logger.info(f"Downloading image: {_truncate_log_string(image_info.url)}")
                t_download_start = datetime.utcnow()
            
                # Download with streaming and early abort rules
                last_exc = None
                for attempt in range(3):
                    try:
                        # Use streaming GET request with manual redirect handling
                        content = await self._download_with_redirect_handling(image_info.url, errors)
                        if content is None:  # Early abort occurred
                            return None, errors
                        
                        break
                    except httpx.HTTPError as e:
                        last_exc = e
                        if attempt < 2:
                            await asyncio.sleep(0.5 * (2 ** attempt))
                            continue
                        raise
                
                t_download_ms = (datetime.utcnow() - t_download_start).total_seconds() * 1000.0
                logger.debug(f"Downloaded image in {t_download_ms:.1f} ms: {_truncate_log_string(image_info.url)}")
                
                return content, errors
                
            except httpx.HTTPError as e:
                error_msg = f"HTTP error downloading {_truncate_log_string(image_info.url)}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                return None, errors
            except Exception as e:
                error_msg = f"Unexpected error downloading {_truncate_log_string(image_info.url)}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                return None, errors
    
    async def _download_with_redirect_handling(self, url: str, errors: List[str]) -> Optional[bytes]:
        """Download with manual redirect handling for security."""
        current_url = url
        redirect_count = 0
        
        while redirect_count <= self.max_redirects:
            # Validate URL security
            is_safe, reason = validate_url_security(current_url)
            if not is_safe:
                logger.warning(f"URL rejected: {reason} - {_truncate_log_string(current_url)}")
                errors.append(f"URL rejected: {reason}")
                return None
            
            try:
                async with self.session.stream("GET", current_url) as response:
                    # Handle redirects
                    if response.status_code in (301, 302, 303, 307, 308):
                        redirect_url = response.headers.get('location')
                        if not redirect_url:
                            errors.append("Redirect without location header")
                            return None
                        
                        # Resolve relative redirects
                        redirect_url = urljoin(current_url, redirect_url)
                        
                        # Validate redirect security
                        is_safe, reason = validate_redirect_security(redirect_url, redirect_count, self.max_redirects)
                        if not is_safe:
                            logger.warning(f"Redirect rejected: {reason} - {_truncate_log_string(redirect_url)}")
                            errors.append(f"Redirect rejected: {reason}")
                            return None
                        
                        current_url = redirect_url
                        redirect_count += 1
                        continue
                    
                    response.raise_for_status()
                    
                    # Validate content security
                    content_type = response.headers.get('content-type', '')
                    content_length = response.headers.get('content-length')
                    content_length = int(content_length) if content_length else None
                    
                    is_safe, reason = validate_content_security(content_type, content_length)
                    if not is_safe:
                        logger.warning(f"Content rejected: {reason} - {content_type}")
                        errors.append(f"Content rejected: {reason}")
                        return None
                    
                    # Stream content with rolling buffer for MIME sniffing and early abort
                    content = await self._stream_image_content(response, ImageInfo(url=current_url, alt_text='', title='', width=None, height=None), errors)
                    return content
                    
            except httpx.HTTPError as e:
                if redirect_count == 0:
                    raise
                else:
                    errors.append(f"HTTP error after {redirect_count} redirects: {str(e)}")
                    return None
        
        errors.append(f"Too many redirects: {redirect_count}")
        return None
    
    async def _stream_image_content(self, response: httpx.Response, image_info: ImageInfo, errors: List[str]) -> Optional[bytes]:
        """
        Stream image content with rolling buffer for MIME sniffing and early abort rules.
        
        Args:
            response: HTTP response object with streaming capability
            image_info: Image information object
            errors: List to append any errors to
            
        Returns:
            Complete image content as bytes, or None if early abort occurred
        """
        content_chunks = []
        total_size = 0
        rolling_buffer = bytearray()
        buffer_size = 128 * 1024  # 128KB rolling buffer
        header_check_size = 32 * 1024  # Check first 32KB for image headers
        header_checked = False
        
        try:
            async for chunk in response.aiter_bytes(chunk_size=64 * 1024):  # 64KB chunks
                total_size += len(chunk)
                
                # Early abort: Check total size against security limit
                if total_size > MAX_CONTENT_LENGTH:
                    errors.append(f"File too large during streaming: {total_size} bytes (max: {MAX_CONTENT_LENGTH})")
                    return None
                
                # Add to rolling buffer for header detection
                rolling_buffer.extend(chunk)
                
                # Keep rolling buffer at desired size
                if len(rolling_buffer) > buffer_size:
                    # Move excess to content chunks
                    excess = len(rolling_buffer) - buffer_size
                    content_chunks.append(bytes(rolling_buffer[:excess]))
                    rolling_buffer = rolling_buffer[excess:]
                
                # Early abort: Check image headers in first 32KB
                if not header_checked and total_size >= header_check_size:
                    header_checked = True
                    if not self._is_image_header(rolling_buffer[:header_check_size]):
                        errors.append("First 32KB does not contain recognizable image header")
                        return None
                
                # Store chunk for final assembly
                content_chunks.append(chunk)
            
            # Add remaining buffer content
            if rolling_buffer:
                content_chunks.append(bytes(rolling_buffer))
            
            # Assemble final content
            content = b''.join(content_chunks)
            
            # Final size check against security limit
            if len(content) > MAX_CONTENT_LENGTH:
                errors.append(f"Final file size too large: {len(content)} bytes (max: {MAX_CONTENT_LENGTH})")
                return None
            
            # Strip EXIF data for security
            content = self._strip_exif_data(content)
            
            return content
            
        except Exception as e:
            error_msg = f"Error during streaming: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None
        finally:
            # Clean up references to free memory immediately
            del content_chunks
            del rolling_buffer
            gc.collect()
    
    def _strip_exif_data(self, image_bytes: bytes) -> bytes:
        """
        Strip EXIF data from image for security.
        
        Args:
            image_bytes: Original image data
            
        Returns:
            Image data with EXIF stripped
        """
        try:
            from PIL import Image
            import io
            
            # Open image and remove EXIF
            image = Image.open(io.BytesIO(image_bytes))
            
            # Create new image without EXIF
            if image.mode in ('RGBA', 'LA', 'P'):
                # Convert to RGB for JPEG compatibility
                image = image.convert('RGB')
            
            # Save without EXIF
            output = io.BytesIO()
            image.save(output, format='JPEG', quality=95, optimize=True)
            
            return output.getvalue()
            
        except Exception as e:
            logger.warning(f"Failed to strip EXIF data: {e}. Using original image.")
            return image_bytes
    
    def _check_image_dimensions(self, image_bytes: bytes, min_dimension: int = 100) -> bool:
        """
        Check if image meets minimum dimension requirements.
        
        Args:
            image_bytes: Image data
            min_dimension: Minimum width or height in pixels (default: 100)
            
        Returns:
            True if image meets dimension requirements, False otherwise
        """
        try:
            from PIL import Image
            import io
            
            # Open image to get dimensions
            image = Image.open(io.BytesIO(image_bytes))
            width, height = image.size
            
            # Check if either dimension is below minimum
            if width < min_dimension or height < min_dimension:
                logger.info(f"Image dimensions {width}x{height} below minimum {min_dimension}px, skipping")
                return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to check image dimensions: {e}. Proceeding with processing.")
            return True  # Allow processing to continue if dimension check fails
    
    def _is_image_header(self, data: bytes) -> bool:
        """
        Check if the given bytes contain a recognizable image header.
        
        Args:
            data: First portion of file data (up to 32KB)
            
        Returns:
            True if the data appears to be from an image file
        """
        if len(data) < 4:
            return False
        
        # Check for JPEG (starts with FFD8)
        if data.startswith(b'\xFF\xD8'):
            return True
        
        # Check for PNG (starts with 89504E470D0A1A0A)
        if data.startswith(b'\x89PNG\r\n\x1a\n'):
            return True
        
        # Check for GIF87a and GIF89a
        if data.startswith(b'GIF87a') or data.startswith(b'GIF89a'):
            return True
        
        # Check for WebP (RIFF container with WEBP format)
        if data.startswith(b'RIFF') and len(data) >= 12 and b'WEBP' in data[:12]:
            return True
        
        # Check for BMP (starts with BM)
        if data.startswith(b'BM'):
            return True
        
        return False
    
    async def _async_face_detection(self, image_bytes: bytes, image_info: ImageInfo) -> Tuple[List, Optional[bytes]]:
        """
        Run face detection using the ImageProcessingService.
        """
        if hasattr(self, '_image_processing_service'):
            return await self._image_processing_service.async_face_detection(image_bytes, image_info)
        else:
            logger.error("ImageProcessingService not available")
            return [], None
    
    async def _process_single_image(self, image_info: ImageInfo, index: int, total: int) -> Tuple[Optional[str], Optional[str], bool, List[str]]:
        """
        Process a single image with memory management and multi-tenancy support.
        """
        # Track this task for memory management
        task = asyncio.current_task()
        if task:
            self._active_tasks.add(task)
        
        try:
            # Update dynamic concurrency before processing
            self._update_dynamic_concurrency()
            
            async with self._processing_semaphore:
                # Proactive memory management
                self._manage_memory_pressure()
                
                logger.info(f"Processing image {index}/{total}: {_truncate_log_string(image_info.url)}")

                # Download the image
                image_bytes, download_errors = await self.download_image(image_info)
                if not image_bytes:
                    logger.warning(f"Failed to download image: {_truncate_log_string(image_info.url)}")
                    return None, None, False, download_errors, []

                # Check image dimensions using ImageProcessingService
                if hasattr(self, '_image_processing_service') and not self._image_processing_service.check_image_dimensions(image_bytes):
                    logger.info(f"Image dimensions too small, skipping: {_truncate_log_string(image_info.url)}")
                    return None, None, False, download_errors, []

                # Check cache using caching facade
                should_skip, cached_key = await self._caching_facade.should_skip_image(image_info.url, image_bytes, self.tenant_id)
                if should_skip and cached_key:
                    logger.info(f"Image {_truncate_log_string(image_info.url)} found in cache. Key: {_truncate_log_string(cached_key)}")
                    return cached_key, cached_key, True, download_errors, []

                # Use async face detection with thread pool
                faces = []
                thumbnail_bytes = None
                if self.require_face or self.crop_faces:
                    t_detect_start = datetime.utcnow()
                    faces, thumbnail_bytes = await self._async_face_detection(image_bytes, image_info)
                    t_detect_ms = (datetime.utcnow() - t_detect_start).total_seconds() * 1000.0
                    logger.debug(f"Face detection pipeline completed in {t_detect_ms:.1f} ms for {_truncate_log_string(image_info.url)}")
                    # Track early-exit usage
                    try:
                        early_exit_used = get_face_service().consume_early_exit_flag()
                        if early_exit_used:
                            setattr(self, "_early_exit_count", getattr(self, "_early_exit_count", 0) + 1)
                    except Exception:
                        pass
                    
                    if self.require_face and not faces:
                        logger.info(f"No faces detected for {_truncate_log_string(image_info.url)}, skipping.")
                        return None, None, False, download_errors, []
                    
                    # When crop_faces=true but no faces detected, don't create thumbnails
                    if self.crop_faces and not faces:
                        logger.info(f"No faces detected for {_truncate_log_string(image_info.url)}, no thumbnail created (crop_faces=true).")
                # No thumbnail created when face requirements are disabled

                # Prepare data for batch storage
                return image_bytes, thumbnail_bytes, False, download_errors, faces

        except Exception as e:
            logger.error(f"Error processing image {_truncate_log_string(image_info.url)}: {e}", exc_info=True)
            return None, None, False, [f"Processing error: {e}"], []
        finally:
            # Remove task from tracking when complete
            if task and task in self._active_tasks:
                self._active_tasks.discard(task)
    
    # ============================================================================
    # URL PROCESSING METHODS
    # ============================================================================
    
    async def fetch_page(self, url: str) -> Tuple[Optional[str], List[str]]:
        """Fetch a web page and return its content and any errors."""
        if not self.session or not self._http_service:
            raise RuntimeError("Crawler not initialized. Use 'async with' context manager.")
            
        errors = []
        
        try:
            # Validate URL security first
            is_safe, reason = validate_url_security(url)
            if not is_safe:
                logger.warning(f"Page URL rejected: {reason} - {_truncate_log_string(url)}")
                errors.append(f"Page URL rejected: {reason}")
                return None, errors
            
            # Use smart fallback: HTML-first, JS if < 3 images found
            content, fetch_errors = await self._fetch_page_with_smart_fallback(url)
            errors.extend(fetch_errors)
            
            if content is None:
                logger.error(f"Failed to fetch page with smart fallback: {_truncate_log_string(url)}")
                return None, errors
            
            return content, errors
            
        except Exception as e:
            error_msg = f"Unexpected error fetching {_truncate_log_string(url)}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None, errors
    
    
    def extract_page_urls(self, html_content: str, base_url: str) -> List[str]:
        """
        Extract URLs from HTML content for further crawling.
        
        Args:
            html_content: HTML content to parse
            base_url: Base URL for resolving relative links
            
        Returns:
            List of URLs found on the page
        """
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            urls = set()
            
            # Extract links from <a> tags
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, href)
                    parsed_url = urlparse(absolute_url)
                    
                    # Filter URLs
                    if self._is_valid_page_url(absolute_url, base_url):
                        urls.add(absolute_url)
            
            # Also look for pagination patterns (common on adult sites)
            pagination_selectors = [
                'a[href*="page="]',
                'a[href*="p="]', 
                'a[href*="/page/"]',
                'a[href*="/p/"]',
                '.pagination a',
                '.pager a',
                '.page-nav a',
                'a[class*="page"]',
                'a[class*="next"]',
                'a[class*="more"]'
            ]
            
            for selector in pagination_selectors:
                for link in soup.select(selector):
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(base_url, href)
                        if self._is_valid_page_url(absolute_url, base_url):
                            urls.add(absolute_url)
            
            return list(urls)
            
        except Exception as e:
            logger.error(f"Error extracting page URLs: {str(e)}")
            return []
    
    def _is_valid_page_url(self, url: str, base_url: str) -> bool:
        """
        Check if a URL is valid for crawling with security validation.
        
        Args:
            url: URL to check
            base_url: Base URL for domain comparison
            
        Returns:
            True if URL is valid for crawling
        """
        try:
            # First, validate URL security
            is_safe, reason = validate_url_security(url)
            if not is_safe:
                logger.info(f"URL rejected for crawling: {reason} - {_truncate_log_string(url)}")
                return False
            
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_url)
            
            # Must have scheme and netloc
            if not parsed_url.scheme or not parsed_url.netloc:
                return False
            
            # Check domain restriction
            if self.same_domain_only and parsed_url.netloc != parsed_base.netloc:
                return False
            
            # Skip non-HTTP protocols
            if parsed_url.scheme not in ['http', 'https']:
                return False
            
            # Skip common non-page URLs
            skip_patterns = [
                'javascript:', 'mailto:', 'tel:', '#',
                '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp',
                '.pdf', '.doc', '.docx', '.zip', '.rar',
                'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com'
            ]
            
            url_lower = url.lower()
            for pattern in skip_patterns:
                if pattern in url_lower:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating URL {_truncate_log_string(url)}: {str(e)}")
            return False
    

    # ============================================================================
    # MAIN CRAWLING METHODS
    # ============================================================================
    
    async def _apply_jitter(self):
        """Apply random jitter to avoid rate limiting."""
        if not self._jitter_applied:
            jitter_ms = random.randint(*self.jitter_range)
            await asyncio.sleep(jitter_ms / 1000.0)
            self._jitter_applied = True
            logger.debug(f"Applied {jitter_ms}ms jitter")
    
    async def _get_host_semaphore(self, url: str) -> asyncio.Semaphore:
        """Get or create a semaphore for per-host concurrency control."""
        parsed_url = urlparse(url)
        host = parsed_url.netloc.lower()
        
        if host not in self._per_host_semaphores:
            # Set per-host limit based on host reputation (placeholder logic)
            limit = self.per_host_concurrency
            if 'cdn' in host or 'static' in host:
                limit = min(limit * 2, 8)  # Allow more for CDNs
            
            self._per_host_semaphores[host] = asyncio.Semaphore(limit)
            self._per_host_limits[host] = limit
            logger.debug(f"Created semaphore for host {host} with limit {limit}")
        
        return self._per_host_semaphores[host]
    
    async def _fetch_page_with_smart_fallback(
        self, url: str
    ) -> Tuple[Optional[str], List[str]]:
        """
        Fetch page with intelligent HTML-first, JS fallback strategy.
        
        Strategy:
        1. Fetch with HTML-only
        2. Extract images using 3x3 selector mining
        3. If < 3 images found, try JavaScript rendering
        4. Cache HTML content if memory allows
        
        Returns:
            Tuple of (html_content, errors)
        """
        errors = []
        
        # Step 1: Fetch with HTML-only
        logger.info(f"Fetching {_truncate_log_string(url)} with HTML-only first")
        html_content, fetch_reason = await self._http_service.fetch_html(
            url, self.session, use_js=False, max_redirects=self.max_redirects
        )
        
        if html_content is None:
            logger.warning(f"HTML fetch failed: {fetch_reason} - {_truncate_log_string(url)}")
            # Try JS immediately if HTML fetch fails
            return await self._fetch_with_javascript(url, errors)
        
        # Step 2: Quick extraction test to see if HTML is sufficient
        images, method_used = await self._extract_and_validate_images(
            html_content, url, method="smart"
        )
        
        # Step 3: Check if we have enough images (threshold: 3 images)
        if len(images) >= 3:
            logger.info(f"HTML-only successful for {_truncate_log_string(url)}: {len(images)} images found")
            return html_content, errors
        
        # Step 4: HTML didn't find enough images, try JavaScript fallback
        logger.warning(f"HTML found only {len(images)} images for {_truncate_log_string(url)}, trying JS fallback")
        
        # Check memory pressure before proceeding
        memory_status = self._memory_monitor.get_memory_status()
        if memory_status['pressure_level'] in ['high', 'critical']:
            logger.warning(f"Memory pressure {memory_status['pressure_level']} ({memory_status['percent']:.1f}%) during JS fallback decision")
        
        js_content, js_errors = await self._fetch_with_javascript(url, errors)
        
        if js_content:
            # Validate JS results
            js_images, _ = await self._extract_and_validate_images(
                js_content, url, method="smart"
            )
            
            if len(js_images) >= len(images):
                logger.info(f"JS fallback successful for {_truncate_log_string(url)}: {len(js_images)} images found (HTML had {len(images)})")
                return js_content, js_errors
            else:
                logger.info(f"JS fallback found {len(js_images)} images vs HTML {len(images)}, using HTML")
                return html_content, errors
        
        # JS failed, return HTML results anyway
        logger.warning(f"JS fallback failed for {_truncate_log_string(url)}, using HTML results with {len(images)} images")
        return html_content, errors
    
    async def _fetch_with_javascript(
        self, url: str, existing_errors: List[str]
    ) -> Tuple[Optional[str], List[str]]:
        """Fetch page with JavaScript rendering."""
        errors = existing_errors.copy()
        
        logger.info(f"Attempting JavaScript rendering for {_truncate_log_string(url)}")
        js_content, fetch_reason = await self._http_service.fetch_html(
            url, self.session, use_js=True, max_redirects=self.max_redirects
        )
        
        if js_content is None:
            logger.error(f"JS fallback failed: {fetch_reason} - {_truncate_log_string(url)}")
            errors.append(f"Both HTML and JS failed: {fetch_reason}")
        
        return js_content, errors
    
    async def _extract_and_validate_images(
        self, html_content: str, base_url: str, method: str
    ) -> Tuple[List[ImageInfo], str]:
        """
        Quick extraction to validate if content has enough images.
        Returns extracted images and method used.
        """
        try:
            # Use the existing extraction infrastructure
            if hasattr(self, '_image_extractor'):
                extractor = self._image_extractor
            else:
                from .crawler_modules.extraction import ImageExtractor
                extractor = ImageExtractor()
            
            images, method_used = extractor.extract_images_by_method(
                html_content, base_url, method
            )
            
            return images, method_used
        except Exception as e:
            logger.error(f"Error in quick extraction for {_truncate_log_string(base_url)}: {e}")
            return [], method
    
    async def crawl_site_list(
        self, 
        urls: List[str], 
        method: str = "smart",
        max_images_per_site: int = 50,
        concurrent_sites: int = 3
    ) -> List[CrawlResult]:
        """
        Crawl multiple sites concurrently with controlled concurrency.
        
        Args:
            urls: List of URLs to crawl
            method: Extraction method to use
            max_images_per_site: Maximum images to process per site
            concurrent_sites: Maximum number of sites to crawl concurrently
            
        Returns:
            List of CrawlResult objects for each site
        """
        logger.info(f"Starting crawl of {len(urls)} sites with {concurrent_sites} concurrent workers")
        
        # Create semaphore to limit concurrent site crawling
        site_semaphore = asyncio.Semaphore(concurrent_sites)
        
        async def crawl_single_site(url: str) -> CrawlResult:
            """Crawl a single site with semaphore control."""
            async with site_semaphore:
                logger.info(f"Starting crawl of {url}")
                try:
                    # Use crawl_site method to continue until target thumbnails are saved
                    result = await self.crawl_site(url, method)
                    logger.info(f"Completed crawl of {url}: {result.images_found} images, {result.raw_images_saved} saved")
                    return result
                except Exception as e:
                    logger.error(f"Error crawling {url}: {e}")
                    # Return error result
                    return CrawlResult(
                        url=url,
                        images_found=0,
                        raw_images_saved=0,
                        thumbnails_saved=0,
                        pages_crawled=0,
                        saved_raw_keys=[],
                        saved_thumbnail_keys=[],
                        errors=[str(e)],
                        targeting_method=method,
                        cache_hits=0,
                        cache_misses=0,
                        redis_hits=0,
                        postgres_hits=0,
                        tenant_id=self.tenant_id,
                        early_exit_count=0
                    )
        
        # Crawl all sites concurrently
        start_time = datetime.utcnow()
        results = await asyncio.gather(*[crawl_single_site(url) for url in urls], return_exceptions=True)
        end_time = datetime.utcnow()
        
        # Handle any exceptions that occurred
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Exception in crawl for {urls[i]}: {result}")
                # Create error result
                processed_results.append(CrawlResult(
                    url=urls[i],
                    images_found=0,
                    raw_images_saved=0,
                    thumbnails_saved=0,
                    pages_crawled=0,
                    saved_raw_keys=[],
                    saved_thumbnail_keys=[],
                    errors=[str(result)],
                    targeting_method=method,
                    cache_hits=0,
                    cache_misses=0,
                    redis_hits=0,
                    postgres_hits=0,
                    tenant_id=self.tenant_id,
                    early_exit_count=0
                ))
            else:
                processed_results.append(result)
        
        # Log summary
        total_images = sum(r.images_found for r in processed_results)
        total_saved = sum(r.raw_images_saved for r in processed_results)
        total_errors = sum(len(r.errors) for r in processed_results)
        total_time = (end_time - start_time).total_seconds()
        
        logger.info(f"Site list crawl completed: {len(urls)} sites, {total_images} images found, "
                   f"{total_saved} saved, {total_errors} errors, {total_time:.2f}s total")
        
        return processed_results

    async def crawl_page(self, url: str, method: str = "smart") -> CrawlResult:
        """Crawl a single page for images using the specified method."""
        start_time = datetime.utcnow()
        logger.info(f"Starting crawl of: {_truncate_log_string(url)} using method: {method} (tenant: {self.tenant_id}, min_face_quality: {self.min_face_quality}, require_face: {self.require_face})")
        
        # Apply jitter to avoid rate limiting
        await self._apply_jitter()
        
        # Get per-host semaphore for concurrency control
        host_semaphore = await self._get_host_semaphore(url)
        
        async with host_semaphore:
            # Reset cache statistics for this crawl
            self._caching_facade.reset_cache_statistics()
            
            saved_raw_keys = []
            saved_thumbnail_keys = []
            all_errors = []
            cache_hits = 0
            cache_misses = 0
            
            # Enable dynamic concurrency with memory management
            self._adjust_concurrency_dynamically()
            
            # Fetch the page
            html_content, fetch_errors = await self.fetch_page(url)
            all_errors.extend(fetch_errors)
            
            if not html_content:
                end_time = datetime.utcnow()
                total_duration = (end_time - start_time).total_seconds()
                logger.error("Failed to fetch page content")
                return CrawlResult(
                    url=url,
                    images_found=0,
                    raw_images_saved=0,
                    thumbnails_saved=0,
                    pages_crawled=0,
                    saved_raw_keys=[],
                    saved_thumbnail_keys=[],
                    errors=all_errors,
                    targeting_method="failed",
                    tenant_id=self.tenant_id,
                    total_duration_seconds=total_duration,
                    start_time=start_time,
                    end_time=end_time
                )
        
        # Extract images using specified method
        images, method_used = self._extractor.extract_images_by_method(html_content, url, method)
        
        if not images:
            end_time = datetime.utcnow()
            total_duration = (end_time - start_time).total_seconds()
            logger.warning("No images found on the page")
            return CrawlResult(
                url=url,
                images_found=0,
                raw_images_saved=0,
                thumbnails_saved=0,
                pages_crawled=0,
                saved_raw_keys=[],
                saved_thumbnail_keys=[],
                errors=all_errors,
                targeting_method=method_used,
                tenant_id=self.tenant_id,
                total_duration_seconds=total_duration,
                start_time=start_time,
                end_time=end_time
            )
        
        # Apply image limit if specified
        if hasattr(self, 'max_total_images') and self.max_total_images > 0:
            images_to_process = images[:self.max_total_images]
            logger.info(f"Processing {len(images_to_process)} images (limited by max_total_images={self.max_total_images})")
        else:
            images_to_process = images
            logger.info(f"Processing all {len(images_to_process)} images found")
            
        # Use streaming pipeline only for larger workloads to avoid overhead
        if len(images_to_process) <= 30:
            logger.info(f"Using batch processing for {len(images_to_process)} images (streaming overhead not worth it)")
            result = await self._process_images_batch(images_to_process, all_errors, cache_hits, cache_misses, method_used, url)
        else:
            logger.info(f"Using streaming pipeline for {len(images_to_process)} images")
            result = await self._process_images_streaming(images_to_process, all_errors, cache_hits, cache_misses, method_used, url)
        
        # Add timing information to the result
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()
        result.total_duration_seconds = total_duration
        result.start_time = start_time
        result.end_time = end_time
        
        logger.info(f"Crawl completed in {total_duration:.2f} seconds")
        return result


    async def crawl_site(self, start_url: str, method: str = "smart") -> CrawlResult:
        """
        Crawl multiple pages on a site to collect images up to max_total_images.
        
        Args:
            start_url: Starting URL for crawling
            method: Targeting method for image extraction
            
        Returns:
            CrawlResult with aggregated statistics
        """
        start_time = datetime.utcnow()
        logger.info(f"Starting site crawl from: {_truncate_log_string(start_url)} (tenant: {self.tenant_id}, max_images: {self.max_total_images}, max_pages: {self.max_pages}, max_depth: {self.max_depth})")
        
        # Initialize crawling state
        visited_urls = set()
        urls_to_visit = [(start_url, 0)]  # (url, depth)
        all_saved_raw_keys = []
        all_saved_thumbnail_keys = []
        all_errors = []
        total_images_found = 0
        total_raw_saved = 0
        total_thumbnails_saved = 0
        pages_crawled = 0
        total_cache_hits = 0
        total_cache_misses = 0
        checked_urls = []  # Track all URLs that were checked
        
        # Use 3x3 depth mining if enabled
        if self.use_3x3_mining and self._selector_mining_service:
            logger.info("Using 3x3 depth mining approach for better selector discovery...")
            try:
                # Perform 3x3 depth mining to discover better URLs
                mining_result = await self._selector_mining_service.mine_selectors_for_site(
                    start_url, 
                    self.session, 
                    max_pages=9,  # 3 categories × 3 content pages each
                    limits=None
                )
                
                # Add discovered URLs to the crawl queue
                discovered_urls = mining_result.checked_urls
                logger.info(f"3x3 mining discovered {len(discovered_urls)} URLs for crawling")
                
                # Add discovered URLs to the queue (excluding the start URL)
                for url in discovered_urls:
                    if url != start_url and url not in visited_urls:
                        urls_to_visit.append((url, 0))  # Start with depth 0 for discovered URLs
                        logger.info(f"Added discovered URL to crawl queue: {_truncate_log_string(url)}")
                
                # Track URLs from mining
                checked_urls.extend(discovered_urls)
                
            except Exception as e:
                logger.warning(f"3x3 mining failed, falling back to standard crawling: {e}")
        
        while urls_to_visit and len(all_saved_thumbnail_keys) < self.max_total_images and pages_crawled < self.max_pages:
            # Get next URL to visit
            current_url, current_depth = urls_to_visit.pop(0)
            
            # Skip if already visited or depth limit exceeded
            if current_url in visited_urls or current_depth > self.max_depth:
                continue
            
            visited_urls.add(current_url)
            checked_urls.append(current_url)  # Track this URL
            pages_crawled += 1
            
            logger.info(f"Crawling page {pages_crawled}/{self.max_pages}: {_truncate_log_string(current_url)}")
            logger.info(f"Thumbnails collected so far: {len(all_saved_thumbnail_keys)}/{self.max_total_images}")
            
            try:
                # Crawl the current page
                page_result = await self.crawl_page(current_url, method)
                
                # Accumulate results
                total_images_found += page_result.images_found
                total_raw_saved += page_result.raw_images_saved
                total_thumbnails_saved += page_result.thumbnails_saved
                all_saved_raw_keys.extend(page_result.saved_raw_keys)
                all_saved_thumbnail_keys.extend(page_result.saved_thumbnail_keys)
                all_errors.extend(page_result.errors)
                total_cache_hits += page_result.cache_hits
                total_cache_misses += page_result.cache_misses
                
                logger.info(f"Page {pages_crawled} results: Found {page_result.images_found}, Raw Saved {page_result.raw_images_saved}, Thumbnails Saved {page_result.thumbnails_saved}")
                
                # If we haven't reached the thumbnail limit, discover new URLs
                if len(all_saved_thumbnail_keys) < self.max_total_images and len(urls_to_visit) < self.max_pages * 2 and current_depth < self.max_depth:
                    # Fetch page content for URL discovery
                    html_content, fetch_errors = await self.fetch_page(current_url)
                    all_errors.extend(fetch_errors)
                    
                    if html_content:
                        # Extract new URLs
                        new_urls = self.extract_page_urls(html_content, current_url)
                        
                        # Add new URLs to visit queue with incremented depth
                        for url in new_urls:
                            if url not in visited_urls and (url, current_depth + 1) not in urls_to_visit:
                                urls_to_visit.append((url, current_depth + 1))
                        
                        logger.info(f"Found {len(new_urls)} new URLs to explore at depth {current_depth + 1}")
                
                # Check if we've reached our goals
                if len(all_saved_thumbnail_keys) >= self.max_total_images:
                    logger.info(f"Reached target of {self.max_total_images} thumbnails, stopping crawl")
                    break
                    
            except Exception as e:
                error_msg = f"Error crawling page {_truncate_log_string(current_url)}: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
                continue
        
        # Create aggregated result
        end_time = datetime.utcnow()
        total_duration = (end_time - start_time).total_seconds()
        result = CrawlResult(
            url=start_url,
            images_found=total_images_found,
            raw_images_saved=total_raw_saved,
            thumbnails_saved=total_thumbnails_saved,
            pages_crawled=pages_crawled,
            saved_raw_keys=all_saved_raw_keys,
            saved_thumbnail_keys=all_saved_thumbnail_keys,
            errors=all_errors,
            targeting_method=method,
            cache_hits=total_cache_hits,
            cache_misses=total_cache_misses,
            tenant_id=self.tenant_id,
            early_exit_count=int(getattr(self, "_early_exit_count", 0)),
            total_duration_seconds=total_duration,
            start_time=start_time,
            end_time=end_time,
            checked_urls=checked_urls
        )
        
        logger.info(f"Site crawl completed in {total_duration:.2f} seconds - Pages: {pages_crawled}, Found: {total_images_found}, Raw Images Saved: {total_raw_saved}, Thumbnails Saved: {total_thumbnails_saved}")
        return result


