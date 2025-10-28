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
from dataclasses import dataclass
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

# Import redirect utilities (local copy in services directory)

# Import site recipes functionality

# Image safety configuration - set once on import
from PIL import Image, ImageFile

# ============================================================================
# SECURITY CONFIGURATION
# ============================================================================

# URL Security Configuration
from PIL import Image
import io

            # Open image to get dimensions

from urllib.parse import urljoin, urlparse, parse_qs
from .redirect_utils import create_safe_client, fetch_html_with_redirects
from ..config.site_recipes import get_recipe_for_url, get_recipe_for_host
from .storage import save_raw_and_thumb_with_precreated_thumb, save_raw_image_only, save_raw_and_thumb_content_addressed_async, save_raw_image_content_addressed, get_storage_cleanup_function
from . import storage
from .face import get_face_service, close_face_service
from . import face
from .cache import get_hybrid_cache_service, close_cache_service
from urllib.parse import urljoin, urlparse

"""
Image Crawler Service

A comprehensive web crawler service for extracting and processing images from websites.
Features include intelligent image detection, face recognition, caching, and multi-tenant support.
"""

Image.MAX_IMAGE_PIXELS = 50_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

def pick_from_srcset(srcset: str, base_url: str, preferred_width: int = 640) -> Optional[str]:
    """
    Pick the best image URL from a srcset attribute.

    Args:
        srcset: The srcset attribute value (e.g., "image1.jpg 320w, image2.jpg 640w")
        base_url: Base URL for resolving relative URLs
        preferred_width: Preferred width in pixels

    Returns:
        Best matching image URL or None if no valid srcset
    """
    if not srcset:
        return None

    try:
        # Parse srcset format: "url1 width1, url2 width2, ..."
        candidates = []
        for entry in srcset.split(','):
            entry = entry.strip()
            if not entry:
                continue

            parts = entry.split()
            if len(parts) < 2:
                continue

            url = parts[0].strip()
            descriptor = parts[1].strip()

            # Parse width descriptor (e.g., "640w")
            if descriptor.endswith('w'):
                try:
                    width = int(descriptor[:-1])
                    absolute_url = urljoin(base_url, url)
                    candidates.append((width, absolute_url))
                except ValueError:
                    continue

        if not candidates:
            return None

        # Find the closest width to preferred_width
        candidates.sort(key=lambda x: abs(x[0] - preferred_width))
        return candidates[0][1]

    except Exception as e:
        logger.debug(f"Error parsing srcset '{srcset}': {e}")
        return None

def extract_style_bg_url(style_attr: str, base_url: str) -> Optional[str]:
    """
    Extract background image URL from CSS style attribute.

    Args:
        style_attr: The style attribute value
        base_url: Base URL for resolving relative URLs

    Returns:
        Extracted background image URL or None
    """
    if not style_attr:
        return None

    try:
        # Look for background-image: url(...) patterns
        patterns = [
            r'background-image:\s*url\(["\']?([^"\']+)["\']?\)',
            r'background:\s*[^;]*url\(["\']?([^"\']+)["\']?\)',
            r'background-image:\s*url\(([^)]+)\)',
            r'background:\s*[^;]*url\(([^)]+)\)'
        ]

        for pattern in patterns:
            match = re.search(pattern, style_attr, re.IGNORECASE)
            if match:
                url = match.group(1).strip()
                if url and not url.startswith('data:'):
                    return urljoin(base_url, url)

        return None

    except Exception as e:
        logger.debug(f"Error extracting background URL from style '{style_attr}': {e}")
        return None

def extract_jsonld_thumbnails(html_content: str, base_url: str) -> List[str]:
    """
    Extract thumbnail URLs from JSON-LD structured data.

    Args:
        html_content: The HTML content to parse
        base_url: Base URL for resolving relative URLs

    Returns:
        List of extracted thumbnail URLs
    """
    thumbnails = []

    try:
        soup = BeautifulSoup(html_content, 'html.parser')

        # Find all script tags with type="application/ld+json"
        json_scripts = soup.find_all('script', type='application/ld+json')

        for script in json_scripts:
            try:
                # Parse JSON-LD
                json_data = json.loads(script.string or '')

                # Handle both single objects and arrays
                if isinstance(json_data, dict):
                    json_data = [json_data]
                elif not isinstance(json_data, list):
                    continue

                for item in json_data:
                    if not isinstance(item, dict):
                        continue

                    # Look for thumbnailUrl or image fields
                    thumbnail_fields = ['thumbnailUrl', 'image', 'contentUrl']

                    for field in thumbnail_fields:
                        if field in item:
                            value = item[field]

                            # Handle different formats
                            if isinstance(value, str):
                                # Direct URL string
                                if value and not value.startswith('data:'):
                                    thumbnails.append(urljoin(base_url, value))
                            elif isinstance(value, dict):
                                # Object with url field
                                if 'url' in value and isinstance(value['url'], str):
                                    url = value['url']
                                    if url and not url.startswith('data:'):
                                        thumbnails.append(urljoin(base_url, url))
                            elif isinstance(value, list):
                                # Array of URLs or objects
                                for item_url in value:
                                    if isinstance(item_url, str):
                                        if item_url and not item_url.startswith('data:'):
                                            thumbnails.append(urljoin(base_url, item_url))
                                    elif isinstance(item_url, dict) and 'url' in item_url:
                                        url = item_url['url']
                                        if url and not url.startswith('data:'):
                                            thumbnails.append(urljoin(base_url, url))

                    # Also check for nested objects (e.g., video.thumbnailUrl)
                    if '@type' in item:
                        # Check for VideoObject thumbnailUrl
                        if item.get('@type') == 'VideoObject' and 'thumbnailUrl' in item:
                            url = item['thumbnailUrl']
                            if url and not url.startswith('data:'):
                                thumbnails.append(urljoin(base_url, url))

                        # Check for ImageObject contentUrl
                        if item.get('@type') == 'ImageObject' and 'contentUrl' in item:
                            url = item['contentUrl']
                            if url and not url.startswith('data:'):
                                thumbnails.append(urljoin(base_url, url))

            except (json.JSONDecodeError, ValueError) as e:
                logger.debug(f"Error parsing JSON-LD: {e}")
                continue

        # Remove duplicates while preserving order
        seen = set()
        unique_thumbnails = []
        for thumbnail in thumbnails:
            if thumbnail not in seen:
                seen.add(thumbnail)
                unique_thumbnails.append(thumbnail)

        return unique_thumbnails

    except Exception as e:
        logger.debug(f"Error extracting JSON-LD thumbnails: {e}")
        return []

# ============================================================================
# MEMORY MANAGEMENT
# ============================================================================

class MemoryMonitor:
    """Memory monitoring with adaptive thresholds for system resource management."""

    def __init__(self):
        self.initial_memory = psutil.virtual_memory().percent
        self.peak_memory = self.initial_memory
        self.memory_history = []
        self.gc_triggered = False

    def get_memory_status(self) -> Dict[str, float]:
        """Get comprehensive memory status."""
        memory = psutil.virtual_memory()
        return {
            'percent': memory.percent,
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'total_gb': memory.total / (1024**3),
            'pressure_level': self._calculate_pressure_level(memory.percent)
        }

    def _calculate_pressure_level(self, memory_percent: float) -> str:
        """Calculate memory pressure level."""
        if memory_percent < 60:
            return 'low'
        elif memory_percent < 75:
            return 'moderate'
        elif memory_percent < 85:
            return 'high'
        else:
            return 'critical'

    def should_trigger_gc(self) -> bool:
        """Determine if garbage collection should be triggered."""
        status = self.get_memory_status()
        logger.info(f"Memory status: {status}")

        # Always trigger GC if memory is critical
        if status['pressure_level'] == 'critical':
            return True

        # Trigger GC if memory is high and we haven't done it recently
        if status['pressure_level'] == 'high' and not self.gc_triggered:
            self.gc_triggered = True
            return True

        # Reset GC flag when memory is low
        if status['pressure_level'] == 'low':
            self.gc_triggered = False

        return False

    def get_safe_concurrency_limit(self, base_concurrency: int) -> int:
        """Calculate safe concurrency limit based on memory pressure."""
        status = self.get_memory_status()

        if status['pressure_level'] == 'critical':
            return max(1, base_concurrency // 4)
        elif status['pressure_level'] == 'high':
            return max(2, base_concurrency // 2)
        elif status['pressure_level'] == 'moderate':
            return int(base_concurrency * 0.75)
        else:  # low
            return min(base_concurrency * 2, 30)  # Cap at 30

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

@dataclass
class CrawlResult:
    """Result of a crawl operation."""
    url: str
    images_found: int
    raw_images_saved: int
    thumbnails_saved: int
    pages_crawled: int
    saved_raw_keys: List[str]
    saved_thumbnail_keys: List[str]
    errors: List[str]
    targeting_method: str
    cache_hits: int = 0
    cache_misses: int = 0
    redis_hits: int = 0
    postgres_hits: int = 0
    tenant_id: str = "default"
    early_exit_count: int = 0
    total_duration_seconds: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


@dataclass
class ImageInfo:
    """Information about a discovered image."""
    url: str
    alt_text: str
    title: str
    width: Optional[int]
    height: Optional[int]


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
        self.enable_audit_logging = enable_audit_logging  # Audit logging support
        self._early_exit_count = 0

        # Security and crawl policy settings
        self.max_depth = DEFAULT_MAX_DEPTH
        self.per_host_concurrency = DEFAULT_PER_HOST_CONCURRENCY
        self.jitter_range = DEFAULT_JITTER_RANGE
        self.respect_robots_txt = False  # Optional robots.txt respect flag
        self.max_redirects = 3
        self._redirect_counts = {}  # Track redirect counts per URL

        # Concurrency control with semaphores and per-host limits
        self._processing_semaphore = asyncio.Semaphore(self.max_concurrent_images)
        self._storage_semaphore = asyncio.Semaphore(self.max_concurrent_images)
        self._download_semaphore = asyncio.Semaphore(min(self.max_concurrent_images * 2, 50))  # Allow more downloads
        self._per_host_semaphores = {}  # Per-host concurrency control
        self._per_host_limits = {}  # Track per-host limits

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

        # Memory monitoring and management
        self._memory_monitor = MemoryMonitor()
        self._active_tasks = weakref.WeakSet()  # Track active tasks for memory cleanup
        self._memory_pressure_threshold = 75  # Memory pressure threshold
        self._gc_frequency = 10  # Force GC every N operations
        self._operation_count = 0
        self._cpu_sample_counter = 0  # Counter for CPU sampling frequency
        self._jitter_applied = False  # Track if jitter has been applied

        self.session: Optional[httpx.AsyncClient] = None
        self.cache_service = get_hybrid_cache_service()  # Hybrid caching service
        self.pending_cache_entries = []  # Batch cache writes

    async def __aenter__(self):
        """Async context manager entry."""
        # Create secure HTTP client with redirect utilities
        limits = httpx.Limits(max_keepalive_connections=100, max_connections=200)
        self.session = create_safe_client(
            timeout=httpx.Timeout(
                connect=10.0,  # Connection timeout
                read=30.0,    # Read timeout
                write=10.0,   # Write timeout
                pool=5.0      # Pool timeout
            ),
            verify=True,  # TLS verification
            http2=True,
            limits=limits,
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
            # Clean up crawler thread pools
            logger.info("Shutting down crawler thread pools...")
            self._face_detection_thread_pool.shutdown(wait=True)
            self._storage_thread_pool.shutdown(wait=True)
            logger.info("Crawler thread pools shutdown complete")
        except Exception as e:
            logger.warning(f"Error shutting down crawler thread pools: {e}")

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
            # Clean up cache service resources
            logger.info("Cleaning up cache service resources...")
            close_cache_service()
            logger.info("Cache service resources cleaned up")
        except Exception as e:
            logger.warning(f"Error cleaning up cache service resources: {e}")

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
                    # Store using content-addressed keys
                    if thumbnail_bytes is not None:
                        raw_key, raw_url, thumbnail_key, thumb_url, metadata = await save_raw_and_thumb_content_addressed_async(
                            image_bytes,
                            thumbnail_bytes,
                            self.tenant_id,
                            image_info.url  # Pass source URL for tracking
                        )

                        if raw_key:
                            saved_raw_keys.append(raw_key)
                            saved_thumbnail_keys.append(thumbnail_key)
                            # Update cache with metadata only
                            await self.cache_service.store_crawled_image(
                                image_info.url,
                                image_bytes,
                                raw_key,
                                thumbnail_key,
                                self.tenant_id,
                                image_info.url  # Pass source URL for tracking
                            )
                    else:
                        raw_key, raw_url, metadata = save_raw_image_content_addressed(
                            image_bytes,
                            self.tenant_id,
                            image_info.url  # Pass source URL for tracking
                        )

                        if raw_key:
                            saved_raw_keys.append(raw_key)
                            # Update cache with metadata only
                            await self.cache_service.store_crawled_image(
                                image_info.url,
                                image_bytes,
                                raw_key,
                                None,
                                self.tenant_id,
                                image_info.url  # Pass source URL for tracking
                            )

        # Get detailed cache statistics
        cache_stats = await self.cache_service.get_cache_stats()

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
            asyncio.create_task(self._streaming_storage_worker(storage_queue, streaming_results))
        ]

        # Feed images into download queue
        for i, image_info in enumerate(images, 1):
            await download_queue.put((image_info, i, len(images)))

        # Signal completion to all queues
        await download_queue.put(None)  # Sentinel value

        # Wait for all workers to complete
        await asyncio.gather(*workers, return_exceptions=True)

        # Get detailed cache statistics
        cache_stats = await self.cache_service.get_cache_stats()

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

    async def _streaming_storage_worker(self, storage_queue: asyncio.Queue, results: dict):
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
                        await self._process_storage_batch(storage_batch, results)
                    break

                storage_batch.append(item)

                # Process batch when full
                if len(storage_batch) >= batch_size:
                    await self._process_storage_batch(storage_batch, results)
                    storage_batch = []

            except asyncio.TimeoutError:
                # Process any pending items on timeout
                if storage_batch:
                    await self._process_storage_batch(storage_batch, results)
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

    async def _process_storage_batch(self, batch: List, results: dict):
        """Process a batch of items for storage."""
        if not batch:
            return

        # Process batch concurrently
        batch_tasks = []
        for image_bytes, thumbnail_bytes, image_info in batch:
            batch_tasks.append(self._store_single_item(image_bytes, thumbnail_bytes, image_info, results))

        await asyncio.gather(*batch_tasks, return_exceptions=True)

    async def _store_single_item(self, image_bytes: bytes, thumbnail_bytes: Optional[bytes], image_info: 'ImageInfo', results: dict):
        """Store a single item and update cache."""
        try:
            async with self._storage_semaphore:
                # Store using content-addressed keys
                if thumbnail_bytes is not None:
                    raw_key, raw_url, thumbnail_key, thumb_url, metadata = await save_raw_and_thumb_content_addressed_async(
                        image_bytes,
                        thumbnail_bytes,
                        self.tenant_id,
                        image_info.url  # Pass source URL for tracking
                    )

                    if raw_key:
                        results['saved_raw_keys'].append(raw_key)
                        results['saved_thumbnail_keys'].append(thumbnail_key)
                        # Update cache with metadata only
                        await self.cache_service.store_crawled_image(
                            image_info.url,
                            image_bytes,
                            raw_key,
                            thumbnail_key,
                            self.tenant_id,
                            image_info.url  # Pass source URL for tracking
                        )
                else:
                    raw_key, raw_url, metadata = save_raw_image_content_addressed(
                        image_bytes,
                        self.tenant_id,
                        image_info.url  # Pass source URL for tracking
                    )

                    if raw_key:
                        results['saved_raw_keys'].append(raw_key)
                        # Update cache with metadata only
                        await self.cache_service.store_crawled_image(
                            image_info.url,
                            image_bytes,
                            raw_key,
                            None,
                            self.tenant_id,
                            image_info.url  # Pass source URL for tracking
                        )

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

                # Process found images with expanded source detection
                for img in imgs:
                    img_urls = self._extract_img_urls(img, base_url)
                    for url in img_urls:
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            # Create a mock img tag with the extracted URL
                            class MockImg:
                                def __init__(self, src_url):
                                    self.src_url = src_url
                                def get(self, attr, default=''):
                                    return self.src_url if attr == 'src' else default
                            mock_img = MockImg(url)
                            all_images.append(mock_img)

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

            images.append(ImageInfo(
                url=absolute_url,
                alt_text=alt_text,
                title=title,
                width=width,
                height=height
            ))

        return images

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
        Run face detection in thread pool to avoid blocking async loop.
        """
        def _run_face_detection():
            """Synchronous face detection function to run in thread pool."""
            try:
                face_service = get_face_service()

                # Enhance image for better face detection
                enhanced_bytes, enhancement_scale = face_service.enhance_image_for_face_detection(image_bytes)
                faces = face_service.detect_and_embed(enhanced_bytes, enhancement_scale, min_size=0)

                if self.crop_faces and faces:
                    # Use the first face for thumbnail creation
                    thumbnail_bytes = face_service.crop_face_and_create_thumbnail(
                        image_bytes, faces[0], self.face_margin
                    )
                elif self.crop_faces and not faces:
                    # Don't create thumbnail for images without faces when CROP_FACES=true
                    # Only crop faces that are actually detected
                    thumbnail_bytes = None
                # If CROP_FACES=false, no thumbnail is created regardless of face detection

                return faces, thumbnail_bytes
            except Exception as e:
                logger.error(f"Error in face detection thread: {e}")
                return [], None

        # Run face detection in thread pool
        loop = asyncio.get_event_loop()
        faces, thumbnail_bytes = await loop.run_in_executor(
            self._face_detection_thread_pool, _run_face_detection
        )

        return faces, thumbnail_bytes

    async def _process_single_image(self, image_info: ImageInfo, index: int, total: int) -> Tuple[Optional[str], Optional[str], bool, List[str]]:
        """
        Process a single image with memory management and multi-tenancy support.
        """
        # Track this task for memory management
        task = asyncio.current_task()
        if task:
            self._active_tasks.add(task)

        try:
            async with self._processing_semaphore:
                # Proactive memory management
                self._manage_memory_pressure()

                logger.info(f"Processing image {index}/{total}: {_truncate_log_string(image_info.url)}")

                # Download the image
                image_bytes, download_errors = await self.download_image(image_info)
                if not image_bytes:
                    logger.warning(f"Failed to download image: {_truncate_log_string(image_info.url)}")
                    return None, None, False, download_errors, []

                # Check image dimensions - skip if any dimension is below 100px
                if not self._check_image_dimensions(image_bytes, min_dimension=100):
                    logger.info(f"Image dimensions too small, skipping: {_truncate_log_string(image_info.url)}")
                    return None, None, False, download_errors + ["Image dimensions below minimum threshold"], []

                # Check cache
                should_skip, cached_key = await self.cache_service.should_skip_crawled_image(image_info.url, image_bytes, self.tenant_id)
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
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use 'async with' context manager.")

        errors = []

        try:
            # Validate URL security first
            is_safe, reason = validate_url_security(url)
            if not is_safe:
                logger.warning(f"Page URL rejected: {reason} - {_truncate_log_string(url)}")
                errors.append(f"Page URL rejected: {reason}")
                return None, errors

            logger.info(f"Fetching page: {_truncate_log_string(url)}")

            # Use redirect utility for page fetching
            content, fetch_reason = await fetch_html_with_redirects(url, self.session, max_hops=self.max_redirects)

            if content is None:
                logger.warning(f"Failed to fetch page: {fetch_reason} - {_truncate_log_string(url)}")
                errors.append(f"Fetch failed: {fetch_reason}")
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
            self.cache_service.reset_cache_stats()

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
        images, method_used = self.extract_images_by_method(html_content, url, method)

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
        if len(images_to_process) <= 10:
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

        while urls_to_visit and len(all_saved_thumbnail_keys) < self.max_total_images and pages_crawled < self.max_pages:
            # Get next URL to visit
            current_url, current_depth = urls_to_visit.pop(0)

            # Skip if already visited or depth limit exceeded
            if current_url in visited_urls or current_depth > self.max_depth:
                continue

            visited_urls.add(current_url)
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
            end_time=end_time
        )

        logger.info(f"Site crawl completed in {total_duration:.2f} seconds - Pages: {pages_crawled}, Found: {total_images_found}, Raw Images Saved: {total_raw_saved}, Thumbnails Saved: {total_thumbnails_saved}")
        return result
