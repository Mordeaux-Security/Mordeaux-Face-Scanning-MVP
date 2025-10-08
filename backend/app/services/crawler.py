"""
Enhanced Image Crawler Service v2

Integrated crawler service that combines the advanced features from basic_crawler1.1
with the commercial readiness features from main branch.

Key improvements:
- Advanced caching with similarity detection (from basic_crawler1.1)
- Multi-tenancy support (from main)
- Enhanced face detection with image enhancement (from basic_crawler1.1)
- Flexible CSS selector patterns (from basic_crawler1.1)
- Concurrent processing with semaphores (from basic_crawler1.1)
- Audit logging and compliance features (from main)
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
from pathlib import Path
from typing import List, Optional, Tuple, Set, Dict
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass
from datetime import datetime

import httpx
from bs4 import BeautifulSoup

from .storage import save_raw_and_thumb_with_precreated_thumb, save_raw_image_only, save_audit_log
from . import storage
from .face import get_face_service
from . import face
from .cache import get_hybrid_cache_service
from urllib.parse import urljoin, urlparse

class MemoryMonitor:
    """Enhanced memory monitoring with adaptive thresholds."""
    
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


@dataclass
class ImageInfo:
    """Information about a discovered image."""
    url: str
    alt_text: str
    title: str
    width: Optional[int]
    height: Optional[int]


class EnhancedImageCrawlerV2:
    """
    Enhanced image crawler v2 that combines advanced features from basic_crawler1.1
    with commercial readiness features from main branch.
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
        max_concurrent_images: int = 20,  # Increased from 10 based on test results
        batch_size: int = 50,
        enable_audit_logging: bool = True,
    ):
        self.tenant_id = tenant_id  # Multi-tenancy support from main
        self.max_file_size = max_file_size
        self.allowed_extensions = allowed_extensions or {
            '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'
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
        self.enable_audit_logging = enable_audit_logging  # Audit logging support from main
        
        # PHASE 2 OPTIMIZATION: Enhanced concurrency control
        self._processing_semaphore = asyncio.Semaphore(self.max_concurrent_images)
        self._storage_semaphore = asyncio.Semaphore(self.max_concurrent_images)
        self._download_semaphore = asyncio.Semaphore(min(self.max_concurrent_images * 2, 50))  # Allow more downloads
        
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
        
        # MEMORY MANAGEMENT: Enhanced memory monitoring
        self._memory_monitor = MemoryMonitor()
        self._active_tasks = weakref.WeakSet()  # Track active tasks for memory cleanup
        self._memory_pressure_threshold = 75  # Conservative threshold
        self._gc_frequency = 10  # Force GC every N operations
        self._operation_count = 0
        
        self.session: Optional[httpx.AsyncClient] = None
        self.cache_service = get_hybrid_cache_service()  # Advanced caching from basic_crawler1.1
        self.pending_cache_entries = []  # Batch cache writes
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = httpx.AsyncClient(
            timeout=self.timeout,
            follow_redirects=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.aclose()
        
        # PHASE 2 OPTIMIZATION: Clean up thread pools
        self._face_detection_thread_pool.shutdown(wait=True)
        self._storage_thread_pool.shutdown(wait=True)
    
    async def fetch_page(self, url: str) -> Tuple[Optional[str], List[str]]:
        """Fetch a web page and return its content and any errors."""
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use 'async with' context manager.")
            
        errors = []
        
        try:
            logger.info(f"Fetching page: {url}")
            response = await self.session.get(url)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type:
                errors.append(f"Content type '{content_type}' is not HTML")
                return None, errors
                
            return response.text, errors
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error fetching {url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None, errors
        except Exception as e:
            error_msg = f"Unexpected error fetching {url}: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            return None, errors
    
    def extract_images_by_method(self, html_content: str, base_url: str, method: str = "smart") -> Tuple[List[ImageInfo], str]:
        """
        Extract images using configurable targeting methods.
        
        Enhanced version from basic_crawler1.1 with flexible CSS selector patterns.
        
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
                
                logger.debug(f"DEBUG: Starting smart method extraction for URL: {base_url}")
                for pattern_name, selectors in patterns:
                    logger.debug(f"DEBUG: Trying pattern: {pattern_name}")
                    images = self._extract_with_selectors(soup, base_url, selectors)
                    logger.debug(f"DEBUG: Pattern {pattern_name} found {len(images)} images")
                    if images:
                        method_used = pattern_name
                        logger.info(f"Smart method selected: {pattern_name} (found {len(images)} images)")
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
            
            logger.info(f"Found {len(images)} images using method: {method_used}")
                
        except Exception as e:
            logger.error(f"Error parsing HTML content: {str(e)}")
            
        return images, method_used
    
    def _extract_with_selectors(self, soup, base_url: str, selectors: List[Dict]) -> List[ImageInfo]:
        """
        Extract images using CSS selectors with flexible matching.
        
        Enhanced version from basic_crawler1.1 with robust selector handling.
        
        Args:
            soup: BeautifulSoup parsed HTML
            base_url: Base URL for resolving relative URLs
            selectors: List of selector dictionaries with 'selector' and 'description' keys
            
        Returns:
            List of ImageInfo objects
        """
        all_images = []
        seen_urls = set()  # Avoid duplicates
        
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
                            # Just attribute exists (e.g., "data-mediumthumb")
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
                
                # Process found images
                for img in imgs:
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src and src not in seen_urls:
                        seen_urls.add(src)
                        all_images.append(img)
                        
            except Exception as e:
                logger.warning(f"Error with selector '{selector_config}': {str(e)}")
                continue
        
        logger.debug(f"Found {len(all_images)} unique images using {len(selectors)} selectors")
        return self._process_img_tags(all_images, base_url)
    
    def add_custom_extraction_pattern(self, pattern_name: str, selectors: List[Dict], priority: int = 100):
        """
        Add a custom extraction pattern for specific sites.
        
        Args:
            pattern_name: Name of the pattern (e.g., 'xvideos-thumbs')
            selectors: List of selector dictionaries
            priority: Lower numbers = higher priority (default: 100)
            
        Example:
            crawler.add_custom_extraction_pattern('xvideos-thumbs', [
                {"selector": ".video-block img", "description": "xvideos video block images"},
                {"selector": "img[data-src*='thumb']", "description": "lazy-loaded thumbnails"}
            ])
        """
        # This could be extended to store custom patterns and inject them into the smart method
        # For now, this is a placeholder for future extensibility
        logger.info(f"Custom pattern '{pattern_name}' registered with {len(selectors)} selectors (priority: {priority})")
    
    def _process_img_tags(self, img_tags, base_url: str) -> List[ImageInfo]:
        """Process a list of img tags and return ImageInfo objects."""
        images = []
        
        for img in img_tags:
            src = img.get('src')
            if not src:
                continue
                
            # Resolve relative URLs
            absolute_url = urljoin(base_url, src)
            
            # Extract image metadata
            alt_text = img.get('alt', '')
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
        """Download an image from its URL with enhanced concurrency control."""
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use 'async with' context manager.")
        
        # PHASE 2 OPTIMIZATION: Use separate download semaphore for better throughput
        async with self._download_semaphore:
            errors = []
            
            try:
                logger.info(f"Downloading image: {image_info.url}")
            
                # Check file extension
                parsed_url = urlparse(image_info.url)
                path = parsed_url.path.lower()
                if not any(path.endswith(ext) for ext in self.allowed_extensions):
                    # Try to get content type from HEAD request
                    head_response = await self.session.head(image_info.url)
                    content_type = head_response.headers.get('content-type', '').lower()
                    if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
                        errors.append(f"File extension not in allowed list: {path}")
                        return None, errors
                
                # Download the image
                response = await self.session.get(image_info.url)
                response.raise_for_status()
                
                # Check file size
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > self.max_file_size:
                    errors.append(f"File too large: {content_length} bytes")
                    return None, errors
                    
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'png', 'gif', 'webp']):
                    errors.append(f"Content type not an image: {content_type}")
                    return None, errors
                
                return response.content, errors
                
            except httpx.HTTPError as e:
                error_msg = f"HTTP error downloading {image_info.url}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                return None, errors
            except Exception as e:
                error_msg = f"Unexpected error downloading {image_info.url}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                return None, errors
    
    async def _async_face_detection(self, image_bytes: bytes, image_info: ImageInfo) -> Tuple[List, Optional[bytes]]:
        """
        PHASE 2 OPTIMIZATION: Run face detection in thread pool to avoid blocking async loop.
        """
        def _run_face_detection():
            """Synchronous face detection function to run in thread pool."""
            try:
                face_service = get_face_service()
                
                # Enhance image first, then detect faces
                enhanced_bytes, enhancement_scale = face_service.enhance_image_for_face_detection(image_bytes)
                faces = face_service.detect_and_embed(enhanced_bytes, enhancement_scale, min_size=0)
                
                if self.crop_faces and faces:
                    # For now, just take the first face for thumbnail
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
    
    def _adjust_concurrency_dynamically(self):
        """
        MEMORY-SAFE DYNAMIC CONCURRENCY: Adjust concurrency based on memory pressure and system resources.
        """
        try:
            # Get comprehensive system metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
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
        
        # Periodic garbage collection
        if self._operation_count % self._gc_frequency == 0:
            gc.collect()
        
        # Check for memory pressure and adjust if needed
        memory_status = self._memory_monitor.get_memory_status()
        if memory_status['pressure_level'] in ['high', 'critical']:
            # Force cleanup of completed tasks
            self._cleanup_completed_tasks()
            
            # Trigger immediate GC
            gc.collect()
            
            logger.debug(f"Memory pressure management: {memory_status['pressure_level']} "
                        f"({memory_status['percent']:.1f}% used)")
    
    def _cleanup_completed_tasks(self):
        """Clean up references to completed tasks."""
        # The WeakSet automatically removes completed tasks
        # This is mainly for logging and monitoring
        active_count = len(self._active_tasks)
        if active_count > 0:
            logger.debug(f"Active tasks: {active_count}")
    
    async def _async_storage_operation(self, storage_func, *args, **kwargs):
        """
        MEMORY-SAFE ASYNC STORAGE: Run storage operations with memory management.
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
            del args, kwargs  # Help GC
            gc.collect()
            
            return result
            
        except Exception as e:
            logger.error(f"Storage operation failed: {e}")
            raise
        finally:
            # Remove task from tracking when complete
            if task and task in self._active_tasks:
                self._active_tasks.discard(task)
    
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
        Check if a URL is valid for crawling.
        
        Args:
            url: URL to check
            base_url: Base URL for domain comparison
            
        Returns:
            True if URL is valid for crawling
        """
        try:
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
                '.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg',
                '.pdf', '.doc', '.docx', '.zip', '.rar',
                'facebook.com', 'twitter.com', 'instagram.com', 'youtube.com'
            ]
            
            url_lower = url.lower()
            for pattern in skip_patterns:
                if pattern in url_lower:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating URL {url}: {str(e)}")
            return False
    
    def check_face_quality(self, image_bytes: bytes) -> Tuple[bool, str, dict]:
        """
        Check if the image contains high-quality faces suitable for recognition.
        Uses enhanced detection methods for better accuracy with low-resolution images.
        
        Args:
            image_bytes: The image data to analyze
            
        Returns:
            Tuple of (is_high_quality, reason, best_face)
        """
        try:
            face_service = get_face_service()
            
            # Use enhanced multi-scale face detection (handles enhancement internally)
            enhanced_bytes, enhancement_scale = face_service.enhance_image_for_face_detection(image_bytes)
            faces = face_service.detect_and_embed(enhanced_bytes, enhancement_scale, min_size=25)
            
            if not faces:
                if self.require_face:
                    return False, "No faces detected", {}
                else:
                    return True, "No faces required", {}
            
            # Check each face for quality
            high_quality_faces = []
            for face in faces:
                det_score = face.get('det_score', 0.0)
                if det_score >= self.min_face_quality:
                    high_quality_faces.append(face)
            
            if not high_quality_faces:
                return False, f"No faces meet quality threshold (min: {self.min_face_quality})", {}
            
            # Find the best face (highest quality score and largest size)
            best_face = None
            best_score = 0
            for face in high_quality_faces:
                bbox = face.get('bbox', [0, 0, 0, 0])
                x1, y1, x2, y2 = bbox
                face_width = x2 - x1
                face_height = y2 - y1
                
                # Skip faces that are too small (less than 50x50 pixels)
                if face_width < 50 or face_height < 50:
                    continue
                
                # Score based on detection confidence and face size
                face_score = face.get('det_score', 0.0) + (face_width * face_height) / 100000.0
                if face_score > best_score:
                    best_score = face_score
                    best_face = face
            
            if best_face:
                return True, f"Found {len(high_quality_faces)} high-quality faces", best_face
            else:
                return False, "All faces are too small for reliable recognition", {}
            
        except Exception as e:
            logger.error(f"Error checking face quality: {str(e)}")
            if self.require_face:
                return False, f"Face quality check failed: {str(e)}", {}
            else:
                return True, f"Face quality check failed, but not required: {str(e)}", {}

    
    async def _process_single_image(self, image_info: ImageInfo, index: int, total: int) -> Tuple[Optional[str], Optional[str], bool, List[str]]:
        """
        MEMORY-SAFE IMAGE PROCESSING: Process with enhanced memory management.
        
        Enhanced version from basic_crawler1.1 with multi-tenancy support.
        """
        # Track this task for memory management
        task = asyncio.current_task()
        if task:
            self._active_tasks.add(task)
        
        try:
            async with self._processing_semaphore:
                # Proactive memory management
                self._manage_memory_pressure()
                
                logger.info(f"Processing image {index}/{total}: {image_info.url}")

                # Download the image
                image_bytes, download_errors = await self.download_image(image_info)
                if not image_bytes:
                    logger.warning(f"Failed to download image: {image_info.url}")
                    return None, None, False, download_errors, []

                # Check cache
                should_skip, cached_key = await self.cache_service.should_skip_crawled_image(image_info.url, image_bytes, self.tenant_id)
                if should_skip and cached_key:
                    logger.info(f"Image {image_info.url} found in cache. Key: {cached_key}")
                    return cached_key, cached_key, True, download_errors, []

                # PHASE 2 OPTIMIZATION: Use async face detection with thread pool
                faces = []
                thumbnail_bytes = None
                if self.require_face or self.crop_faces:
                    faces, thumbnail_bytes = await self._async_face_detection(image_bytes, image_info)
                    
                    if self.require_face and not faces:
                        logger.info(f"No faces detected for {image_info.url}, skipping.")
                        return None, None, False, download_errors, []
                    
                    # When crop_faces=true but no faces detected, don't create thumbnails
                    if self.crop_faces and not faces:
                        logger.info(f"No faces detected for {image_info.url}, no thumbnail created (crop_faces=true).")
                # Note: When REQUIRE_FACE=false and CROP_FACES=false, no thumbnail is created

                # Prepare data for batch storage
                return image_bytes, thumbnail_bytes, False, download_errors, faces

        except Exception as e:
            logger.error(f"Error processing image {image_info.url}: {e}", exc_info=True)
            return None, None, False, [f"Processing error: {e}"], []
        finally:
            # Remove task from tracking when complete
            if task and task in self._active_tasks:
                self._active_tasks.discard(task)

    async def crawl_page(self, url: str, method: str = "smart") -> CrawlResult:
        """Crawl a single page for images using the specified method."""
        logger.info(f"Starting crawl of: {url} using method: {method} (tenant: {self.tenant_id}, min_face_quality: {self.min_face_quality}, require_face: {self.require_face})")
        
        # Reset cache statistics for this crawl
        self.cache_service.reset_cache_stats()
        
        saved_raw_keys = []
        saved_thumbnail_keys = []
        all_errors = []
        cache_hits = 0
        cache_misses = 0
        
        # MEMORY-SAFE DYNAMIC CONCURRENCY: Enable with enhanced memory management
        self._adjust_concurrency_dynamically()
        
        # Fetch the page
        html_content, fetch_errors = await self.fetch_page(url)
        all_errors.extend(fetch_errors)
        
        if not html_content:
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
                tenant_id=self.tenant_id
            )
        
        # Extract images using specified method
        images, method_used = self.extract_images_by_method(html_content, url, method)
        
        if not images:
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
                tenant_id=self.tenant_id
            )
        
        # Process ALL images found (no artificial per-page limits)
        images_to_process = images
        logger.info(f"Processing all {len(images_to_process)} images found")
        
        processing_tasks = []
        for i, image_info in enumerate(images_to_process, 1):
            processing_tasks.append(self._process_single_image(image_info, i, len(images_to_process)))

        processed_results = await asyncio.gather(*processing_tasks, return_exceptions=True)

        storage_batch = []
        for i, result in enumerate(processed_results):
            image_info = images_to_process[i]
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
                    storage_batch.append({
                        "image_bytes": image_bytes,
                        "thumbnail_bytes": thumbnail_bytes,
                        "image_info": image_info,
                        "faces": faces,
                        "key_prefix": "",
                        "save_type": "both" if thumbnail_bytes is not None else "raw_only"
                    })

            # Process batch if size reached or it's the last image
            if len(storage_batch) >= self.batch_size or (i == len(processed_results) - 1 and storage_batch):
                logger.info(f"Saving batch of {len(storage_batch)} images to storage.")
                
                # Save images individually but concurrently
                # PHASE 2 OPTIMIZATION: Use async storage operations with thread pool
                async def save_single_item(item):
                    async with self._storage_semaphore:
                        try:
                            if item["thumbnail_bytes"] is not None:
                                raw_key, raw_url, thumbnail_key, thumb_url = await self._async_storage_operation(
                                    storage.save_raw_and_thumb_with_precreated_thumb,
                                    item["image_bytes"], 
                                    item["thumbnail_bytes"],
                                    self.tenant_id  # Multi-tenancy support
                                )
                            else:
                                raw_key, raw_url = await self._async_storage_operation(
                                    storage.save_raw_image_only,
                                    item["image_bytes"], 
                                    self.tenant_id
                                )
                                thumbnail_key = None
                            
                            if raw_key:
                                await self.cache_service.store_crawled_image(
                                    item["image_info"].url,
                                    item["image_bytes"],
                                    raw_key,
                                    thumbnail_key,
                                    self.tenant_id
                                )
                                return raw_key, thumbnail_key, None
                            else:
                                return None, None, "Failed to save image"
                        except Exception as e:
                            return None, None, str(e)
                
                # Process all items in the batch concurrently
                batch_tasks = [save_single_item(item) for item in storage_batch]
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for j, result in enumerate(batch_results):
                    if isinstance(result, Exception):
                        logger.warning(f"Failed to save image {storage_batch[j]['image_info'].url} in batch: {result}")
                        all_errors.append(f"Batch save error for {storage_batch[j]['image_info'].url}: {result}")
                    else:
                        raw_key, thumbnail_key, error = result
                        if raw_key:
                            saved_raw_keys.append(raw_key)
                            if thumbnail_key:
                                saved_thumbnail_keys.append(thumbnail_key)
                        else:
                            logger.warning(f"Failed to save image {storage_batch[j]['image_info'].url} in batch: {error}")
                            all_errors.append(f"Batch save error for {storage_batch[j]['image_info'].url}: {error}")
                storage_batch = []  # Clear batch after saving
        
        # Create audit log entry if enabled
        if self.enable_audit_logging:
            try:
                audit_data = {
                    "tenant_id": self.tenant_id,
                    "url": url,
                    "method": method_used,
                    "images_found": len(images),
                    "raw_images_saved": len(saved_raw_keys),
                    "thumbnails_saved": len(saved_thumbnail_keys),
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "timestamp": datetime.utcnow().isoformat(),
                    "errors": all_errors
                }
                import json
                audit_bytes = json.dumps(audit_data, indent=2).encode('utf-8')
                audit_key, audit_url = save_audit_log(audit_bytes, self.tenant_id, "crawl")
                logger.info(f"Audit log saved: {audit_key}")
            except Exception as e:
                logger.warning(f"Failed to save audit log: {e}")
        
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
            cache_hits=cache_stats['total_hits'],
            cache_misses=cache_stats['cache_misses'],
            redis_hits=cache_stats['redis_hits'],
            postgres_hits=cache_stats['postgres_hits'],
            tenant_id=self.tenant_id
        )
        
        logger.info(f"Crawl completed - Found: {result.images_found}, Raw Images Saved: {result.raw_images_saved}, Thumbnails Saved: {result.thumbnails_saved}")
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
        logger.info(f"Starting site crawl from: {start_url} (tenant: {self.tenant_id}, max_images: {self.max_total_images}, max_pages: {self.max_pages})")
        
        # Initialize crawling state
        visited_urls = set()
        urls_to_visit = [start_url]
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
            current_url = urls_to_visit.pop(0)
            
            # Skip if already visited
            if current_url in visited_urls:
                continue
            
            visited_urls.add(current_url)
            pages_crawled += 1
            
            logger.info(f"Crawling page {pages_crawled}/{self.max_pages}: {current_url}")
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
                if len(all_saved_thumbnail_keys) < self.max_total_images and len(urls_to_visit) < self.max_pages * 2:
                    # Fetch page content for URL discovery
                    html_content, fetch_errors = await self.fetch_page(current_url)
                    all_errors.extend(fetch_errors)
                    
                    if html_content:
                        # Extract new URLs
                        new_urls = self.extract_page_urls(html_content, current_url)
                        
                        # Add new URLs to visit queue (prioritize unseen URLs)
                        for url in new_urls:
                            if url not in visited_urls and url not in urls_to_visit:
                                urls_to_visit.append(url)
                        
                        logger.info(f"Found {len(new_urls)} new URLs to explore")
                
                # Check if we've reached our goals
                if len(all_saved_thumbnail_keys) >= self.max_total_images:
                    logger.info(f"Reached target of {self.max_total_images} thumbnails, stopping crawl")
                    break
                    
            except Exception as e:
                error_msg = f"Error crawling page {current_url}: {str(e)}"
                logger.error(error_msg)
                all_errors.append(error_msg)
                continue
        
        # Create aggregated result
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
            tenant_id=self.tenant_id
        )
        
        logger.info(f"Site crawl completed - Pages: {pages_crawled}, Found: {total_images_found}, Raw Images Saved: {total_raw_saved}, Thumbnails Saved: {total_thumbnails_saved}")
        return result


# Convenience function for easy integration
async def crawl_images_from_url(
    url: str, 
    tenant_id: str = "default",
    method: str = "smart"
) -> CrawlResult:
    """
    Convenience function to crawl images from a URL.
    
    Args:
        url: The URL to crawl
        tenant_id: Tenant ID for multi-tenancy support
        method: Targeting method ('smart', 'data-mediumthumb', 'js-videoThumb', etc.)
        
    Returns:
        CrawlResult with crawl statistics
    """
    async with EnhancedImageCrawlerV2(tenant_id=tenant_id) as crawler:
        return await crawler.crawl_page(url, method)
