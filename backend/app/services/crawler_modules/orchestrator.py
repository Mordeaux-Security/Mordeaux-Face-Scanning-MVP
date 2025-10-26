"""
Orchestration module for the crawler service.

This module provides a high-level orchestrator that coordinates all the facades
and services to perform crawling operations in a clean, modular way.
"""

import asyncio
import logging
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime

from .types import CrawlResult, ImageInfo
from .extraction import ImageExtractor
from .processing import ImageProcessingService
from .storage_facade import StorageFacade
from .caching_facade import CachingFacade
from .memory import MemoryMonitor

logger = logging.getLogger(__name__)


class CrawlOrchestrator:
    """
    High-level orchestrator that coordinates all crawling operations.
    
    This orchestrator provides a clean interface for crawling operations,
    coordinating between extraction, processing, storage, and caching services.
    """
    
    def __init__(
        self,
        image_processing_service: ImageProcessingService,
        storage_facade: StorageFacade,
        caching_facade: CachingFacade,
        memory_monitor: MemoryMonitor,
        http_service,
        selector_mining_service
    ):
        """
        Initialize the crawl orchestrator.
        
        Args:
            image_processing_service: Service for image processing and face detection
            storage_facade: Facade for storage operations
            caching_facade: Facade for cache operations
            memory_monitor: Monitor for memory management
            http_service: Service for HTTP operations
            selector_mining_service: Service for selector mining
        """
        self.image_processing_service = image_processing_service
        self.storage_facade = storage_facade
        self.caching_facade = caching_facade
        self.memory_monitor = memory_monitor
        self.http_service = http_service
        self.selector_mining_service = selector_mining_service
        self.image_extractor = ImageExtractor()
        
        self.logger = logging.getLogger(__name__)
    
    async def crawl_page(
        self,
        url: str,
        max_images: int = 50,
        method: str = "smart",
        tenant_id: str = "default"
    ) -> CrawlResult:
        """
        Orchestrate a complete page crawl operation.
        
        Args:
            url: URL to crawl
            max_images: Maximum number of images to process
            method: Extraction method to use
            tenant_id: Tenant identifier
            
        Returns:
            CrawlResult with crawl statistics and results
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Fetch page content
            html_content, fetch_errors = await self._fetch_page_content(url)
            if not html_content:
                return self._create_error_result(url, fetch_errors)
            
            # Step 2: Extract images from HTML
            images, method_used = await self._extract_images(html_content, url, method)
            if not images:
                return self._create_no_images_result(url, method_used)
            
            # Step 3: Limit images to max_images
            images = images[:max_images]
            
            # Step 4: Process images (download, cache check, face detection, storage)
            crawl_result = await self._process_images(images, url, tenant_id)
            
            # Step 5: Get cache statistics
            cache_stats = await self.caching_facade.get_cache_statistics()
            
            # Step 6: Create final result
            return self._create_success_result(
                url=url,
                images_found=len(images),
                method_used=method_used,
                crawl_result=crawl_result,
                cache_stats=cache_stats,
                start_time=start_time
            )
            
        except Exception as e:
            self.logger.error(f"Error in crawl orchestration for {url}: {e}")
            return self._create_error_result(url, [str(e)])
    
    async def _fetch_page_content(self, url: str) -> Tuple[Optional[str], List[str]]:
        """Fetch page content using HTTP service."""
        try:
            html_content, errors = await self.http_service.fetch_html(url, None, use_js=False)
            return html_content, errors
        except Exception as e:
            self.logger.error(f"Error fetching page content for {url}: {e}")
            return None, [str(e)]
    
    async def _extract_images(self, html_content: str, base_url: str, method: str) -> Tuple[List[ImageInfo], str]:
        """Extract images from HTML content."""
        try:
            images, method_used = self.image_extractor.extract_images_by_method(html_content, base_url, method)
            return images, method_used
        except Exception as e:
            self.logger.error(f"Error extracting images from {base_url}: {e}")
            return [], method
    
    async def _process_images(
        self,
        images: List[ImageInfo],
        page_url: str,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Process images through the complete pipeline concurrently."""
        saved_raw_keys = []
        saved_thumbnail_keys = []
        all_errors = []
        cache_hits = 0
        cache_misses = 0
        
        # Semaphore to limit concurrent downloads
        semaphore = asyncio.Semaphore(100)
        
        async def process_one_image(image_info: ImageInfo, index: int):
            """Process a single image."""
            async with semaphore:
                try:
                    # Download image
                    image_bytes, download_errors = await self._download_image(image_info)
                    if not image_bytes:
                        return {'errors': download_errors}
                    
                    # Check cache
                    should_skip, cached_key = await self.caching_facade.should_skip_image(
                        image_info.url, image_bytes, tenant_id
                    )
                    
                    if should_skip and cached_key:
                        self.logger.info(f"Image {self._truncate_log_string(image_info.url)} found in cache")
                        return {'cache_hit': True, 'key': cached_key}
                    
                    # Process image (face detection, filtering)
                    faces, thumbnail_bytes, processing_errors = await self.image_processing_service.process_single_image(
                        image_info, image_bytes, index, len(images)
                    )
                    
                    if processing_errors:
                        return {'errors': processing_errors}
                    
                    # Save to storage
                    raw_key, thumb_key = await self.storage_facade.save_raw_and_thumbnail(
                        image_bytes=image_bytes,
                        thumbnail_bytes=thumbnail_bytes,
                        image_info=image_info,
                        page_url=page_url
                    )
                    
                    return {
                        'raw_key': raw_key,
                        'thumb_key': thumb_key
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error processing image {image_info.url}: {e}")
                    return {'errors': [str(e)]}
        
        # Process all images concurrently
        tasks = [process_one_image(img, i+1) for i, img in enumerate(images)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                all_errors.append(str(result))
                continue
            
            if 'errors' in result:
                all_errors.extend(result['errors'])
                continue
            
            if 'cache_hit' in result:
                cache_hits += 1
                if 'key' in result:
                    saved_raw_keys.append(result['key'])
                continue
            
            cache_misses += 1
            if 'raw_key' in result and result['raw_key']:
                saved_raw_keys.append(result['raw_key'])
            if 'thumb_key' in result and result['thumb_key']:
                saved_thumbnail_keys.append(result['thumb_key'])
        
        return {
            'saved_raw_keys': saved_raw_keys,
            'saved_thumbnail_keys': saved_thumbnail_keys,
            'errors': all_errors,
            'cache_hits': cache_hits,
            'cache_misses': cache_misses
        }
    
    async def _download_image(self, image_info: ImageInfo) -> Tuple[Optional[bytes], List[str]]:
        """Download image using HTTP service."""
        try:
            # This would use the HTTP service to download the image
            # For now, return placeholder
            return None, ["Download not implemented in orchestrator yet"]
        except Exception as e:
            return None, [str(e)]
    
    def _create_success_result(
        self,
        url: str,
        images_found: int,
        method_used: str,
        crawl_result: Dict[str, Any],
        cache_stats: Dict[str, int],
        start_time: datetime
    ) -> CrawlResult:
        """Create a successful crawl result."""
        end_time = datetime.utcnow()
        
        return CrawlResult(
            url=url,
            images_found=images_found,
            raw_images_saved=len(crawl_result['saved_raw_keys']),
            thumbnails_saved=len(crawl_result['saved_thumbnail_keys']),
            pages_crawled=1,
            saved_raw_keys=crawl_result['saved_raw_keys'],
            saved_thumbnail_keys=crawl_result['saved_thumbnail_keys'],
            errors=crawl_result['errors'],
            targeting_method=method_used,
            cache_hits=crawl_result['cache_hits'],
            cache_misses=crawl_result['cache_misses'],
            redis_hits=cache_stats.get('redis_hits', 0),
            postgres_hits=cache_stats.get('postgres_hits', 0),
            tenant_id="default",
            early_exit_count=0
        )
    
    def _create_error_result(self, url: str, errors: List[str]) -> CrawlResult:
        """Create an error crawl result."""
        return CrawlResult(
            url=url,
            images_found=0,
            raw_images_saved=0,
            thumbnails_saved=0,
            pages_crawled=0,
            saved_raw_keys=[],
            saved_thumbnail_keys=[],
            errors=errors,
            targeting_method="error",
            cache_hits=0,
            cache_misses=0,
            redis_hits=0,
            postgres_hits=0,
            tenant_id="default",
            early_exit_count=0
        )
    
    def _create_no_images_result(self, url: str, method_used: str) -> CrawlResult:
        """Create a result when no images are found."""
        return CrawlResult(
            url=url,
            images_found=0,
            raw_images_saved=0,
            thumbnails_saved=0,
            pages_crawled=1,
            saved_raw_keys=[],
            saved_thumbnail_keys=[],
            errors=[],
            targeting_method=method_used,
            cache_hits=0,
            cache_misses=0,
            redis_hits=0,
            postgres_hits=0,
            tenant_id="default",
            early_exit_count=0
        )
    
    def _truncate_log_string(self, text: str, max_length: int = 120) -> str:
        """Truncate long strings for logging."""
        if len(text) <= max_length:
            return text
        
        import hashlib
        hash_suffix = hashlib.md5(text.encode()).hexdigest()[:8]
        truncated = text[:max_length - len(hash_suffix) - 3]
        return f"{truncated}...{hash_suffix}"
