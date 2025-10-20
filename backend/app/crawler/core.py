"""
Core Orchestration Engine

Main orchestration engine for crawling operations. Coordinates between
extraction, processing, storage, and mining components with proper
resource management and error handling.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
from urllib.parse import urlparse

from .config import CrawlerConfig, get_config
from .extractor import ImageExtractor, ExtractedImage
from .processor import ImageProcessor, ProcessedImage
from .storage import StorageManager, StorageResult
from .memory import MemoryManager, get_memory_manager
from .miner import SelectorMiner, MiningResult

logger = logging.getLogger(__name__)


@dataclass
class CrawlResult:
    """Result of a crawling operation."""
    url: str
    domain: str
    images_found: int
    images_processed: int
    faces_detected: int
    images_saved: int
    thumbnails_saved: int
    duration_seconds: float
    success: bool
    error: Optional[str] = None
    mining_attempted: bool = False
    mining_success: bool = False
    recipe_used: Optional[str] = None


@dataclass
class CrawlStats:
    """Statistics for crawling operations."""
    total_sites: int
    successful_sites: int
    failed_sites: int
    total_images_found: int
    total_images_processed: int
    total_faces_detected: int
    total_images_saved: int
    total_thumbnails_saved: int
    total_duration: float
    mining_attempts: int
    mining_successes: int


class CrawlerEngine:
    """
    Main orchestration engine for crawling operations.
    
    Coordinates between extraction, processing, storage, and mining components
    with proper resource management, error handling, and streaming capabilities.
    """
    
    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or get_config()
        self.memory_manager = get_memory_manager(self.config)
        
        # Component instances (lazy initialization)
        self.extractor: Optional[ImageExtractor] = None
        self.processor: Optional[ImageProcessor] = None
        self.storage: Optional[StorageManager] = None
        self.miner: Optional[SelectorMiner] = None
        
        # Crawling statistics
        self.stats = CrawlStats(
            total_sites=0,
            successful_sites=0,
            failed_sites=0,
            total_images_found=0,
            total_images_processed=0,
            total_faces_detected=0,
            total_images_saved=0,
            total_thumbnails_saved=0,
            total_duration=0.0,
            mining_attempts=0,
            mining_successes=0
        )
        
        # Active crawling state
        self._active_crawls: Dict[str, bool] = {}
        self._crawl_lock = asyncio.Lock()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_components()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
    
    async def _initialize_components(self):
        """Initialize all components."""
        try:
            # Initialize components
            self.extractor = ImageExtractor(self.config)
            self.processor = ImageProcessor(self.config, self.memory_manager)
            self.storage = StorageManager(self.config)
            self.miner = SelectorMiner(self.config)
            
            # Initialize storage
            await self.storage.initialize()
            
            # Start memory monitoring
            await self.memory_manager.start_monitoring()
            
            logger.info("Crawler engine initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize crawler engine: {e}")
            raise
    
    async def _cleanup(self):
        """Cleanup all components."""
        try:
            # Stop memory monitoring
            await self.memory_manager.stop_monitoring()
            
            # Cleanup components
            if self.extractor:
                await self.extractor.__aexit__(None, None, None)
            if self.processor:
                await self.processor.__aexit__(None, None, None)
            if self.storage:
                await self.storage.cleanup()
            if self.miner:
                await self.miner.__aexit__(None, None, None)
            
            # Cleanup memory manager
            await self.memory_manager.cleanup()
            
            logger.info("Crawler engine cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during crawler cleanup: {e}")
    
    async def crawl_site(self, url: str) -> CrawlResult:
        """
        Crawl a single site with full processing pipeline.
        
        Args:
            url: URL to crawl
            
        Returns:
            CrawlResult with crawling statistics and results
        """
        start_time = time.time()
        domain = self._extract_domain(url)
        
        # Check if already crawling this site
        async with self._crawl_lock:
            if url in self._active_crawls:
                return CrawlResult(
                    url=url,
                    domain=domain,
                    images_found=0,
                    images_processed=0,
                    faces_detected=0,
                    images_saved=0,
                    thumbnails_saved=0,
                    duration_seconds=time.time() - start_time,
                    success=False,
                    error="Site already being crawled"
                )
            
            self._active_crawls[url] = True
        
        try:
            logger.info(f"Starting crawl of {domain}")
            
            # Initialize components if not already done
            if not self.storage:
                await self._initialize_components()
            
            # Get or mine recipe
            recipe, mining_attempted, mining_success = await self._get_or_mine_recipe(url)
            if not recipe:
                return CrawlResult(
                    url=url,
                    domain=domain,
                    images_found=0,
                    images_processed=0,
                    faces_detected=0,
                    images_saved=0,
                    thumbnails_saved=0,
                    duration_seconds=time.time() - start_time,
                    success=False,
                    error="Failed to get or mine recipe",
                    mining_attempted=mining_attempted,
                    mining_success=mining_success
                )
            
            # Extract image URLs
            extraction_result = await self.extractor.extract_images(url, recipe)
            if not extraction_result.success:
                return CrawlResult(
                    url=url,
                    domain=domain,
                    images_found=0,
                    images_processed=0,
                    faces_detected=0,
                    images_saved=0,
                    thumbnails_saved=0,
                    duration_seconds=time.time() - start_time,
                    success=False,
                    error=f"Image extraction failed: {extraction_result.error}",
                    mining_attempted=mining_attempted,
                    mining_success=mining_success
                )
            
            # Process images in streaming batches
            processing_result = await self._process_images_streaming(
                extraction_result.images, url
            )
            
            # Update statistics
            self._update_stats(processing_result, mining_attempted, mining_success)
            
            duration = time.time() - start_time
            
            logger.info(f"Completed crawl of {domain}: {processing_result['images_saved']} images saved in {duration:.2f}s")
            
            return CrawlResult(
                url=url,
                domain=domain,
                images_found=len(extraction_result.images),
                images_processed=processing_result['images_processed'],
                faces_detected=processing_result['faces_detected'],
                images_saved=processing_result['images_saved'],
                thumbnails_saved=processing_result['thumbnails_saved'],
                duration_seconds=duration,
                success=True,
                mining_attempted=mining_attempted,
                mining_success=mining_success,
                recipe_used=recipe.get('method', 'unknown')
            )
            
        except Exception as e:
            logger.error(f"Error crawling {domain}: {e}")
            self.stats.failed_sites += 1
            
            return CrawlResult(
                url=url,
                domain=domain,
                images_found=0,
                images_processed=0,
                faces_detected=0,
                images_saved=0,
                thumbnails_saved=0,
                duration_seconds=time.time() - start_time,
                success=False,
                error=str(e)
            )
        
        finally:
            # Remove from active crawls
            async with self._crawl_lock:
                self._active_crawls.pop(url, None)
    
    async def crawl_site_streaming(self, url: str) -> AsyncIterator[ProcessedImage]:
        """
        Stream processed images as they're completed.
        
        Args:
            url: URL to crawl
            
        Yields:
            ProcessedImage objects as they're processed
        """
        try:
            # Get or mine recipe
            recipe, _, _ = await self._get_or_mine_recipe(url)
            if not recipe:
                return
            
            # Stream image extraction
            async for extracted_image in self.extractor.extract_images_streaming(url, recipe):
                # Process image immediately
                processed_image = await self.processor.process_single_image(
                    extracted_image.url, url, extracted_image.context
                )
                
                if processed_image:
                    # Store image
                    storage_result = await self._store_processed_image(processed_image, url)
                    if storage_result and storage_result.success:
                        yield processed_image
                
                # Check memory pressure
                if self.memory_manager.is_memory_pressured():
                    await self.memory_manager.force_gc("streaming_crawl")
        
        except Exception as e:
            logger.error(f"Error in streaming crawl of {url}: {e}")
    
    async def crawl_list(self, urls: List[str], max_concurrent: int = 2) -> List[CrawlResult]:
        """
        Crawl multiple sites with concurrency control.
        
        Args:
            urls: List of URLs to crawl
            max_concurrent: Maximum number of concurrent crawls
            
        Returns:
            List of CrawlResult objects
        """
        logger.info(f"Starting list crawl of {len(urls)} sites")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                return await self.crawl_site(url)
        
        # Process all sites
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        crawl_results = []
        for i, result in enumerate(results):
            if isinstance(result, CrawlResult):
                crawl_results.append(result)
            elif isinstance(result, Exception):
                logger.error(f"Error crawling {urls[i]}: {result}")
                crawl_results.append(CrawlResult(
                    url=urls[i],
                    domain=self._extract_domain(urls[i]),
                    images_found=0,
                    images_processed=0,
                    faces_detected=0,
                    images_saved=0,
                    thumbnails_saved=0,
                    duration_seconds=0.0,
                    success=False,
                    error=str(result)
                ))
        
        logger.info(f"List crawl completed: {len([r for r in crawl_results if r.success])} successful, {len([r for r in crawl_results if not r.success])} failed")
        
        return crawl_results
    
    async def _get_or_mine_recipe(self, url: str) -> tuple[Optional[Dict[str, Any]], bool, bool]:
        """Get existing recipe or mine new one."""
        mining_attempted = False
        mining_success = False
        
        # Check existing recipes first
        recipe = self.storage.get_recipe_for_url(url)
        if recipe:
            logger.debug(f"Using existing recipe for {self._extract_domain(url)}")
            return recipe, mining_attempted, mining_success
        
        # Mine new recipe if auto-mining is enabled
        if self.config.list_crawl_auto_selector_mining:
            logger.info(f"Mining selectors for {self._extract_domain(url)}")
            mining_attempted = True
            
            try:
                mining_result = await self.miner.mine_site(url)
                if mining_result.success and mining_result.candidates:
                    # Generate recipe from mining result
                    recipe = self._generate_recipe_from_mining(mining_result)
                    
                    # Save recipe
                    await self.storage.save_recipe(url, recipe)
                    
                    mining_success = True
                    logger.info(f"Successfully mined and saved recipe for {self._extract_domain(url)}")
                else:
                    logger.warning(f"Selector mining failed for {self._extract_domain(url)}")
                    recipe = self.storage._get_default_recipe()
            except Exception as e:
                logger.error(f"Error mining selectors for {self._extract_domain(url)}: {e}")
                recipe = self.storage._get_default_recipe()
        else:
            # Use default recipe
            recipe = self.storage._get_default_recipe()
        
        return recipe, mining_attempted, mining_success
    
    def _generate_recipe_from_mining(self, mining_result: MiningResult) -> Dict[str, Any]:
        """Generate recipe from mining result."""
        selectors = []
        for candidate in mining_result.candidates:
            selectors.append({
                'kind': candidate.kind,
                'css': candidate.selector,
                'description': candidate.description,
                'score': candidate.score
            })
        
        return {
            'selectors': selectors,
            'attributes_priority': ['data-src', 'data-srcset', 'srcset', 'src'],
            'extra_sources': [
                "meta[property='og:image']::attr(content)",
                "img::attr(srcset)",
                "source::attr(data-srcset)"
            ],
            'method': 'mined',
            'confidence': mining_result.candidates[0].score if mining_result.candidates else 0.0
        }
    
    async def _process_images_streaming(self, extracted_images: List[ExtractedImage], source_url: str) -> Dict[str, int]:
        """Process images in streaming batches with quota management and memory management."""
        images_processed = 0
        faces_detected = 0
        images_saved = 0
        thumbnails_saved = 0
        inflight_gets = 0
        stop_queuing_heads = False
        
        # Process in batches to control memory usage
        for i in range(0, len(extracted_images), self.config.batch_size):
            batch_images = extracted_images[i:i + self.config.batch_size]
            
            # Check quota before processing batch
            if stop_queuing_heads or (images_saved + inflight_gets >= self.config.max_images):
                stop_queuing_heads = True
                logger.info(f"Quota reached or near limit: saved={images_saved}, inflight={inflight_gets}, max={self.config.max_images}")
                break
            
            # Check memory pressure before processing batch
            if self.memory_manager.is_memory_pressured():
                await self.memory_manager.force_gc("batch_processing")
                await asyncio.sleep(0.1)  # Brief pause for GC
            
            # Process batch
            batch_urls = [img.url for img in batch_images]
            batch_context = [img.context for img in batch_images]
            
            # Process images concurrently within batch
            semaphore = asyncio.Semaphore(self.config.concurrent_downloads)
            
            async def process_with_semaphore(url: str, context: Dict[str, Any]) -> Optional[ProcessedImage]:
                nonlocal inflight_gets
                async with semaphore:
                    inflight_gets += 1
                    try:
                        return await self.processor.process_single_image(url, source_url, context)
                    finally:
                        inflight_gets -= 1
            
            tasks = [process_with_semaphore(url, context) for url, context in zip(batch_urls, batch_context)]
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store processed images
            for i, result in enumerate(batch_results):
                if isinstance(result, ProcessedImage):
                    images_processed += 1
                    faces_detected += len(result.faces)
                    
                    # Get context for this image
                    image_context = batch_context[i] if i < len(batch_context) else {}
                    
                    # Store image with context
                    storage_result = await self._store_processed_image(result, source_url, image_context)
                    if storage_result and storage_result.success:
                        images_saved += 1
                        thumbnails_saved += storage_result.thumbnail_count
                        
                        # Check if quota is hit
                        if images_saved >= self.config.max_images:
                            logger.info(f"Quota hit: {images_saved} images saved, stopping processing")
                            # Cancel any remaining pending tasks
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            break
            
            # Periodic GC
            if images_processed % self.config.gc_frequency == 0:
                await self.memory_manager.force_gc("periodic")
        
        return {
            'images_processed': images_processed,
            'faces_detected': faces_detected,
            'images_saved': images_saved,
            'thumbnails_saved': thumbnails_saved
        }
    
    async def _store_processed_image(self, processed_image: ProcessedImage, source_url: str, context: Optional[Dict[str, Any]] = None) -> Optional[StorageResult]:
        """Store processed image with metadata and all face thumbnails."""
        try:
            # Extract domain for site identification
            domain = self._extract_domain(source_url)
            
            # Prepare comprehensive metadata
            metadata = {
                'original_url': processed_image.original_url,
                'source_url': source_url,
                'dimensions': processed_image.dimensions,
                'perceptual_hash': processed_image.perceptual_hash,
                'faces_detected': len(processed_image.faces),
                'enhancement_applied': processed_image.enhancement_applied,
                'face_embeddings': [face.get('embedding') for face in processed_image.faces if face.get('embedding')],
                'face_bboxes': [face.get('bbox') for face in processed_image.faces if face.get('bbox')],
                'face_scores': [face.get('det_score') for face in processed_image.faces if face.get('det_score')],
                'face_areas': [face.get('face_area') for face in processed_image.faces if face.get('face_area')],
                'face_ratios': [face.get('face_ratio') for face in processed_image.faces if face.get('face_ratio')],
                'upscaled_faces': [face.get('upscaled', False) for face in processed_image.faces]
            }
            
            # Extract additional context information
            context = context or {}
            selector_source = context.get('selector', 'unknown')
            
            # Store image with all thumbnails and comprehensive metadata
            storage_result = await self.storage.store_image_with_multiple_thumbnails(
                processed_image.image_data,
                processed_image.thumbnail_data,  # List of thumbnails
                metadata,
                site=domain,
                page_url=source_url,
                source_video_url=source_url,  # For now, same as page_url (could be enhanced for video/album detection)
                source_image_url=processed_image.original_url,
                selector_source=selector_source
            )
            
            return storage_result
            
        except Exception as e:
            logger.error(f"Error storing processed image: {e}")
            return None
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower()
        except:
            return url.lower()
    
    def _update_stats(self, processing_result: Dict[str, int], mining_attempted: bool, mining_success: bool):
        """Update crawling statistics."""
        self.stats.total_sites += 1
        self.stats.successful_sites += 1
        self.stats.total_images_processed += processing_result['images_processed']
        self.stats.total_faces_detected += processing_result['faces_detected']
        self.stats.total_images_saved += processing_result['images_saved']
        self.stats.total_thumbnails_saved += processing_result['thumbnails_saved']
        
        if mining_attempted:
            self.stats.mining_attempts += 1
        if mining_success:
            self.stats.mining_successes += 1
    
    def get_statistics(self) -> CrawlStats:
        """Get crawling statistics."""
        return self.stats
    
    def reset_statistics(self) -> None:
        """Reset crawling statistics."""
        self.stats = CrawlStats(
            total_sites=0,
            successful_sites=0,
            failed_sites=0,
            total_images_found=0,
            total_images_processed=0,
            total_faces_detected=0,
            total_images_saved=0,
            total_thumbnails_saved=0,
            total_duration=0.0,
            mining_attempts=0,
            mining_successes=0
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health_status = {
            'overall_healthy': True,
            'components': {},
            'memory_stats': self.memory_manager.get_memory_stats().__dict__,
            'crawl_stats': self.stats.__dict__
        }
        
        try:
            # Check storage health
            if self.storage:
                storage_health = await self.storage.health_check()
                health_status['components']['storage'] = storage_health
                if not storage_health.get('healthy', False):
                    health_status['overall_healthy'] = False
            
            # Check memory health
            if self.memory_manager.is_memory_critical():
                health_status['components']['memory'] = {'healthy': False, 'status': 'critical'}
                health_status['overall_healthy'] = False
            elif self.memory_manager.is_memory_pressured():
                health_status['components']['memory'] = {'healthy': True, 'status': 'pressured'}
            else:
                health_status['components']['memory'] = {'healthy': True, 'status': 'normal'}
            
            # Check active crawls
            health_status['components']['crawler'] = {
                'healthy': True,
                'active_crawls': len(self._active_crawls),
                'max_concurrent': self.config.concurrent_downloads
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            health_status['overall_healthy'] = False
            health_status['error'] = str(e)
        
        return health_status
