"""
Extractor Worker for New Crawler System

Image download and batch preparation worker process.
Downloads images, performs HEAD/GET validation, computes phash, and creates batches.
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Tuple, Union

from .config import get_config
from .redis_manager import get_redis_manager
from .cache_manager import get_cache_manager
from .http_utils import get_http_utils
from .data_structures import CandidateImage, CandidatePost, ImageTask, PostTask, BatchRequest, TaskStatus
from .timing_logger import get_timing_logger

logger = logging.getLogger(__name__)


class ExtractorWorker:
    """Extractor worker for image download and batch preparation."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.config = get_config()
        self.redis = get_redis_manager()
        self.cache = get_cache_manager()
        self.http_utils = get_http_utils()
        self.timing_logger = get_timing_logger()
        
        # Worker state
        self.running = False
        self.processed_candidates = 0
        self.downloaded_images = 0
        self.cached_images = 0
        
        # Concurrency control - divide total concurrency among workers
        concurrency_per_worker = self.config.nc_extractor_concurrency // self.config.num_extractors
        self._semaphore = asyncio.Semaphore(max(1, concurrency_per_worker))
        logger.info(f"[Extractor {self.worker_id}] Concurrency: {concurrency_per_worker} per worker "
                   f"(total: {self.config.nc_extractor_concurrency} across {self.config.num_extractors} workers)")
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"extractor_{worker_id}_")
        # Track temp files for cleanup (file_path -> timestamp when created)
        self._temp_files: dict = {}
        logger.info(f"[Extractor {self.worker_id}] Using temp directory: {self.temp_dir}")
    
    async def process_candidate(self, candidate) -> Optional:
        """Process a single candidate (either image or post)."""
        extraction_start_time = time.time()

        # Handle different candidate types
        if isinstance(candidate, CandidatePost):
            return await self._process_post_candidate(candidate, extraction_start_time)
        elif isinstance(candidate, CandidateImage):
            # Skip image processing if image extraction is disabled
            if not self.config.nc_enable_image_extraction:
                logger.debug(f"[Extractor {self.worker_id}] Image extraction disabled, skipping: {candidate.img_url}")
                return None
            return await self._process_image_candidate(candidate, extraction_start_time)
        else:
            logger.warning(f"[Extractor {self.worker_id}] Unknown candidate type: {type(candidate)}")
            return None

    async def _process_image_candidate(self, candidate: CandidateImage, extraction_start_time: float) -> Optional[ImageTask]:
        """Process a single candidate image."""
        async with self._semaphore:  # Limit concurrent downloads
            try:
                logger.debug(f"[Extractor {self.worker_id}] Processing candidate: {candidate.img_url}")
                
                # Check URL deduplication before download
                if await self.redis.url_seen_async(candidate.img_url):
                    logger.debug(f"[Extractor {self.worker_id}] URL already seen, skipping: {candidate.img_url}")
                    # Log extraction end
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                    return None
                
                # Check strict_limits BEFORE downloading image (quick cutoff to avoid wasting resources)
                if self.config.nc_strict_limits:
                    # Check if site has reached image limit (check directly, not just flag)
                    stats = await asyncio.to_thread(self.redis.get_site_stats, candidate.site_id)
                    if stats:
                        thumbs = stats.get('images_saved_thumbs', 0)
                        if self.config.nc_max_images_per_site > 0 and thumbs >= self.config.nc_max_images_per_site:
                            logger.debug(f"[EXTRACTOR-{self.worker_id}] Not processing candidate (image limit reached: {thumbs} >= {self.config.nc_max_images_per_site}): {candidate.img_url}")
                            # Log extraction end
                            extraction_duration = (time.time() - extraction_start_time) * 1000
                            self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                            return None
                    # Also check site limit flag (set when pages or images limit reached)
                    if await self.redis.is_site_limit_reached_async(candidate.site_id):
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] Not processing candidate (site limit flag set): {candidate.img_url}")
                        # Log extraction end
                        extraction_duration = (time.time() - extraction_start_time) * 1000
                        self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                        return None
                
                # Log extraction start
                self.timing_logger.log_extraction_start(candidate.site_id, candidate.img_url)
                
                # If we have metadata from HTML, skip HEAD check entirely!
                if candidate.content_type and candidate.width and candidate.height:
                    logger.debug(f"[EXTRACTOR-{self.worker_id}] Using HTML metadata, skipping HEAD check")
                    head_info = {
                        'content_type': candidate.content_type,
                        'content_length': candidate.estimated_size or 0
                    }
                elif not self.config.nc_skip_head_check:
                    # Fallback: HEAD check if metadata not available
                    is_valid, head_info = await self.http_utils.head_check(candidate.img_url)
                    if not is_valid:
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] HEAD check: {candidate.img_url} - FAILED")
                        # Log extraction end
                        extraction_duration = (time.time() - extraction_start_time) * 1000
                        self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                        return None
                    logger.debug(f"[EXTRACTOR-{self.worker_id}] HEAD check: {candidate.img_url} - OK")
                else:
                    # No metadata, no HEAD check - just proceed
                    logger.debug(f"[EXTRACTOR-{self.worker_id}] Skipping HEAD check (config enabled)")
                    head_info = {}
                
                # Download image to temp file (only necessary HTTP request!)
                temp_path, download_info = await self.http_utils.download_to_temp(
                    candidate.img_url, self.temp_dir
                )
                
                # Track temp file for cleanup
                if temp_path:
                    self._temp_files[temp_path] = time.time()
                
                if not temp_path:
                    logger.debug(f"[EXTRACTOR-{self.worker_id}] Download: {candidate.img_url} - FAILED")
                    # Log extraction end
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                    return None
                
                file_size = download_info.get('content_length', 0)
                logger.debug(f"[EXTRACTOR-{self.worker_id}] Download: {candidate.img_url} - {file_size}bytes")
                
                # Log temp file creation
                if os.path.exists(temp_path):
                    actual_size = os.path.getsize(temp_path)
                    file_mtime = os.path.getmtime(temp_path)
                    file_age = time.time() - file_mtime
                    logger.info(f"[EXTRACTOR-{self.worker_id}] [TEMP-FILE] Created temp file: {temp_path}, "
                              f"size={actual_size}bytes, expected={file_size}bytes, "
                              f"age={file_age:.1f}s, mtime={file_mtime:.1f}")
                
                # Compute phash in thread pool (CPU-bound)
                phash = await asyncio.to_thread(
                    self.cache.compute_phash, temp_path
                )
                if not phash:
                    logger.debug(f"[EXTRACTOR-{self.worker_id}] Phash: {temp_path} - FAILED")
                    # Log intentional deletion before removing
                    if os.path.exists(temp_path):
                        file_size = os.path.getsize(temp_path)
                        file_mtime = os.path.getmtime(temp_path)
                        file_age = time.time() - file_mtime
                        logger.info(f"[EXTRACTOR-{self.worker_id}] [TEMP-FILE] Deleting temp file (phash failed): "
                                  f"{temp_path}, size={file_size}bytes, age={file_age:.1f}s")
                    os.remove(temp_path)
                    # Log extraction end
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                    return None
                
                # Check cache
                is_cached = await asyncio.to_thread(
                    self.cache.is_image_cached, phash
                )
                logger.debug(f"[EXTRACTOR-{self.worker_id}] Phash: {phash[:8]}... - cached={is_cached}")
                
                if is_cached:
                    # Log intentional deletion before removing
                    if os.path.exists(temp_path):
                        file_size = os.path.getsize(temp_path)
                        file_mtime = os.path.getmtime(temp_path)
                        file_age = time.time() - file_mtime
                        logger.info(f"[EXTRACTOR-{self.worker_id}] [TEMP-FILE] Deleting temp file (cached): "
                                  f"{temp_path}, size={file_size}bytes, age={file_age:.1f}s")
                    os.remove(temp_path)
                    self.cached_images += 1
                    # Log extraction end
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                    return None
                
                # Mark URL as seen (after successful download)
                ttl_hours = self.config.nc_url_dedup_ttl_hours
                ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else None
                await self.redis.mark_url_seen_async(candidate.img_url, ttl_seconds)

                # Get actual file size for filtering
                actual_file_size = os.path.getsize(temp_path) if os.path.exists(temp_path) else 0

                # Check minimum file size for GPU processing (filter out tiny avatars/UI elements)
                if actual_file_size < self.config.min_image_file_size_bytes:
                    logger.debug(f"[EXTRACTOR-{self.worker_id}] Image too small for GPU processing "
                                f"({actual_file_size} < {self.config.min_image_file_size_bytes} bytes): {candidate.img_url}")
                    # Clean up temp file - don't waste GPU time on tiny images
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    # Log extraction end
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                    return None

                # Check strict_limits before creating image task (quick check in case limit reached during processing)
                # Note: Comprehensive check already done before download, this is just a safety check
                if self.config.nc_strict_limits:
                    if await self.redis.is_site_limit_reached_async(candidate.site_id):
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] Not creating image task (site limit flag set during processing): {candidate.img_url}")
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        # Log extraction end
                        extraction_duration = (time.time() - extraction_start_time) * 1000
                        self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                        return None

                # Final validation: ensure temp file still exists before creating image task
                if not os.path.exists(temp_path):
                    logger.warning(f"[EXTRACTOR-{self.worker_id}] Temp file disappeared before image task creation: {temp_path}")
                    # Log extraction end
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                    return None

                # All checks passed - create image task
                image_task = ImageTask(
                    temp_path=temp_path,
                    phash=phash,
                    candidate=candidate,
                    file_size=download_info.get('content_length', 0),
                    mime_type=download_info.get('content_type', 'image/jpeg')
                )

                # Push immediately to GPU inbox (no batching - scheduler handles it)
                payload = self.redis.serialize_image_task(image_task)
                inbox_key = getattr(self.config, 'gpu_inbox_key', 'gpu:inbox')
                pushed = await asyncio.to_thread(self.redis.push_many, inbox_key, [payload])
                if pushed != 1:
                    logger.warning(f"[EXTRACTOR-{self.worker_id}] Failed to enqueue task: {image_task.candidate.img_url}")

                logger.debug(f"[Extractor {self.worker_id}] Created image task: {phash[:8]}...")
                # Log extraction end
                extraction_duration = (time.time() - extraction_start_time) * 1000
                self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                return image_task
                
            except Exception as e:
                logger.error(f"[Extractor {self.worker_id}] Error processing candidate {candidate.img_url}: {e}")
                # Log extraction end
                extraction_duration = (time.time() - extraction_start_time) * 1000
                self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                return None

    async def _process_post_candidate(self, candidate: CandidatePost, extraction_start_time: float) -> Optional[PostTask]:
        """Process a diabetes-related post candidate."""
        async with self._semaphore:  # Limit concurrent processing
            try:
                logger.debug(f"[Extractor {self.worker_id}] Processing diabetes post: {candidate.post_url}")

                # Check URL deduplication
                if await self.redis.url_seen_async(candidate.post_url):
                    logger.debug(f"[Extractor {self.worker_id}] Post URL already seen, skipping: {candidate.post_url}")
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.post_url, extraction_duration)
                    return None

                # Check strict_limits
                if self.config.nc_strict_limits:
                    stats = await asyncio.to_thread(self.redis.get_site_stats, candidate.site_id)
                    if stats:
                        posts_saved = stats.get('posts_saved', 0)
                        if self.config.nc_max_posts_per_site > 0 and posts_saved >= self.config.nc_max_posts_per_site:
                            logger.debug(f"[EXTRACTOR-{self.worker_id}] Not processing post (limit reached: {posts_saved} >= {self.config.nc_max_posts_per_site}): {candidate.post_url}")
                            extraction_duration = (time.time() - extraction_start_time) * 1000
                            self.timing_logger.log_extraction_end(candidate.site_id, candidate.post_url, extraction_duration)
                            return None
                    if await self.redis.is_site_limit_reached_async(candidate.site_id):
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] Not processing post (site limit flag set): {candidate.post_url}")
                        extraction_duration = (time.time() - extraction_start_time) * 1000
                        self.timing_logger.log_extraction_end(candidate.site_id, candidate.post_url, extraction_duration)
                        return None

                # Log extraction start
                self.timing_logger.log_extraction_start(candidate.site_id, candidate.post_url)

                # Create content hash
                import hashlib
                content_hash = hashlib.sha256((candidate.content or "").encode('utf-8')).hexdigest()

                # Check if post contains diabetes keywords
                diabetes_keywords = [
                    'diabetes', 'diabetic', 'insulin', 'blood sugar', 'glucose',
                    'type 1 diabetes', 'type 2 diabetes', 'gestational diabetes',
                    'diabetes mellitus', 'hyperglycemia', 'hypoglycemia',
                    'a1c', 'hba1c', 'blood glucose', 'sugar levels'
                ]
                has_keywords = False
                # Check title
                if candidate.title:
                    title_lower = candidate.title.lower()
                    if any(kw in title_lower for kw in diabetes_keywords):
                        has_keywords = True
                # Check content
                if not has_keywords and candidate.content:
                    content_lower = candidate.content.lower()
                    if any(kw in content_lower for kw in diabetes_keywords):
                        has_keywords = True

                # Mark URL as seen
                ttl_hours = self.config.nc_url_dedup_ttl_hours
                ttl_seconds = ttl_hours * 3600 if ttl_hours > 0 else None
                await self.redis.mark_url_seen_async(candidate.post_url, ttl_seconds)

                # Create post task with keyword flag
                post_task = PostTask(
                    candidate=candidate,
                    content_hash=content_hash,
                    has_keywords=has_keywords
                )

                # Push directly to storage queue (bypass GPU worker)
                await self.redis.push_post_task_async(post_task)

                if has_keywords:
                    logger.info(f"[Extractor {self.worker_id}] Pushed post with keywords to storage: {content_hash[:8]}...")
                else:
                    logger.info(f"[Extractor {self.worker_id}] Pushed post without keywords to storage (will save to raw-images only): {content_hash[:8]}...")
                extraction_duration = (time.time() - extraction_start_time) * 1000
                self.timing_logger.log_extraction_end(candidate.site_id, candidate.post_url, extraction_duration)

                self.downloaded_images += 1  # Reuse counter for posts
                return post_task

            except Exception as e:
                logger.error(f"[Extractor {self.worker_id}] Error processing post {candidate.post_url}: {e}")
                extraction_duration = (time.time() - extraction_start_time) * 1000
                self.timing_logger.log_extraction_end(candidate.site_id, candidate.post_url, extraction_duration)
                return None

    
    
    
    async def _process_candidate_with_stats(self, candidate: Union[CandidateImage, CandidatePost], site_extraction_times: dict) -> None:
        """Process candidate and update statistics."""
        site_id = candidate.site_id
        
        # Process candidate (returns ImageTask or PostTask)
        task = await self.process_candidate(candidate)
        
        if task:
            # Handle different task types
            if isinstance(task, ImageTask):
                self.downloaded_images += 1
                
                # Update extraction end time
                site_extraction_times[site_id]['end'] = datetime.now()
                
                # Update statistics in Redis
                await self.redis.update_site_stats_async(
                    site_id,
                    {
                        'images_processed': 1,
                        'extraction_end_time': site_extraction_times[site_id]['end']
                    }
                )
                
                # Check and set site limit flag if reached
                stats = await asyncio.to_thread(self.redis.get_site_stats, site_id)
                thumbs = stats.get('images_saved_thumbs', 0) if stats else 0
                if thumbs >= self.config.nc_max_images_per_site:
                    await self.redis.set_site_limit_reached_async(site_id)
                    # If strict_limits enabled, cleanup gpu:inbox queue for this site
                    if self.config.nc_strict_limits:
                        removed = await self.redis.remove_site_items_from_queue_async('gpu:inbox', site_id)
                        if removed > 0:
                            logger.info(f"[Extractor {self.worker_id}] Removed {removed} items from gpu:inbox for site {site_id} (image limit reached)")
            elif isinstance(task, PostTask):
                # PostTask is already pushed to storage queue, just update stats
                self.downloaded_images += 1  # Reuse counter for posts
                
                # Update extraction end time
                site_extraction_times[site_id]['end'] = datetime.now()
                
                # Update statistics in Redis
                await self.redis.update_site_stats_async(
                    site_id,
                    {
                        'posts_processed': 1,
                        'extraction_end_time': site_extraction_times[site_id]['end']
                    }
                )
        
        self.processed_candidates += 1
        
        # Log progress periodically with throughput
        if self.processed_candidates % 100 == 0 and self.processed_candidates > 0:
            logger.info(f"[Extractor {self.worker_id}] Processed {self.processed_candidates} candidates, "
                       f"{self.downloaded_images} downloaded, {self.cached_images} cached")
        
        # Diagnostic throughput logging
        if self.config.nc_diagnostic_logging and self.processed_candidates % self.config.nc_diagnostic_log_interval == 0:
            # Track throughput
            now = time.time()
            if not hasattr(self, '_throughput_start_time'):
                self._throughput_start_time = now
                self._throughput_image_count = 0
            
            elapsed = now - self._throughput_start_time
            if elapsed > 0:
                images_per_sec = self.downloaded_images / elapsed if hasattr(self, '_throughput_start_time') else 0
                logger.info(f"[EXTRACTOR-DIAG-{self.worker_id}] Throughput: {images_per_sec:.1f} images/sec, "
                          f"processed={self.processed_candidates}, downloaded={self.downloaded_images}")
                
                # Reset for next window
                self._throughput_start_time = now
    
    async def run(self):
        """Main worker loop."""
        logger.info(f"[Extractor {self.worker_id}] Starting extractor worker")
        self.running = True

        # Track first and last extraction times per site
        site_extraction_times = {}  # {site_id: {'start': datetime, 'end': datetime}}
        
        # Batch size for pop operations
        batch_pop_size = self.config.nc_extractor_batch_pop_size
        
        # Diagnostic logging state
        if self.config.nc_diagnostic_logging:
            self._last_empty_pop_time = None
            self._empty_pop_wait_time = 0.0
        
        try:
            while self.running:
                # Batch pop candidates (much faster than individual pops)
                queue_name = self.config.get_queue_name('candidates')
                
                # Diagnostic: Get queue depth before pop
                queue_depth_before = None
                if self.config.nc_diagnostic_logging:
                    queue_depth_before = await self.redis.get_queue_length_by_key_async(queue_name)
                
                pop_start = time.time()
                raw_candidates = await asyncio.to_thread(
                    self.redis.blpop_many, queue_name, max_n=batch_pop_size, timeout=2.0
                )
                pop_duration = time.time() - pop_start
                
                # Diagnostic logging for batch pop
                if self.config.nc_diagnostic_logging:
                    if not raw_candidates:
                        # Queue was empty - track wait time
                        if self._last_empty_pop_time is not None:
                            self._empty_pop_wait_time += pop_duration
                        else:
                            self._last_empty_pop_time = time.time()
                        logger.debug(f"[EXTRACTOR-DIAG-{self.worker_id}] Popped 0 candidates, "
                                   f"queue_depth={queue_depth_before}, wait_time={pop_duration:.3f}s")
                    else:
                        # Got candidates - log and reset empty wait tracking
                        queue_depth_after = await self.redis.get_queue_length_by_key_async(queue_name)
                        if self._last_empty_pop_time is not None:
                            total_empty_wait = self._empty_pop_wait_time + pop_duration
                            logger.info(f"[EXTRACTOR-DIAG-{self.worker_id}] Queue was empty for {total_empty_wait:.2f}s, "
                                     f"now popped {len(raw_candidates)} candidates, "
                                     f"queue_depth_before={queue_depth_before}, queue_depth_after={queue_depth_after}")
                            self._last_empty_pop_time = None
                            self._empty_pop_wait_time = 0.0
                        else:
                            logger.debug(f"[EXTRACTOR-DIAG-{self.worker_id}] Popped {len(raw_candidates)} candidates, "
                                       f"queue_depth_before={queue_depth_before}, queue_depth_after={queue_depth_after}")
                
                if not raw_candidates:
                    await asyncio.sleep(0.1)
                    continue
                
                # Deserialize all candidates (can be CandidateImage or CandidatePost)
                candidates = []
                for raw in raw_candidates:
                    try:
                        # Try to determine type by checking JSON structure
                        data_str = raw.decode('utf-8') if isinstance(raw, bytes) else raw
                        import json
                        data_dict = json.loads(data_str)
                        # Check if it's a CandidatePost by looking for post_url field
                        if 'post_url' in data_dict and 'img_url' not in data_dict:
                            candidate = self.redis._deserialize(raw, CandidatePost)
                        else:
                            candidate = self.redis._deserialize(raw, CandidateImage)
                        if candidate:
                            candidates.append(candidate)
                    except Exception as e:
                        logger.warning(f"[Extractor {self.worker_id}] Failed to deserialize candidate: {e}")
                        continue
                
                if not candidates:
                    continue
                
                # Process all candidates concurrently (up to semaphore limit)
                tasks = []
                for candidate in candidates:
                    site_id = candidate.site_id
                    
                    # Check site limit
                    if await self.redis.is_site_limit_reached_async(site_id):
                        if site_id not in site_extraction_times:
                            logger.info(f"[EXTRACTOR-{self.worker_id}] Dropping candidate for {site_id} (limit reached)")
                            site_extraction_times[site_id] = {'start': None, 'end': None}
                        continue
                    
                    # Track extraction start time
                    if site_id not in site_extraction_times:
                        site_extraction_times[site_id] = {
                            'start': datetime.now(),
                            'end': None
                        }
                        await self.redis.update_site_stats_async(
                            site_id,
                            {'extraction_start_time': site_extraction_times[site_id]['start']}
                        )
                    
                    # Create async task for each candidate (semaphore inside process_candidate limits concurrency)
                    tasks.append(self._process_candidate_with_stats(candidate, site_extraction_times))
                
                # Process all candidates concurrently
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Periodic cleanup of old temp files (every batch)
                # More aggressive cleanup: reduce threshold and delay
                max_tracked_files = 500
                if len(self._temp_files) > max_tracked_files:
                    logger.info(f"[Extractor {self.worker_id}] Temp files exceed limit ({len(self._temp_files)} > {max_tracked_files}), forcing cleanup")
                    self._cleanup_temp_dir()
                elif len(self._temp_files) > 100:  # Cleanup when we have many tracked files
                    self._cleanup_temp_dir()
                    
        finally:
            # Final cleanup on exit
            self._cleanup_temp_dir()
    
    def _cleanup_temp_dir(self):
        """Cleanup temporary files with proper tracking and delayed deletion."""
        try:
            import time
            import os
            
            # Clean up files older than 2 minutes (reduced from 5 minutes)
            current_time = time.time()
            cleanup_delay = 120  # 2 minutes
            files_cleaned = 0
            files_failed = 0
            
            # Clean tracked files
            for file_path, created_time in list(self._temp_files.items()):
                if current_time - created_time > cleanup_delay:
                    try:
                        if os.path.exists(file_path):
                            os.unlink(file_path)
                            files_cleaned += 1
                        del self._temp_files[file_path]
                    except Exception as e:
                        logger.warning(f"[Extractor {self.worker_id}] Failed to delete temp file {file_path}: {e}")
                        files_failed += 1
                        # Keep in dict to retry later
            
            # Also try to clean entire temp directory if empty or old
            try:
                if os.path.exists(self.temp_dir):
                    remaining_files = [f for f in os.listdir(self.temp_dir) if os.path.isfile(os.path.join(self.temp_dir, f))]
                    if not remaining_files:
                        os.rmdir(self.temp_dir)
                        logger.info(f"[Extractor {self.worker_id}] Cleaned up empty temp directory: {self.temp_dir}")
                    else:
                        logger.debug(f"[Extractor {self.worker_id}] Temp directory still has {len(remaining_files)} files")
            except Exception as e:
                logger.debug(f"[Extractor {self.worker_id}] Could not remove temp directory: {e}")
            
            if files_cleaned > 0:
                logger.info(f"[Extractor {self.worker_id}] Cleaned up {files_cleaned} temp files (failed: {files_failed}, remaining: {len(self._temp_files)})")
        except Exception as e:
            logger.error(f"[Extractor {self.worker_id}] Error during cleanup: {e}")
    
    def cleanup(self):
        """Cleanup temporary files."""
        self._cleanup_temp_dir()
    
    def stop(self):
        """Stop the worker."""
        self.running = False
    
    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'processed_candidates': self.processed_candidates,
            'downloaded_images': self.downloaded_images,
            'cached_images': self.cached_images,
            'running': self.running
        }


def extractor_worker_process(worker_id: int):
    """Extractor worker process entry point."""
    # Reset signal handlers to default (let parent handle)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    
    # Configure logging for multiprocessing
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - Extractor-{worker_id} - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"[Extractor {worker_id}] Starting extractor worker process")
    
    try:
        # Create worker
        worker = ExtractorWorker(worker_id)
        
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run worker
            loop.run_until_complete(worker.run())
        finally:
            # Cleanup
            worker.cleanup()
            loop.close()
            
    except Exception as e:
        logger.error(f"[Extractor {worker_id}] Fatal error: {e}", exc_info=True)
    finally:
        logger.info(f"[Extractor {worker_id}] Extractor worker process ended")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True)
    
    args = parser.parse_args()
    extractor_worker_process(args.worker_id)
