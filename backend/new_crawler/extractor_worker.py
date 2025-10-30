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
import tempfile
import time
from datetime import datetime
from typing import List, Optional, Tuple

from .config import get_config
from .redis_manager import get_redis_manager
from .cache_manager import get_cache_manager
from .http_utils import get_http_utils
from .data_structures import CandidateImage, ImageTask, BatchRequest, TaskStatus
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
        self.batches_created = 0
        
        # Shared batch configuration (no local batch)
        self.batch_size_threshold = self.config.nc_batch_size  # 64 images
        logger.info(f"[Extractor {self.worker_id}] Using shared batch with threshold: {self.batch_size_threshold}")
        # Remove these:
        # self.current_batch: List[ImageTask] = []
        # self.per_worker_batch_size = ...
        # self._batch_lock = asyncio.Lock()
        # self._flush_task = ...
        
        # Concurrency control - per-worker semaphore for parallel downloads
        # Each worker handles nc_extractor_concurrency // num_extractors concurrent downloads
        per_worker_concurrency = max(1, self.config.nc_extractor_concurrency // self.config.num_extractors)
        self._semaphore = asyncio.Semaphore(per_worker_concurrency)
        logger.info(f"[Extractor {self.worker_id}] Per-worker concurrency: {per_worker_concurrency} (total: {self.config.nc_extractor_concurrency})")
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"extractor_{worker_id}_")
        logger.info(f"[Extractor {self.worker_id}] Using temp directory: {self.temp_dir}")
        
        # Track last flush time for time-based flushing
        self._last_flush_time = time.time()

        # Sensible defaults with config overrides for metadata/HEAD gating
        self.min_image_bytes = getattr(self.config, "nc_min_image_bytes", 4_096)
        self.max_image_bytes = getattr(self.config, "nc_max_image_bytes", 25_000_000)
        self.min_side_px = getattr(self.config, "nc_min_side_px", 64)
        self.max_aspect_ratio = getattr(self.config, "nc_max_aspect", 4.0)
        self.disallow_svg = getattr(self.config, "nc_disallow_svg", True)
    
    def _metadata_gate(self, candidate) -> tuple[bool, dict, str]:
        """
        Use HTML-derived hints to decide if we can skip HEAD.
        Returns (ok, head_info_like_dict, reason).
        """
        ct = (candidate.content_type or "").lower()
        w = candidate.width
        h = candidate.height
        est_size = getattr(candidate, 'estimated_size', None)

        if not (ct or w or h or est_size):
            return False, {}, "no_metadata"

        # Content type must look like a real raster image, if present.
        if ct:
            if "image" not in ct:
                return False, {}, f"ctype_not_image:{ct}"
            if self.disallow_svg and "svg" in ct:
                return False, {}, "svg_disallowed"
        else:
            # No content-type from HTML: rely on URL extension + keep SVG ban
            url_lower = (getattr(candidate, "img_url", "") or "").lower()
            if self.disallow_svg and url_lower.endswith(".svg"):
                return False, {}, "svg_disallowed"
            if not url_lower.endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                return False, {}, "ext_not_raster"

        # Dimensions, if present, must be sane
        if w and h:
            if min(w, h) < self.min_side_px:
                return False, {}, f"too_small:{w}x{h}"
            ar = (w / h) if h else 0
            if ar and (ar > self.max_aspect_ratio or ar < 1 / self.max_aspect_ratio):
                return False, {}, f"extreme_aspect:{ar:.2f}"

        # Estimated size, if present, must be within range
        if est_size is not None:
            try:
                # allow string/num
                size_val = int(est_size)
            except Exception:
                size_val = None
            if size_val is not None:
                if size_val < self.min_image_bytes:
                    return False, {}, f"too_tiny:{size_val}"
                if size_val > self.max_image_bytes:
                    return False, {}, f"too_large:{size_val}"

        # Looks good enough to skip HEAD; fabricate a minimal head_info
        head_info = {
            "content_type": ct,
            "content_length": str(est_size) if est_size is not None else None,
            "last_modified": None,
            "etag": None,
        }
        return True, head_info, "html_metadata_ok"

    async def process_candidate(self, candidate: CandidateImage) -> Optional[ImageTask]:
        """Process a single candidate image."""
        extraction_start_time = time.time()
        async with self._semaphore:  # Limit concurrent downloads
            try:
                logger.debug(f"[Extractor {self.worker_id}] Processing candidate: {candidate.img_url}")
                
                # Log extraction start
                self.timing_logger.log_extraction_start(candidate.site_id, candidate.img_url)
                
                # --- BEGIN: reconciled metadata/HEAD gating ---
                use_meta = False
                head_info = {}

                # 1) Try to use HTML metadata if strong enough
                ok_meta = False
                if candidate.content_type or candidate.width or candidate.height or getattr(candidate, 'estimated_size', None) is not None:
                    ok_meta, head_info, reason = self._metadata_gate(candidate)
                    if ok_meta:
                        use_meta = True
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] Using HTML metadata (skip HEAD): {reason}")
                    else:
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] HTML metadata not sufficient: {reason}. Will consider HEAD.")

                # 2) If HTML metadata wasn’t strong, try HEAD (unless globally skipped)
                if not use_meta:
                    if not self.config.nc_skip_head_check:
                        is_valid, hi = await self.http_utils.head_check(candidate.img_url)
                        head_info = hi or {}
                        if not is_valid:
                            logger.debug(f"[EXTRACTOR-{self.worker_id}] HEAD check: {candidate.img_url} - FAILED")
                            extraction_duration = (time.time() - extraction_start_time) * 1000
                            self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                            return None
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] HEAD check: {candidate.img_url} - OK")
                    else:
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] Skipping HEAD check (config nc_skip_head_check=true)")
                        head_info = {}

                # 3) Proceed to download — we have either trusted HTML or passed HEAD
                # --- END: reconciled metadata/HEAD gating ---
                
                # Download image to temp file (only necessary HTTP request!)
                temp_path, download_info = await self.http_utils.download_to_temp(
                    candidate.img_url, self.temp_dir
                )
                
                if not temp_path:
                    logger.debug(f"[EXTRACTOR-{self.worker_id}] Download: {candidate.img_url} - FAILED")
                    # Log extraction end
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                    return None
                
                file_size = download_info.get('content_length', 0)
                logger.debug(f"[EXTRACTOR-{self.worker_id}] Download: {candidate.img_url} - {file_size}bytes")
                
                # Compute phash in thread pool (CPU-bound)
                phash = await asyncio.to_thread(
                    self.cache.compute_phash, temp_path
                )
                if not phash:
                    logger.debug(f"[EXTRACTOR-{self.worker_id}] Phash: {temp_path} - FAILED")
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
                    os.remove(temp_path)
                    self.cached_images += 1
                    # Log extraction end
                    extraction_duration = (time.time() - extraction_start_time) * 1000
                    self.timing_logger.log_extraction_end(candidate.site_id, candidate.img_url, extraction_duration)
                    return None
                
                # Create image task
                image_task = ImageTask(
                    temp_path=temp_path,
                    phash=phash,
                    candidate=candidate,
                    file_size=download_info.get('content_length', 0),
                    mime_type=download_info.get('content_type', 'image/jpeg')
                )
                
                # Push to shared batch and check if flush needed
                new_batch_size = await self.redis.push_to_shared_batch_async(image_task)
                logger.debug(f"[Extractor {self.worker_id}] Pushed to shared batch, new size: {new_batch_size}")
                
                # Try to flush if batch is ready (atomic operation)
                if new_batch_size >= self.batch_size_threshold:
                    # Flush batch to GPU (site limit already checked at candidate level)
                    site_id = candidate.site_id
                    flushed = await self.redis.flush_shared_batch_if_ready_async(self.batch_size_threshold)
                    if flushed:
                        logger.info(f"[EXTRACTOR-{self.worker_id}] Flushed shared batch: {self.batch_size_threshold} images (site {site_id})")
                        self.batches_created += 1
                
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
    
    async def _download_worker_task(self, task_id: int):
        """Individual download task that continuously pops and processes candidates."""
        logger.debug(f"[Extractor {self.worker_id}] Download task {task_id} started")
        
        # Track first and last extraction times per site for this task
        site_extraction_times = {}  # {site_id: {'start': datetime, 'end': datetime}}
        
        while self.running:
            try:
                # Pop candidate with timeout
                candidate = await asyncio.to_thread(
                    self.redis.pop_candidate, timeout=1.0
                )
                
                if candidate is None:
                    # No candidates available, brief pause before retry
                    await asyncio.sleep(0.1)
                    continue
                
                site_id = candidate.site_id
                
                # Check if site already reached limit
                if await self.redis.is_site_limit_reached_async(site_id):
                    if site_id not in site_extraction_times:
                        logger.debug(f"[EXTRACTOR-{self.worker_id}-T{task_id}] Dropping candidate for {site_id} (limit reached)")
                        site_extraction_times[site_id] = {'start': None, 'end': None}
                    continue
                
                # Track extraction start time for this site
                if site_id not in site_extraction_times:
                    site_extraction_times[site_id] = {
                        'start': datetime.now(),
                        'end': None
                    }
                    await self.redis.update_site_stats_async(
                        site_id,
                        {'extraction_start_time': site_extraction_times[site_id]['start']}
                    )
                
                # Process candidate (this will use the semaphore internally)
                image_task = await self.process_candidate(candidate)
                
                if image_task:
                    self.downloaded_images += 1
                    # Update extraction end time
                    site_extraction_times[site_id]['end'] = datetime.now()
                    
                    # Update statistics in Redis
                    await self.redis.update_site_stats_async(
                        candidate.site_id,
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
                
                self.processed_candidates += 1
                
            except Exception as e:
                logger.error(f"[Extractor {self.worker_id}] Download task {task_id} error: {e}")
                # Brief pause on error to prevent tight error loops
                await asyncio.sleep(0.1)
        
        logger.debug(f"[Extractor {self.worker_id}] Download task {task_id} ended")
    
    async def run(self):
        """Main worker loop with concurrent download tasks."""
        logger.info(f"[Extractor {self.worker_id}] Starting extractor worker with parallel downloads")
        self.running = True
        
        # Background task: time-based aggressive batch flusher
        async def _stale_flusher():
            try:
                while self.running:
                    try:
                        gpu_idle = await self.redis.is_gpu_idle_async()
                        batch_size = await self.redis.get_shared_batch_size_async()
                        
                        # Multi-strategy flushing with robust GPU idle checking
                        if batch_size >= self.config.gpu_min_batch_size:
                            # Check time since last flush
                            time_since_last_flush = time.time() - self._last_flush_time
                            
                            # Three flushing strategies:
                            # 1. Full batch ready - flush immediately regardless of GPU state
                            # 2. GPU idle + minimum batch - flush to keep GPU fed
                            # 3. Time-based timeout - force flush after 2 seconds
                            should_flush = (
                                batch_size >= self.batch_size_threshold or  # Full batch (64 images)
                                (gpu_idle and batch_size >= self.config.gpu_min_batch_size) or  # GPU idle with 8+ images
                                time_since_last_flush >= self.config.nc_batch_flush_timeout  # Timeout (2 seconds)
                            )
                            
                            if should_flush:
                                # Flush up to batch_size_threshold (64 images)
                                flush_size = min(batch_size, self.batch_size_threshold)
                                flushed = await self.redis.flush_shared_batch_if_ready_async(flush_size)
                                if flushed:
                                    self._last_flush_time = time.time()
                                    reason = "full_batch" if batch_size >= self.batch_size_threshold else \
                                            "gpu_idle" if gpu_idle else "timeout"
                                    logger.info(f"[EXTRACTOR-{self.worker_id}] Stale flush: {flush_size} images "
                                               f"(reason={reason}, waited={time_since_last_flush:.1f}s)")
                        
                        # Check every 50ms (responsive without being too aggressive)
                        await asyncio.sleep(0.05)
                            
                    except Exception as e:
                        logger.debug(f"Stale flusher error: {e}")
                        
            except asyncio.CancelledError:
                return
        flusher_task = asyncio.create_task(_stale_flusher())

        # Background task: periodic progress logging
        async def _progress_logger():
            try:
                while self.running:
                    await asyncio.sleep(30)  # Log every 30 seconds
                    if self.processed_candidates > 0:
                        logger.info(f"[Extractor {self.worker_id}] Processed {self.processed_candidates} candidates, "
                                   f"{self.downloaded_images} downloaded, {self.cached_images} cached, "
                                   f"{self.batches_created} batches created")
            except asyncio.CancelledError:
                return
        progress_task = asyncio.create_task(_progress_logger())
        
        try:
            # Calculate number of concurrent download tasks per worker
            per_worker_concurrency = max(1, self.config.nc_extractor_concurrency // self.config.num_extractors)
            logger.info(f"[Extractor {self.worker_id}] Starting {per_worker_concurrency} concurrent download tasks")
            
            # Create concurrent download tasks
            download_tasks = [
                asyncio.create_task(self._download_worker_task(i))
                for i in range(per_worker_concurrency)
            ]
            
            # Run all tasks concurrently
            await asyncio.gather(*download_tasks, return_exceptions=True)
            
        finally:
            # Cleanup tasks
            try:
                flusher_task.cancel()
                progress_task.cancel()
            except Exception:
                pass
            
            # Flush any remaining images in shared batch (don't wait for threshold)
            remaining = await self.redis.get_shared_batch_size_async()
            if remaining > 0:
                logger.info(f"[Extractor {self.worker_id}] Flushing {remaining} remaining images from shared batch")
                # Force flush by using threshold=1
                await self.redis.flush_shared_batch_if_ready_async(1)
            self._cleanup_temp_dir()
    
    def _cleanup_temp_dir(self):
        """Cleanup temporary files with delay to prevent race condition."""
        try:
            # Don't cleanup immediately - give GPU processor time to copy files
            # Files will be cleaned by OS tmpfs or periodic cleanup
            # This prevents race condition where GPU processor can't access temp files
            logger.info(f"[Extractor {self.worker_id}] Temp files in {self.temp_dir} will be cleaned by system")
            logger.info(f"[Extractor {self.worker_id}] Delaying cleanup to prevent race condition with GPU processor")
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
            'batches_created': self.batches_created,
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
