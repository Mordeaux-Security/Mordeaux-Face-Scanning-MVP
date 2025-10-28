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
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(self.config.nc_extractor_concurrency)
        
        # Create temp directory
        self.temp_dir = tempfile.mkdtemp(prefix=f"extractor_{worker_id}_")
        logger.info(f"[Extractor {self.worker_id}] Using temp directory: {self.temp_dir}")
    
    async def process_candidate(self, candidate: CandidateImage) -> Optional[ImageTask]:
        """Process a single candidate image."""
        extraction_start_time = time.time()
        async with self._semaphore:  # Limit concurrent downloads
            try:
                logger.debug(f"[Extractor {self.worker_id}] Processing candidate: {candidate.img_url}")
                
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
    
    
    
    async def run(self):
        """Main worker loop."""
        logger.info(f"[Extractor {self.worker_id}] Starting extractor worker")
        self.running = True
        
        # Background task: time-based stale batch flush
        async def _stale_flusher():
            try:
                while self.running:
                    try:
                        gpu_idle = await self.redis.is_gpu_idle_async()
                        batch_size = await self.redis.get_shared_batch_size_async()
                        
                        # GPU idle + min batch ready = flush immediately
                        if gpu_idle and batch_size >= self.config.gpu_min_batch_size:
                            # Flush entire batch (up to max) when GPU is idle
                            flush_size = min(batch_size, self.batch_size_threshold)
                            flushed = await self.redis.flush_shared_batch_if_ready_async(flush_size)
                            if flushed:
                                logger.info(f"[EXTRACTOR-{self.worker_id}] GPU-idle flush: {batch_size} images")
                        
                        # Check every 50ms regardless of GPU status
                        await asyncio.sleep(0.05)
                            
                    except Exception:
                        pass
                        
            except asyncio.CancelledError:
                return
        flusher_task = asyncio.create_task(_stale_flusher())

        # Track first and last extraction times per site
        site_extraction_times = {}  # {site_id: {'start': datetime, 'end': datetime}}
        
        try:
            while self.running:
                # Pop candidate
                candidate = await asyncio.to_thread(
                    self.redis.pop_candidate, timeout=2.0
                )
                
                if candidate is None:
                    await asyncio.sleep(0.1)
                    continue
                
                site_id = candidate.site_id
                # If site already reached limit, drop immediately (log once per site)
                if await self.redis.is_site_limit_reached_async(site_id):
                    if site_id not in site_extraction_times:
                        logger.info(f"[EXTRACTOR-{self.worker_id}] Dropping candidate for {site_id} (limit reached)")
                        site_extraction_times[site_id] = {'start': None, 'end': None}
                    else:
                        logger.debug(f"[EXTRACTOR-{self.worker_id}] Skipping candidate (limit reached): {candidate.img_url}")
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
                
                # Process candidate
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
                
                # Log progress periodically
                if self.processed_candidates % 100 == 0 and self.processed_candidates > 0:
                    logger.info(f"[Extractor {self.worker_id}] Processed {self.processed_candidates} candidates, "
                               f"{self.downloaded_images} downloaded, {self.cached_images} cached, "
                               f"{self.batches_created} batches created")
        finally:
            try:
                flusher_task.cancel()
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
            'current_batch_size': len(self.current_batch),
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
