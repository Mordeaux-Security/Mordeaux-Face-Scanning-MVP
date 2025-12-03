"""
Crawler Worker for New Crawler System

HTML fetching and selector mining worker process.
Handles site crawling with 3x3 mining and pushes candidates to Redis queue.
"""

import asyncio
import logging
import multiprocessing
import os
import signal
import sys
import time
from typing import List, Optional

from .config import get_config
from .redis_manager import get_redis_manager
from .selector_miner import get_selector_miner
from .http_utils import get_http_utils
from .data_structures import SiteTask, CandidateImage, CandidatePost, TaskStatus
from .timing_logger import get_timing_logger

logger = logging.getLogger(__name__)


class CrawlerWorker:
    """Crawler worker for HTML fetching and selector mining."""
    
    def __init__(self, worker_id: int):
        self.worker_id = worker_id
        self.config = get_config()
        self.redis = get_redis_manager()
        self.selector_miner = get_selector_miner()
        self.http_utils = get_http_utils()
        self.timing_logger = get_timing_logger()
        
        # Worker state
        self.running = False
        self.processed_sites = 0
        self.total_candidates = 0
        self._shutdown_event = asyncio.Event()
        # Track active tasks via Redis for orchestrator coordination
        self._active_tasks_incr = self.redis.incr_active_tasks
        self._active_tasks_decr = self.redis.decr_active_tasks
        
    async def _enqueue_candidates(self, candidates) -> int:
        """Enqueue a batch of candidates to Redis queue."""
        # Handle both CandidateImage and CandidatePost types
        candidate_type = type(candidates[0]).__name__ if candidates else None

        # Filter out candidates from sites that reached limit
        valid_candidates = []
        for candidate in candidates:
            # Skip enqueue if site limit already reached
            if await self.redis.is_site_limit_reached_async(candidate.site_id):
                url_attr = getattr(candidate, 'img_url', getattr(candidate, 'post_url', 'unknown'))
                logger.debug(f"[Crawler {self.worker_id}] Not enqueueing candidate (limit reached): {url_attr}")
                continue

            # If strict_limits enabled, also check if site has reached pages limit
            if self.config.nc_strict_limits:
                stats = await asyncio.to_thread(self.redis.get_site_stats, candidate.site_id)
                if stats:
                    pages_crawled = stats.get('pages_crawled', 0)
                    if self.config.nc_max_pages_per_site > 0 and pages_crawled >= self.config.nc_max_pages_per_site:
                        url_attr = getattr(candidate, 'img_url', getattr(candidate, 'post_url', 'unknown'))
                        logger.debug(f"[Crawler {self.worker_id}] Not enqueueing candidate (pages limit reached): {url_attr}")
                        continue

            valid_candidates.append(candidate)

        if not valid_candidates:
            return 0

        # Batch push all valid candidates at once
        queue_name = self.config.get_queue_name('candidates')
        payloads = [self.redis._serialize(c) for c in valid_candidates]
        pushed = await asyncio.to_thread(self.redis.push_many, queue_name, payloads)

        # Diagnostic logging
        if self.config.nc_diagnostic_logging:
            queue_depth = await self.redis.get_queue_length_by_key_async(queue_name)

            # Track candidates/second rate
            now = time.time()
            if not hasattr(self, '_last_enqueue_time'):
                self._last_enqueue_time = now
                self._last_enqueue_count = 0
                self._enqueue_count_window = 0
                self._enqueue_time_window = now

            self._enqueue_count_window += pushed
            elapsed = now - self._enqueue_time_window
            if elapsed >= 1.0:  # Calculate rate every second
                candidates_per_sec = self._enqueue_count_window / elapsed if elapsed > 0 else 0
                logger.info(f"[CRAWLER-DIAG-{self.worker_id}] Throughput: {candidates_per_sec:.1f} candidates/sec, "
                          f"queue_depth={queue_depth}")
                self._enqueue_count_window = 0
                self._enqueue_time_window = now

            logger.debug(f"[CRAWLER-DIAG-{self.worker_id}] Pushed {pushed} {candidate_type}s, queue_depth={queue_depth}")

        return pushed if pushed else 0

    async def process_site(self, site_task: SiteTask) -> int:
        """Process a single site task."""
        site_start_time = time.time()
        try:
            logger.debug(f"[CRAWLER-{self.worker_id}] Starting site: {site_task.url}, max_pages={site_task.max_pages}")
            
            # Log site start
            self.timing_logger.log_site_start(site_task.site_id, site_task.url)
            
            # Mine posts for diabetes mentions using streaming approach
            # Increment active tasks counter for duration of JS/HTTP crawl
            self._active_tasks_incr(1)
            try:
                enqueue_tasks = []
                total_pages = 0
                async for page_url, page_candidates in self.selector_miner.mine_posts_with_3x3_crawl(
                    site_task.url, site_task.site_id, site_task.max_pages
                ):
                    # Log page start/end timing
                    page_start_time = time.time()
                    self.timing_logger.log_page_start(site_task.site_id, page_url)

                    # Enqueue candidates in background while we continue crawling
                    task = asyncio.create_task(self._enqueue_candidates(page_candidates))
                    enqueue_tasks.append(task)

                    # Log page end
                    page_duration = (time.time() - page_start_time) * 1000
                    self.timing_logger.log_page_end(site_task.site_id, page_url, page_duration, len(page_candidates))
                    total_pages += 1

                # After crawl completes, wait for all enqueues to finish
                results = await asyncio.gather(*enqueue_tasks)
                total_enqueued = sum(results)
                
                # Persist accurate pages crawled and images found for this site
                await self.redis.update_site_stats_async(
                    site_task.site_id,
                    {
                        'pages_crawled': total_pages,
                        'images_found': total_enqueued
                    }
                )
                
                # Check if pages limit reached - only stop feeding, don't clear queues
                # NOTE: We do NOT set site limit flag here because that would block extractor
                # from processing existing candidates. The site limit flag should only be set
                # when IMAGES limit is reached (handled in orchestrator/storage worker).
                # For pages limit, we only stop feeding new candidates (handled in _enqueue_candidates).
                if self.config.nc_strict_limits:
                    if self.config.nc_max_pages_per_site > 0 and total_pages >= self.config.nc_max_pages_per_site:
                        logger.info(f"[Crawler {self.worker_id}] Pages limit reached for site {site_task.site_id}, stopping new candidates (existing items will continue processing)")
                
            finally:
                self._active_tasks_decr(1)
            
            # Log site end
            site_duration = (time.time() - site_start_time) * 1000
            self.timing_logger.log_site_end(site_task.site_id, site_duration, total_pages, total_enqueued)
            
            logger.debug(f"[CRAWLER-{self.worker_id}] Site complete: {site_task.url}, found {total_enqueued} candidates")
            return total_enqueued
            
        except Exception as e:
            logger.error(f"[Crawler {self.worker_id}] Error processing site {site_task.url}: {e}")
            return 0
    
    async def _process_site_task(self, site_task: SiteTask):
        """Process a single site task with statistics tracking."""
        try:
            # Process site
            candidates_count = await self.process_site(site_task)
            
            # Update statistics in Redis
            await self.redis.update_site_stats_async(
                site_task.site_id,
                {
                    'images_found': candidates_count
                }
            )
            
            # Update local statistics
            self.processed_sites += 1
            self.total_candidates += candidates_count
            
            logger.info(f"[Crawler {self.worker_id}] Completed site {site_task.site_id}: "
                       f"{candidates_count} candidates. Total: {self.processed_sites} sites, "
                       f"{self.total_candidates} candidates")
        
        except Exception as e:
            logger.error(f"[Crawler {self.worker_id}] Error processing site {site_task.url}: {e}")
    
    async def run(self):
        """Main worker loop with concurrent site processing."""
        logger.info(f"[Crawler {self.worker_id}] Starting worker with concurrent site processing")
        self.running = True
        
        # Track active site processing tasks
        active_site_tasks = set()
        max_concurrent_sites = self.config.nc_max_concurrent_sites_per_worker
        
        while self.running and not self._shutdown_event.is_set():
            try:
                # Clean up completed tasks
                active_site_tasks = {t for t in active_site_tasks if not t.done()}
                
                # If we have capacity, try to pop a new site
                if len(active_site_tasks) < max_concurrent_sites:
                    site_task = await asyncio.to_thread(
                        self.redis.pop_site, timeout=2.0  # Reduced from 10.0
                    )
                    
                    if site_task:
                        # Start processing site in background
                        task = asyncio.create_task(self._process_site_task(site_task))
                        active_site_tasks.add(task)
                        logger.info(f"[Crawler {self.worker_id}] Started site {site_task.site_id} "
                                   f"({len(active_site_tasks)}/{max_concurrent_sites} active)")
                
                # Brief pause before checking for more sites
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info(f"[Crawler {self.worker_id}] Cancelled, shutting down")
                break
            except Exception as e:
                logger.error(f"[Crawler {self.worker_id}] Error in main loop: {e}")
                await asyncio.sleep(1)
        
        # Wait for active tasks to complete
        if active_site_tasks:
            logger.info(f"[Crawler {self.worker_id}] Waiting for {len(active_site_tasks)} active sites to complete")
            await asyncio.gather(*active_site_tasks, return_exceptions=True)
        
        # Cleanup
        await self.http_utils.close()
        logger.info(f"[Crawler {self.worker_id}] Worker stopped")
    
    def stop(self):
        """Stop the worker."""
        self.running = False
        self._shutdown_event.set()
    
    def get_stats(self) -> dict:
        """Get worker statistics."""
        return {
            'worker_id': self.worker_id,
            'processed_sites': self.processed_sites,
            'total_candidates': self.total_candidates,
            'running': self.running
        }


def crawler_worker_process(worker_id: int):
    """Crawler worker process entry point."""
    # Reset signal handlers to default (let parent handle)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    
    # Configure logging for multiprocessing
    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - Crawler-{worker_id} - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"[Crawler {worker_id}] Starting crawler worker process")
    
    try:
        # Create worker
        worker = CrawlerWorker(worker_id)
        
        # Create new event loop for this process
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Run worker
            loop.run_until_complete(worker.run())
        finally:
            # Close loop
            loop.close()
            
    except Exception as e:
        logger.error(f"[Crawler {worker_id}] Fatal error: {e}", exc_info=True)
    finally:
        logger.info(f"[Crawler {worker_id}] Crawler worker process ended")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True)
    
    args = parser.parse_args()
    crawler_worker_process(args.worker_id)
