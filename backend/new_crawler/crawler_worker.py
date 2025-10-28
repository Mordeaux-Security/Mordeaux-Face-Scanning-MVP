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
from .data_structures import SiteTask, CandidateImage, TaskStatus
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
        
    async def _enqueue_candidates(self, candidates: List[CandidateImage]) -> int:
        """Enqueue a batch of candidates to Redis queue."""
        candidates_pushed = 0
        for candidate in candidates:
            # Skip enqueue if site limit already reached
            if await self.redis.is_site_limit_reached_async(candidate.site_id):
                logger.debug(f"[Crawler {self.worker_id}] Not enqueueing candidate (limit reached): {candidate.img_url}")
                continue
            if await self.redis.push_candidate_async(candidate):
                candidates_pushed += 1
            else:
                logger.warning(f"[Crawler {self.worker_id}] Failed to push candidate: {candidate.img_url}")
        return candidates_pushed

    async def process_site(self, site_task: SiteTask) -> int:
        """Process a single site task."""
        site_start_time = time.time()
        try:
            logger.debug(f"[CRAWLER-{self.worker_id}] Starting site: {site_task.url}, max_pages={site_task.max_pages}")
            
            # Log site start
            self.timing_logger.log_site_start(site_task.site_id, site_task.url)
            
            # Mine selectors from site using streaming approach
            # Increment active tasks counter for duration of JS/HTTP crawl
            self._active_tasks_incr(1)
            try:
                enqueue_tasks = []
                total_pages = 0
                async for page_url, page_candidates in self.selector_miner.mine_with_3x3_crawl(
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
    
    async def run(self):
        """Main worker loop."""
        logger.info(f"[Crawler {self.worker_id}] Starting worker")
        self.running = True
        
        while self.running and not self._shutdown_event.is_set():
            try:
                # Non-blocking pop with timeout
                site_task = await asyncio.to_thread(
                    self.redis.pop_site, timeout=10.0  # Increased from 2.0 to 10.0 seconds
                )
                
                if site_task is None:
                    # Check shutdown every 2 seconds
                    await asyncio.sleep(0.1)
                    continue
                
                # Process site
                candidates_count = await self.process_site(site_task)
                
                # Update statistics in Redis
                await self.redis.update_site_stats_async(
                    site_task.site_id,
                    {
                        'pages_crawled': 1,  # Each site task represents 1 page crawled
                        'images_found': candidates_count
                    }
                )
                
                # Update local statistics
                self.processed_sites += 1
                self.total_candidates += candidates_count
                
                logger.info(f"[Crawler {self.worker_id}] Processed {self.processed_sites} sites, "
                           f"{self.total_candidates} total candidates")
                
            except asyncio.CancelledError:
                logger.info(f"[Crawler {self.worker_id}] Cancelled, shutting down")
                break
            except Exception as e:
                logger.error(f"[Crawler {self.worker_id}] Error in main loop: {e}")
                await asyncio.sleep(1)
        
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
