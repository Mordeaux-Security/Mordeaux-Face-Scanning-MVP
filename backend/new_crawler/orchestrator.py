"""
Orchestrator for New Crawler System

Main coordinator with process management and back-pressure control.
Manages all worker processes and monitors system health.
"""

import asyncio
import logging
import multiprocessing
import signal
import sys
import time
from typing import List, Dict, Any, Optional
from multiprocessing import Process, Queue
import json

from .config import get_config
from .redis_manager import get_redis_manager
from .data_structures import SiteTask, ProcessingStats, SystemMetrics, CrawlResults, FaceResult
from .timing_logger import get_timing_logger

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestrator for the new crawler system."""
    
    def __init__(self):
        self.config = get_config()
        self.redis = get_redis_manager()
        self.timing_logger = get_timing_logger()
        
        # Process management
        self.crawler_processes: List[Process] = []
        self.extractor_processes: List[Process] = []
        self.gpu_processor_processes: List[Process] = []
        self.storage_processes: List[Process] = []
        
        # Back-pressure monitoring
        self._backpressure_task: Optional[asyncio.Task] = None
        self._should_throttle = False
        self.running = False
        
        # Results consumer
        self._results_consumer_task: Optional[asyncio.Task] = None
        self._consumed_results: List[FaceResult] = []
        
        # Control
        self.start_time = None
        self.site_stats: Dict[str, ProcessingStats] = {}
        
        # Metrics
        self.total_sites_processed = 0
        self.total_images_processed = 0
        self.total_faces_detected = 0
        
        # Log system start
        self.timing_logger.log_system_start()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    async def _monitor_backpressure(self):
        """Monitor queue depths and apply back-pressure."""
        while self.running:
            try:
                queue_lengths = await self.redis.get_queue_lengths_async()
                
                # Check if candidates queue is backing up
                candidates_depth = queue_lengths.get('candidates', 0)
                candidates_util = candidates_depth / self.config.nc_max_queue_depth
                
                # Check if GPU inbox queue is backing up (replaces old images queue monitoring)
                inbox_key = getattr(self.config, 'gpu_inbox_key', 'gpu:inbox')
                inbox_depth = await asyncio.to_thread(
                    self.redis._get_client().llen, inbox_key
                )
                inbox_util = inbox_depth / self.config.nc_max_queue_depth
                
                # Apply back-pressure if any queue > 75% full
                self._should_throttle = (
                    candidates_util > 0.75 or inbox_util > 0.75
                )
                
                if self._should_throttle:
                    logger.warning(f"Back-pressure active: candidates={candidates_util:.1%}, gpu_inbox={inbox_util:.1%}")
                
                await asyncio.sleep(self.config.backpressure_check_interval)
            except Exception as e:
                logger.error(f"Error monitoring back-pressure: {e}")
                await asyncio.sleep(5.0)
    
    async def _consume_results(self):
        """Continuously consume results from results queue."""
        while self.running:
            try:
                # Pop result with short timeout to allow frequent checks
                result = await asyncio.to_thread(
                    self.redis.pop_face_result, 
                    timeout=1.0
                )
                
                if result:
                    self._consumed_results.append(result)
                    # Update aggregated statistics
                    self.total_images_processed += 1
                    self.total_faces_detected += len(result.faces)
                    
                    # Diagnostic logging for result tracking
                    image_id = result.image_task.phash[:8] if (result.image_task and result.image_task.phash) else 'NO_PHASH'
                    faces_count = len(result.faces)
                    thumbs_count = len(result.thumbnail_keys) if result.thumbnail_keys else 0
                    saved_raw = result.saved_to_raw
                    
                    # Log periodically or when there are issues
                    if len(self._consumed_results) % 100 == 0:
                        logger.info(f"Consumed {len(self._consumed_results)} results from queue")
                    elif faces_count > 0 and thumbs_count == 0:
                        logger.debug(f"DIAG: Consumed result {image_id}... with {faces_count} faces but 0 thumbs saved, "
                                   f"raw={saved_raw}, saved_to_thumbs={result.saved_to_thumbs}")
                    elif faces_count != thumbs_count:
                        logger.debug(f"DIAG: Consumed result {image_id}... with {faces_count} faces but {thumbs_count} thumbs, "
                                   f"raw={saved_raw}")
                
            except Exception as e:
                logger.error(f"Error consuming results: {e}")
                await asyncio.sleep(1.0)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()
    
    def start_workers(self):
        """Start all worker processes."""
        logger.info("Starting worker processes...")
        
        # Start crawler workers
        for i in range(self.config.num_crawlers):
            process = Process(
                target=self._run_crawler_worker,
                args=(i,),
                name=f"Crawler-{i}"
            )
            process.start()
            self.crawler_processes.append(process)
            logger.info(f"Started crawler worker {i} (PID: {process.pid})")
        
        # Start extractor workers
        for i in range(self.config.num_extractors):
            process = Process(
                target=self._run_extractor_worker,
                args=(i,),
                name=f"Extractor-{i}"
            )
            process.start()
            self.extractor_processes.append(process)
            logger.info(f"Started extractor worker {i} (PID: {process.pid})")
        
        # Start GPU processor workers
        for i in range(self.config.num_gpu_processors):
            process = Process(
                target=self._run_gpu_processor_worker,
                args=(i,),
                name=f"GPU-Processor-{i}"
            )
            process.start()
            self.gpu_processor_processes.append(process)
            logger.info(f"Started GPU processor worker {i} (PID: {process.pid})")
        
        # Start storage workers
        for i in range(self.config.num_storage_workers):
            process = Process(
                target=self._run_storage_worker,
                args=(i,),
                name=f"Storage-{i}"
            )
            process.start()
            self.storage_processes.append(process)
            logger.info(f"Started storage worker {i} (PID: {process.pid})")
        
        logger.info(f"Started {len(self.crawler_processes)} crawlers, "
                   f"{len(self.extractor_processes)} extractors, "
                   f"{len(self.gpu_processor_processes)} GPU processors, "
                   f"{len(self.storage_processes)} storage workers")
    
    def stop_workers(self):
        """Stop all worker processes."""
        logger.info("Stopping worker processes...")
        
        # Stop GPU processors first
        for process in self.gpu_processor_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(f"Force killing GPU processor {process.name}")
                    process.kill()
        
        # Stop extractors
        for process in self.extractor_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(f"Force killing extractor {process.name}")
                    process.kill()
        
        # Stop crawlers
        for process in self.crawler_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(f"Force killing crawler {process.name}")
                    process.kill()
        
        # Stop storage workers
        for process in self.storage_processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
                if process.is_alive():
                    logger.warning(f"Force killing storage worker {process.name}")
                    process.kill()
        
        logger.info("All worker processes stopped")
    
    def _run_crawler_worker(self, worker_id: int):
        """Run crawler worker process."""
        from .crawler_worker import crawler_worker_process
        crawler_worker_process(worker_id)
    
    def _run_extractor_worker(self, worker_id: int):
        """Run extractor worker process."""
        from .extractor_worker import extractor_worker_process
        extractor_worker_process(worker_id)
    
    def _run_gpu_processor_worker(self, worker_id: int):
        """Run GPU processor worker process."""
        from .gpu_processor_worker import gpu_processor_worker_process
        gpu_processor_worker_process(worker_id)
    
    def _run_storage_worker(self, worker_id: int):
        """Run storage worker process."""
        from .storage_worker import storage_worker_process
        storage_worker_process(worker_id)
    
    def push_sites(self, sites: List[str]):
        """Push sites to crawl queue."""
        logger.info(f"Pushing {len(sites)} sites to crawl queue")
        
        for i, site_url in enumerate(sites):
            site_task = SiteTask(
                url=site_url,
                site_id=f"site_{i:04d}",
                max_pages=self.config.nc_max_pages_per_site,
                use_3x3_mining=self.config.nc_use_3x3_mining
            )
            
            if self.redis.push_site(site_task):
                # Initialize site stats
                self.site_stats[site_task.site_id] = ProcessingStats(
                    site_id=site_task.site_id,
                    site_url=site_task.url
                )
            else:
                logger.error(f"Failed to push site: {site_url}")
        
        logger.info(f"Successfully pushed {len(sites)} sites to queue")
    
    def monitor_queues(self) -> Dict[str, Any]:
        """Monitor queue health and metrics."""
        try:
            # Get queue metrics
            queue_metrics = self.redis.get_all_queue_metrics()
            queue_lengths = self.redis.get_queue_lengths()
            
            # Check for back-pressure
            should_apply_backpressure = self.redis.should_apply_backpressure()
            
            # Get cache stats
            cache_stats = self.redis.get_cache_stats() if hasattr(self.redis, 'get_cache_stats') else {}
            
            return {
                'queue_metrics': [qm.dict() for qm in queue_metrics],
                'queue_lengths': queue_lengths,
                'backpressure_needed': should_apply_backpressure,
                'cache_stats': cache_stats,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error monitoring queues: {e}")
            return {}
    
    async def monitor_queues_async(self) -> Dict[str, Any]:
        """Monitor queue health and metrics (async version)."""
        try:
            # Get queue metrics
            queue_metrics = await self.redis.get_all_queue_metrics_async()
            queue_lengths = await self.redis.get_queue_lengths_async()
            
            # Check for back-pressure
            should_apply_backpressure = await self.redis.should_apply_backpressure_async()
            
            # Get cache stats
            cache_stats = self.redis.get_cache_stats() if hasattr(self.redis, 'get_cache_stats') else {}
            
            return {
                'queue_metrics': [qm.dict() for qm in queue_metrics],
                'queue_lengths': queue_lengths,
                'backpressure_needed': should_apply_backpressure,
                'cache_stats': cache_stats,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Error monitoring queues (async): {e}")
            return {}
    
    def check_worker_health(self) -> Dict[str, Any]:
        """Check health of worker processes."""
        health = {
            'crawlers': [],
            'extractors': [],
            'gpu_processors': [],
            'overall_healthy': True
        }
        
        # Check crawler processes
        for i, process in enumerate(self.crawler_processes):
            is_alive = process.is_alive()
            health['crawlers'].append({
                'worker_id': i,
                'pid': process.pid,
                'alive': is_alive,
                'exitcode': process.exitcode
            })
            if not is_alive:
                health['overall_healthy'] = False
        
        # Check extractor processes
        for i, process in enumerate(self.extractor_processes):
            is_alive = process.is_alive()
            health['extractors'].append({
                'worker_id': i,
                'pid': process.pid,
                'alive': is_alive,
                'exitcode': process.exitcode
            })
            if not is_alive:
                health['overall_healthy'] = False
        
        # Check GPU processor processes
        for i, process in enumerate(self.gpu_processor_processes):
            is_alive = process.is_alive()
            health['gpu_processors'].append({
                'worker_id': i,
                'pid': process.pid,
                'alive': is_alive,
                'exitcode': process.exitcode
            })
            if not is_alive:
                health['overall_healthy'] = False
        
        return health
    
    def is_crawl_complete(self) -> bool:
        """Check if crawl is complete."""
        try:
            # Check if all queues are empty
            queue_lengths = self.redis.get_queue_lengths()
            
            # Crawl is complete if sites, candidates, and images queues are empty
            sites_empty = queue_lengths.get('sites', 0) == 0
            candidates_empty = queue_lengths.get('candidates', 0) == 0
            images_empty = queue_lengths.get('images', 0) == 0
            
            # Also check if results queue is being drained (allow small buffer)
            results_depth = queue_lengths.get('results', 0)
            results_manageable = results_depth < 50
            
            return sites_empty and candidates_empty and images_empty and results_manageable
        except Exception as e:
            logger.error(f"Error checking crawl completion: {e}")
            return False
    
    async def is_crawl_complete_async(self) -> bool:
        """Check if crawl is complete (async version)."""
        try:
            # Check if all queues are empty
            queue_lengths = await self.redis.get_queue_lengths_async()
            
            sites_empty = queue_lengths.get('sites', 0) == 0
            candidates_empty = queue_lengths.get('candidates', 0) == 0
            images_empty = queue_lengths.get('images', 0) == 0
            storage_empty = queue_lengths.get('storage', 0) == 0
            results_depth = queue_lengths.get('results', 0)
            results_manageable = results_depth < 50
            
            # Also check gpu:inbox queue
            gpu_inbox_depth = await self.redis.get_queue_length_by_key_async('gpu:inbox')
            gpu_inbox_empty = gpu_inbox_depth == 0
            
            # Check for in-flight images in GPU processor workers
            # Look for any staging or inflight batches across all workers
            gpu_inflight_images = await self._check_gpu_inflight_images()
            
            return (sites_empty and candidates_empty and images_empty and 
                    storage_empty and gpu_inbox_empty and results_manageable and 
                    gpu_inflight_images == 0)
        except Exception as e:
            logger.error(f"Error checking crawl completion (async): {e}")
            return False
    
    async def _check_gpu_inflight_images(self) -> int:
        """Check total number of images in-flight across all GPU processor workers."""
        try:
            client = await self.redis._get_async_client()
            total_inflight = 0
            
            # Check all possible worker IDs (0 to num_gpu_processors - 1)
            for worker_id in range(self.config.num_gpu_processors):
                staging_key = f"gpu:staging:{worker_id}"
                inflight_key = f"gpu:inflight:{worker_id}"
                
                # Get staging count (images waiting to be batched)
                staging_count = await client.get(staging_key)
                if staging_count:
                    total_inflight += int(staging_count)
                
                # Get inflight batch count (multiply by average batch size for image count)
                # If there are inflight batches, estimate images in flight
                inflight_batches = await client.get(inflight_key)
                if inflight_batches and int(inflight_batches) > 0:
                    # Estimate: multiply inflight batches by target batch size
                    # This gives us a conservative estimate of images in flight
                    total_inflight += int(inflight_batches) * self.config.gpu_target_batch
            
            if total_inflight > 0:
                logger.debug(f"GPU inflight images detected: {total_inflight} (staging + processing batches)")
            else:
                # Log idle state explicitly so it's visible
                logger.info(f"GPU processor IDLE: no images in staging or processing (all workers: staging=0, inflight=0)")
            
            return total_inflight
        except Exception as e:
            logger.debug(f"Error checking GPU inflight images: {e}")
            # If we can't check, assume there might be images in flight (conservative)
            return 1
    
    def get_system_metrics(self) -> SystemMetrics:
        """Get overall system metrics."""
        try:
            # Get queue metrics
            queue_metrics = self.redis.get_all_queue_metrics()
            
            # Get worker health
            worker_health = self.check_worker_health()
            
            # Get GPU worker availability
            gpu_worker_available = False
            try:
                from .gpu_interface import get_gpu_interface
                gpu_interface = get_gpu_interface()
                # Note: This is a sync method, not async
                gpu_worker_available = gpu_interface._is_available
            except:
                pass
            
            return SystemMetrics(
                active_crawlers=sum(1 for p in self.crawler_processes if p.is_alive()),
                active_extractors=sum(1 for p in self.extractor_processes if p.is_alive()),
                active_gpu_processors=sum(1 for p in self.gpu_processor_processes if p.is_alive()),
                queue_metrics=queue_metrics,
                total_sites_processed=self.total_sites_processed,
                total_images_processed=self.total_images_processed,
                total_faces_detected=self.total_faces_detected,
                gpu_worker_available=gpu_worker_available
            )
        except Exception as e:
            logger.error(f"Error getting system metrics: {e}")
            return SystemMetrics()
    
    async def monitor_loop(self):
        """Main monitoring loop with back-pressure."""
        logger.info("Starting monitoring loop")
        
        # Start back-pressure monitor
        self._backpressure_task = asyncio.create_task(self._monitor_backpressure())
        # Start results consumer
        self._results_consumer_task = asyncio.create_task(self._consume_results())
        
        try:
            idle_grace_start = None
            while self.running:
                # Check if crawl is complete
                queues_empty = await self.is_crawl_complete_async()
                # Additionally, ensure no active crawler JS tasks remain
                active_tasks = 0
                try:
                    active_tasks = await asyncio.to_thread(self.redis.get_active_task_count)
                except Exception:
                    pass
                # Fast-drain: if all sites hit image limit, purge candidate/image/gpu_inbox queues
                try:
                    if self.site_stats:
                        site_ids = list(self.site_stats.keys())
                        all_limits = await self.redis.all_sites_limit_reached_async(site_ids)
                        if all_limits:
                            await self.redis.clear_queue_async('candidates')
                            await self.redis.clear_queue_async('images')
                            await self.redis.clear_queue_async('gpu:inbox')
                            logger.info("All sites reached image limits; purged candidates/images/gpu_inbox queues for fast shutdown")
                except Exception:
                    pass
                
                # Strict limits: check each site and cleanup queues when images limit reached
                # Pages limit only stops feeding, images limit stops feeding AND clears queues
                if self.config.nc_strict_limits:
                    try:
                        if self.site_stats:
                            for site_id in list(self.site_stats.keys()):
                                # Check if site has reached limits
                                stats = await asyncio.to_thread(self.redis.get_site_stats, site_id)
                                if stats:
                                    pages_crawled = stats.get('pages_crawled', 0)
                                    images_saved_thumbs = stats.get('images_saved_thumbs', 0)
                                    
                                    # Check if pages limit reached (only stop feeding, don't clear)
                                    pages_limit_reached = (self.config.nc_max_pages_per_site > 0 and 
                                                          pages_crawled >= self.config.nc_max_pages_per_site)
                                    # Check if images limit reached (stop feeding AND clear queues)
                                    images_limit_reached = (images_saved_thumbs >= self.config.nc_max_images_per_site)
                                    
                                    # Pages limit: only set flag to stop feeding, don't clear queues
                                    if pages_limit_reached:
                                        if not await self.redis.is_site_limit_reached_async(site_id):
                                            await self.redis.set_site_limit_reached_async(site_id)
                                            logger.debug(f"Strict limits: Pages limit reached for site {site_id}, stopping new items (existing items continue processing)")
                                    
                                    # Images limit: set flag AND clear queues
                                    if images_limit_reached:
                                        # Set limit flag if not already set
                                        if not await self.redis.is_site_limit_reached_async(site_id):
                                            await self.redis.set_site_limit_reached_async(site_id)
                                        
                                        # Cleanup queues for this site (images limit reached)
                                        removed_candidates = await self.redis.remove_site_items_from_queue_async('candidates', site_id)
                                        removed_gpu = await self.redis.remove_site_items_from_queue_async('gpu:inbox', site_id)
                                        removed_storage = await self.redis.remove_site_items_from_queue_async('storage', site_id)
                                        
                                        if removed_candidates > 0 or removed_gpu > 0 or removed_storage > 0:
                                            logger.info(f"Strict limits: Images limit reached for site {site_id}, cleaned up queues - "
                                                       f"candidates={removed_candidates}, gpu:inbox={removed_gpu}, "
                                                       f"storage={removed_storage} "
                                                       f"(images={images_saved_thumbs}/{self.config.nc_max_images_per_site})")
                                        
                                        # Clear counters for this site
                                        await self.redis.clear_site_queue_counters_async(site_id)
                    except Exception as e:
                        logger.error(f"Error checking strict limits in monitor loop: {e}")
                if queues_empty and active_tasks == 0:
                    # Grace period to avoid racing with late JS callbacks
                    if idle_grace_start is None:
                        idle_grace_start = time.time()
                    if time.time() - idle_grace_start >= 5.0:
                        logger.info("Crawl complete (queues empty, no active tasks), shutting down")
                        break
                else:
                    idle_grace_start = None
                
                # Monitor queue metrics
                queue_info = await self.monitor_queues_async()
                
                # Check worker health
                worker_health = self.check_worker_health()
                
                # Log status - include gpu:inbox queue depth and GPU processor state
                gpu_inbox_depth = await self.redis.get_queue_length_by_key_async('gpu:inbox')
                gpu_inflight_images = await self._check_gpu_inflight_images()
                gpu_processor_state = "IDLE" if gpu_inflight_images == 0 else f"BUSY ({gpu_inflight_images} in-flight)"
                logger.info(f"Queues: sites={queue_info.get('queue_lengths', {}).get('sites', 0)}, "
                           f"candidates={queue_info.get('queue_lengths', {}).get('candidates', 0)}, "
                           f"images={queue_info.get('queue_lengths', {}).get('images', 0)}, "
                           f"gpu_inbox={gpu_inbox_depth}, "
                           f"storage={queue_info.get('queue_lengths', {}).get('storage', 0)}, "
                           f"results={queue_info.get('queue_lengths', {}).get('results', 0)}, "
                           f"gpu_processor={gpu_processor_state}")
                
                await asyncio.sleep(5.0)  # Reduced from 10.0 for faster completion detection
        finally:
            if self._backpressure_task:
                self._backpressure_task.cancel()
            if self._results_consumer_task:
                self._results_consumer_task.cancel()
    
    async def crawl_sites(self, sites: List[str]) -> CrawlResults:
        """Main crawl orchestration."""
        logger.info(f"Starting crawl of {len(sites)} sites")
        self.start_time = time.time()
        self.running = True
        
        # Log crawl start
        self.timing_logger.log_crawl_start()
        
        try:
            # Clear queues, stats, and shared batch
            self.redis.clear_queues()
            self.redis.clear_site_stats()
            # Clear shared batch accumulator
            try:
                client = self.redis._get_client()
                client.delete("batch:accumulator")
                logger.info("Cleared shared batch accumulator")
            except Exception as e:
                logger.warning(f"Failed to clear shared batch accumulator: {e}")
            
            # Push sites to queue
            self.push_sites(sites)
            
            # Check GPU worker availability before starting workers
            logger.info("Checking GPU worker availability...")
            try:
                from .gpu_interface import get_gpu_interface
                gpu_interface = get_gpu_interface()
                gpu_available = await gpu_interface._check_health()
                
                if not gpu_available:
                    logger.warning("GPU worker not available - will use CPU fallback")
                    logger.warning("To start GPU worker, run: .\\start-gpu-worker.ps1 -SkipDocker")
                else:
                    logger.info("✓ GPU worker is available")
            except Exception as e:
                logger.warning(f"Failed to check GPU worker availability: {e}")
                logger.warning("Will use CPU fallback")
            
            # Start workers
            self.start_workers()
            
            # Wait for workers to start
            await asyncio.sleep(2)
            
            # Start monitoring loop
            await self.monitor_loop()
            
            # Stop workers
            self.stop_workers()
            
            # Calculate final results
            total_time = time.time() - self.start_time
            
            # Get final site stats from Redis and update ProcessingStats objects
            redis_stats = self.redis.get_all_site_stats()
            for site_id, stats_dict in redis_stats.items():
                if site_id in self.site_stats:
                    site = self.site_stats[site_id]
                    site.pages_crawled = int(stats_dict.get('pages_crawled', 0))
                    site.images_found = int(stats_dict.get('images_found', 0))
                    site.images_processed = int(stats_dict.get('images_processed', 0))
                    site.faces_detected = int(stats_dict.get('faces_detected', 0))
                    site.images_saved_raw = int(stats_dict.get('images_saved_raw', 0))
                    site.images_saved_thumbs = int(stats_dict.get('images_saved_thumbs', 0))
                    site.images_skipped_limit = int(stats_dict.get('images_skipped_limit', 0))
                    site.images_cached = int(stats_dict.get('images_cached', 0))
            
            # Log what we pulled from Redis for debugging
            logger.info(f"Retrieved stats for {len(redis_stats)} sites from Redis")
            for site_id, stats_dict in redis_stats.items():
                logger.debug(f"Site {site_id}: {stats_dict}")
            
            # Get final site stats
            site_stats_list = list(self.site_stats.values())
            for stats in site_stats_list:
                stats.end_time = time.time()
                stats.total_time_seconds = total_time
            
            # Get final system metrics
            system_metrics = self.get_system_metrics()
            
            # Add validation logging with diagnostic details
            logger.info(f"Consumed {len(self._consumed_results)} results from queue")
            consumed_faces = sum(len(r.faces) for r in self._consumed_results)
            consumed_saved_raw = sum(1 for r in self._consumed_results if r.saved_to_raw)
            consumed_saved_thumbs = sum(1 for r in self._consumed_results if r.saved_to_thumbs)
            consumed_thumb_count = sum(len(r.thumbnail_keys) if r.thumbnail_keys else 0 for r in self._consumed_results)
            
            # Diagnostic: Find results with mismatches
            mismatch_count = sum(1 for r in self._consumed_results 
                                if len(r.faces) > 0 and (not r.thumbnail_keys or len(r.thumbnail_keys) < len(r.faces)))
            
            logger.info(f"From consumed results: faces={consumed_faces}, raw={consumed_saved_raw}, "
                       f"thumbs={consumed_saved_thumbs} (thumb_count={consumed_thumb_count}), "
                       f"mismatches={mismatch_count}")
            
            # Use site stats as source of truth (already pulled from Redis)
            total_images_found = sum(site.images_found for site in site_stats_list)
            total_images_processed = sum(site.images_processed for site in site_stats_list)
            total_faces_detected = sum(site.faces_detected for site in site_stats_list)
            total_images_saved_raw = sum(site.images_saved_raw for site in site_stats_list)
            total_images_saved_thumbs = sum(site.images_saved_thumbs for site in site_stats_list)
            total_images_skipped_limit = sum(site.images_skipped_limit for site in site_stats_list)
            
            logger.info(f"From Redis stats: faces={total_faces_detected}, raw={total_images_saved_raw}, thumbs={total_images_saved_thumbs}")
            logger.info(f"Total images processed: {total_images_processed}")
            
            success_rate = (total_images_processed / total_images_found * 100) if total_images_found > 0 else 0
            
            # Calculate overall throughput
            overall_images_per_second = total_images_processed / total_time if total_time > 0 else 0
            
            # Print detailed summary with throughput
            logger.info(f"=== Crawl Summary ===")
            logger.info(f"Total sites processed: {len(site_stats_list)}")
            logger.info(f"Total images processed: {total_images_processed}")
            logger.info(f"Total images saved to raw-images: {total_images_saved_raw}")
            logger.info(f"Total thumbnails saved: {total_images_saved_thumbs}")
            logger.info(f"Total images skipped due to limits: {total_images_skipped_limit}")
            logger.info(f"Success rate: {success_rate:.1f}%")
            logger.info(f"Overall throughput: {overall_images_per_second:.2f} images/second")
            
            results = CrawlResults(
                sites=site_stats_list,
                system_metrics=system_metrics,
                total_time_seconds=total_time,
                success_rate=success_rate
            )
            
            logger.info(f"Crawl completed in {total_time:.1f}s, success rate: {success_rate:.1f}%, throughput: {overall_images_per_second:.2f} images/second")
            
            # Log crawl end
            self.timing_logger.log_crawl_end(total_time * 1000, len(sites), results.total_images_processed)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during crawl: {e}")
            raise
        finally:
            self.running = False
            # Log system shutdown
            self.timing_logger.log_system_shutdown()
    
    def stop(self):
        """Stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self.running = False
        
        # Cancel tasks
        if self._backpressure_task:
            self._backpressure_task.cancel()
        if self._results_consumer_task:
            self._results_consumer_task.cancel()
        
        self.stop_workers()

