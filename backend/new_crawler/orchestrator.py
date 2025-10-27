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
from .data_structures import SiteTask, ProcessingStats, SystemMetrics, CrawlResults

logger = logging.getLogger(__name__)


class Orchestrator:
    """Main orchestrator for the new crawler system."""
    
    def __init__(self):
        self.config = get_config()
        self.redis = get_redis_manager()
        
        # Process management
        self.crawler_processes: List[Process] = []
        self.extractor_processes: List[Process] = []
        self.gpu_processor_processes: List[Process] = []
        
        # Back-pressure monitoring
        self._backpressure_task: Optional[asyncio.Task] = None
        self._should_throttle = False
        self.running = False
        
        # Control
        self.start_time = None
        self.site_stats: Dict[str, ProcessingStats] = {}
        
        # Metrics
        self.total_sites_processed = 0
        self.total_images_processed = 0
        self.total_faces_detected = 0
        
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
                
                # Check if image batches queue is backing up
                images_depth = queue_lengths.get('images', 0)
                images_util = images_depth / self.config.nc_max_queue_depth
                
                # Apply back-pressure if any queue > 75% full
                self._should_throttle = (
                    candidates_util > 0.75 or images_util > 0.75
                )
                
                if self._should_throttle:
                    logger.warning(f"Back-pressure active: candidates={candidates_util:.1%}, images={images_util:.1%}")
                
                await asyncio.sleep(self.config.backpressure_check_interval)
            except Exception as e:
                logger.error(f"Error monitoring back-pressure: {e}")
                await asyncio.sleep(5.0)
    
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
        
        logger.info(f"Started {len(self.crawler_processes)} crawlers, "
                   f"{len(self.extractor_processes)} extractors, "
                   f"{len(self.gpu_processor_processes)} GPU processors")
    
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
            
            # Crawl is complete if sites and candidates queues are empty
            sites_empty = queue_lengths.get('sites', 0) == 0
            candidates_empty = queue_lengths.get('candidates', 0) == 0
            
            return sites_empty and candidates_empty
        except Exception as e:
            logger.error(f"Error checking crawl completion: {e}")
            return False
    
    async def is_crawl_complete_async(self) -> bool:
        """Check if crawl is complete (async version)."""
        try:
            # Check if all queues are empty
            queue_lengths = await self.redis.get_queue_lengths_async()
            
            # Crawl is complete if sites and candidates queues are empty
            sites_empty = queue_lengths.get('sites', 0) == 0
            candidates_empty = queue_lengths.get('candidates', 0) == 0
            
            return sites_empty and candidates_empty
        except Exception as e:
            logger.error(f"Error checking crawl completion (async): {e}")
            return False
    
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
        
        try:
            while self.running:
                # Check if crawl is complete
                if await self.is_crawl_complete_async():
                    logger.info("Crawl complete, shutting down")
                    break
                
                # Monitor queue metrics
                queue_info = await self.monitor_queues_async()
                
                # Check worker health
                worker_health = self.check_worker_health()
                
                # Log status
                logger.info(f"Queues: sites={queue_info.get('queue_lengths', {}).get('sites', 0)}, "
                           f"candidates={queue_info.get('queue_lengths', {}).get('candidates', 0)}, "
                           f"images={queue_info.get('queue_lengths', {}).get('images', 0)}")
                
                await asyncio.sleep(10.0)
        finally:
            if self._backpressure_task:
                self._backpressure_task.cancel()
    
    async def crawl_sites(self, sites: List[str]) -> CrawlResults:
        """Main crawl orchestration."""
        logger.info(f"Starting crawl of {len(sites)} sites")
        self.start_time = time.time()
        self.running = True
        
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
            
            # Get final site stats
            site_stats_list = list(self.site_stats.values())
            for stats in site_stats_list:
                stats.end_time = time.time()
                stats.total_time_seconds = total_time
            
            # Get final system metrics
            system_metrics = self.get_system_metrics()
            
            # Calculate success rate
            total_images_found = sum(site.images_found for site in site_stats_list)
            total_images_processed = sum(site.images_processed for site in site_stats_list)
            total_images_saved_raw = sum(site.images_saved_raw for site in site_stats_list)
            total_images_saved_thumbs = sum(site.images_saved_thumbs for site in site_stats_list)
            total_images_skipped_limit = sum(site.images_skipped_limit for site in site_stats_list)
            success_rate = (total_images_processed / total_images_found * 100) if total_images_found > 0 else 0
            
            # Print detailed summary
            logger.info(f"=== Crawl Summary ===")
            logger.info(f"Total sites processed: {len(site_stats_list)}")
            logger.info(f"Total images processed: {total_images_processed}")
            logger.info(f"Total images saved to raw-images: {total_images_saved_raw}")
            logger.info(f"Total thumbnails saved: {total_images_saved_thumbs}")
            logger.info(f"Total images skipped due to limits: {total_images_skipped_limit}")
            logger.info(f"Success rate: {success_rate:.1f}%")
            
            results = CrawlResults(
                sites=site_stats_list,
                system_metrics=system_metrics,
                total_time_seconds=total_time,
                success_rate=success_rate
            )
            
            logger.info(f"Crawl completed in {total_time:.1f}s, success rate: {success_rate:.1f}%")
            return results
            
        except Exception as e:
            logger.error(f"Error during crawl: {e}")
            raise
        finally:
            self.running = False
    
    def stop(self):
        """Stop the orchestrator."""
        logger.info("Stopping orchestrator...")
        self.running = False
        
        # Cancel back-pressure task
        if self._backpressure_task:
            self._backpressure_task.cancel()
        
        self.stop_workers()

