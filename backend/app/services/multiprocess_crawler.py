"""
Multiprocessing Crawler Coordinator

Coordinates multiple worker processes for parallel crawling and image extraction.
Uses Redis queues for inter-process communication and a shared batch queue for GPU processing.
"""

import logging
import multiprocessing
import sys
import os
import time
from typing import List, Optional, Dict, Any
from queue import Queue
from dataclasses import dataclass

# Configure logging for main process
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)


@dataclass
class SiteResult:
    """Result for a single site crawl."""
    url: str
    images_found: int = 0
    images_processed: int = 0
    pages_crawled: int = 0
    errors: List[str] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class CrawlResults:
    """Results for the entire multisite crawl."""
    sites: List[SiteResult]
    total_time: float
    batch_stats: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.batch_stats is None:
            self.batch_stats = {}


# Import workers
from .workers.crawling_worker import crawling_worker
from .workers.extraction_worker import extraction_worker
from .workers.batch_processor import batch_processor
from .redis_queues import get_redis_client, setup_redis_queues, push_sites_to_queue
from .batch_queue_manager import BatchQueueManager


class MultiprocessCrawler:
    """
    Multiprocessing crawler coordinator.
    
    Manages crawling workers, extraction workers, and batch queue processing.
    Uses Redis for inter-process communication and a shared queue for batching.
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        num_crawlers: int = 5,
        num_extractors: int = 1,
        num_batch_processors: int = 1,
        batch_size: int = 64,
        use_3x3_mining: bool = False,
        max_pages: Optional[int] = None,
        max_images_per_site: Optional[int] = None
    ):
        """
        Initialize the multiprocessing crawler.
        
        Args:
            redis_url: Redis connection URL
            num_crawlers: Number of crawling worker processes
            num_extractors: Number of extraction worker processes
            num_batch_processors: Number of batch processing workers
            batch_size: Batch size for GPU processing
            use_3x3_mining: Whether to enable 3x3 mining
            max_pages: Maximum pages to crawl per site (None = unlimited)
            max_images_per_site: Maximum images to process per site (None = unlimited)
            
        Note: GPU processing is handled by the native Windows GPU worker service
        accessible via HTTP. No local GPU worker processes are spawned.
        """
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://redis:6379/0')
        self.num_crawlers = num_crawlers
        self.num_extractors = num_extractors
        self.num_batch_processors = num_batch_processors
        self.batch_size = batch_size
        self.use_3x3_mining = use_3x3_mining
        self.max_pages = max_pages
        self.max_images_per_site = max_images_per_site
        
        # Worker processes
        self.crawler_processes: List[multiprocessing.Process] = []
        self.extractor_processes: List[multiprocessing.Process] = []
        self.batch_processes: List[multiprocessing.Process] = []
        
        # Shared batch queue
        manager = multiprocessing.Manager()
        self.batch_queue = manager.Queue(maxsize=10)
        
        # Batch queue manager
        self.batch_manager: Optional[BatchQueueManager] = None
        
        logger.info(f"MultiprocessCrawler initialized with {num_crawlers} crawlers, {num_extractors} extractors, {num_batch_processors} batch processors")
    
    def start_workers(self, site_results_list):
        """Start all worker processes."""
        logger.info("Starting worker processes...")
        
        # Start batch queue manager first
        self.batch_manager = BatchQueueManager(
            batch_size=self.batch_size,
            max_queue_depth=10,
            flush_timeout=2.0,
            enabled=True,
            shared_queue=self.batch_queue,
            max_images_per_site=self.max_images_per_site
        )
        self.batch_manager.start()
        logger.info("Batch queue manager started")
        
        # Start crawling workers
        for i in range(self.num_crawlers):
            process = multiprocessing.Process(
                target=crawling_worker,
                args=(i, self.redis_url, self.use_3x3_mining, self.max_pages, site_results_list),
                name=f"Crawler-{i}"
            )
            process.start()
            self.crawler_processes.append(process)
            logger.info(f"Started crawling worker {i}")
        
        # Start extraction workers
        for i in range(self.num_extractors):
            process = multiprocessing.Process(
                target=extraction_worker,
                args=(i, self.redis_url, self.batch_manager, self.max_images_per_site, site_results_list),
                name=f"Extractor-{i}"
            )
            process.start()
            self.extractor_processes.append(process)
            logger.info(f"Started extraction worker {i}")
        
        # Start batch processors
        for i in range(self.num_batch_processors):
            process = multiprocessing.Process(
                target=batch_processor,
                args=(i, self.redis_url, self.batch_queue, self.batch_size, site_results_list),
                name=f"Batch-Processor-{i}"
            )
            process.start()
            self.batch_processes.append(process)
            logger.info(f"Started batch processor {i}")
    
    def stop_workers(self):
        """Stop all worker processes."""
        logger.info("Stopping worker processes...")
        
        # Stop batch queue manager
        if self.batch_manager:
            self.batch_manager.stop()
        
        # Terminate and join processes
        all_processes = (self.crawler_processes + self.extractor_processes + 
                         self.batch_processes)
        for process in all_processes:
            process.terminate()
            process.join(timeout=5.0)
            if process.is_alive():
                logger.warning(f"Force killing process {process.name}")
                process.kill()
        
        logger.info("All worker processes stopped")
    
    def crawl_sites(self, sites: List[str]) -> CrawlResults:
        """
        Start a crawl of multiple sites.
        
        Args:
            sites: List of site URLs to crawl
            
        Returns:
            CrawlResults with detailed statistics
        """
        logger.info(f"Starting crawl of {len(sites)} sites")
        start_time = time.time()
        
        # Initialize results
        manager = multiprocessing.Manager()
        site_results_list = manager.list([SiteResult(url=site) for site in sites])
        
        # Setup Redis queues
        redis_client = get_redis_client(self.redis_url)
        
        # Clear Redis queues first (BEFORE pushing sites)
        logger.info("Clearing Redis queues...")
        setup_redis_queues(redis_client)
        
        # Push sites to queue
        logger.info(f"Pushing {len(sites)} sites to queue...")
        push_sites_to_queue(redis_client, sites)
        
        # Verify sites were pushed
        queue_length = redis_client.llen('sites_to_crawl')
        logger.info(f"Redis queue 'sites_to_crawl' length: {queue_length}")
        
        # Start workers
        self.start_workers(site_results_list)
        
        logger.info("Crawl started. Workers are processing sites...")
        
        try:
            # Wait for all sites to be processed
            while True:
                queue_length = redis_client.llen('sites_to_crawl')
                if queue_length == 0:
                    logger.info("All sites processed")
                    break
                
                time.sleep(5)  # Check every 5 seconds
                
        except KeyboardInterrupt:
            logger.info("Crawl interrupted by user")
        finally:
            # Stop workers
            self.stop_workers()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Collect batch statistics if available
        batch_stats = {}
        if self.batch_manager:
            batch_stats = self.batch_manager.get_stats()
        
        # Convert manager.list to regular list for return
        results = CrawlResults(
            sites=list(site_results_list),
            total_time=total_time,
            batch_stats=batch_stats
        )
        
        return results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_workers() 