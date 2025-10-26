"""
Batch Processor Worker

Worker process responsible for collecting and batching images for GPU processing.
Manages the batch queue and coordinates with GPU workers.
"""

import logging
import sys
import os
import time
from queue import Queue
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.redis_queues import get_redis_client, get_extraction_result

# Configure logging for multiprocessing
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - Batch-Processor-%(process)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def batch_processor(worker_id: int, redis_url: str, batch_queue: Queue, batch_size: int = 64, site_results_list = None):
    """
    Batch processor worker process main loop.
    
    Args:
        worker_id: Worker ID
        redis_url: Redis connection URL
        batch_queue: Queue for batched images (from multiprocessing.Manager)
        batch_size: Size of batches to create
        site_results_list: Shared list of SiteResult objects to update
    """
    logger.info(f"[Batch Processor {worker_id}] Starting process")
    
    # Get Redis client
    redis_client = get_redis_client(redis_url)
    
    logger.info(f"[Batch Processor {worker_id}] Starting batch processor loop")
    
    current_batch = []
    last_batch_time = time.time()
    batch_timeout = 2.0  # Flush batch after 2 seconds of inactivity
    
    while True:
        try:
            # Try to get extraction result from Redis
            result = get_extraction_result(redis_client, timeout=1)
            
            if result:
                current_batch.append(result)
                logger.info(f"[Batch Processor {worker_id}] Added to batch: {result.get('image_url', 'unknown')}")
            
            # Check if we should flush the batch
            current_time = time.time()
            should_flush = (
                len(current_batch) >= batch_size or
                (current_batch and current_time - last_batch_time > batch_timeout)
            )
            
            if should_flush:
                if current_batch:
                    logger.info(f"[Batch Processor {worker_id}] Flushing batch of {len(current_batch)} items")
                    
                    try:
                        batch_queue.put(current_batch, timeout=1.0)
                        logger.info(f"[Batch Processor {worker_id}] Batch queued successfully")
                        
                        # Update site results with batch processing info
                        if site_results_list:
                            # Count images per site in this batch
                            site_counts = {}
                            for item in current_batch:
                                site = item.get('site', 'unknown')
                                site_counts[site] = site_counts.get(site, 0) + 1
                            
                            # Update site results
                            for site, count in site_counts.items():
                                for site_result in site_results_list:
                                    if site_result.url == site:
                                        # This represents images that went through GPU processing
                                        # (closer to "thumbnails saved" concept)
                                        site_result.images_processed = site_result.images_processed + count
                                        break
                        
                    except Exception as e:
                        logger.warning(f"[Batch Processor {worker_id}] Failed to queue batch: {e}")
                    
                    current_batch = []
                    last_batch_time = current_time
                
        except KeyboardInterrupt:
            logger.info(f"[Batch Processor {worker_id}] Interrupted, shutting down")
            break
        except Exception as e:
            logger.error(f"[Batch Processor {worker_id}] Error: {e}", exc_info=True)
            time.sleep(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True)
    parser.add_argument('--redis-url', type=str, required=True)
    parser.add_argument('--batch-queue', type=str, required=True)  # Would need serialization
    parser.add_argument('--batch-size', type=int, default=64)
    
    args = parser.parse_args()
    # Note: batch_queue would need to be passed differently in actual multiprocessing


