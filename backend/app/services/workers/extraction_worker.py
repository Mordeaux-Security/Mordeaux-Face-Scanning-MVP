"""
Extraction Worker Process

Worker process responsible for downloading and validating images from crawled pages,
then adding them to the batch queue for GPU processing.
"""

import asyncio
import logging
import sys
import os
import httpx
from queue import Queue

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.redis_queues import get_redis_client, get_crawled_page

# Configure logging for multiprocessing
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - Extractor-%(process)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


async def download_image(image_url: str, semaphore: asyncio.Semaphore) -> tuple:
    """
    Download an image from the given URL.
    
    Args:
        image_url: URL of the image
        semaphore: Semaphore to limit concurrent downloads
        
    Returns:
        Tuple of (image_bytes, image_info) or (None, None) on failure
    """
    async with semaphore:
        try:
            timeout = httpx.Timeout(30.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.get(image_url)
                if response.status_code == 200:
                    image_bytes = response.content
                    
                    # Basic validation
                    if len(image_bytes) > 10 * 1024 * 1024:  # 10MB limit
                        logger.warning(f"Image too large: {image_url}")
                        return None, None
                    
                    image_info = {
                        'url': image_url,
                        'size': len(image_bytes),
                        'content_type': response.headers.get('content-type', '')
                    }
                    
                    return image_bytes, image_info
                    
        except Exception as e:
            logger.error(f"Error downloading {image_url}: {e}")
            return None, None


async def process_crawled_page(page_data: dict, batch_queue_manager, worker_id: int) -> int:
    """
    Process a crawled page by downloading images and adding to batch queue.
    
    Args:
        page_data: Crawled page data dict
        batch_queue_manager: BatchQueueManager instance
        worker_id: Worker ID for logging
        
    Returns:
        Number of images processed
    """
    site = page_data.get('site', '')
    images = page_data.get('images', [])
    
    if not images:
        return 0
    
    logger.info(f"[Extractor {worker_id}] Processing {len(images)} images from {site}")
    
    # Limit concurrent downloads
    semaphore = asyncio.Semaphore(20)
    
    # Download images concurrently
    tasks = [download_image(img['url'], semaphore) for img in images]
    results = await asyncio.gather(*tasks)
    
    # Add to batch queue using the manager
    images_processed = 0
    for image_bytes, image_info in results:
        if image_bytes and image_info:
            try:
                # Add image info with site context
                image_info['site'] = site
                image_info['page_url'] = page_data.get('page_url', site)
                
                success = batch_queue_manager.add_image(image_bytes, image_info)
                if success:
                    logger.info(f"[Extractor {worker_id}] Queued image: {image_info['url']} ({len(image_bytes)} bytes)")
                    images_processed += 1
                else:
                    logger.warning(f"[Extractor {worker_id}] Failed to queue image (batch queue disabled)")
            except Exception as e:
                logger.warning(f"[Extractor {worker_id}] Failed to queue image: {e}")
    
    return images_processed


def extraction_worker(worker_id: int, redis_url: str, batch_queue_manager, max_images_per_site: int = None, site_results_list = None):
    """
    Extraction worker process main loop.
    
    Args:
        worker_id: Worker ID
        redis_url: Redis connection URL
        batch_queue_manager: BatchQueueManager instance
        max_images_per_site: Maximum images to process per site (None = unlimited) - used for tracking only
        site_results_list: Shared list of SiteResult objects to update
    """
    logger.info(f"[Extractor {worker_id}] Starting process")
    
    # Get Redis client
    redis_client = get_redis_client(redis_url)
    
    logger.info(f"[Extractor {worker_id}] Starting extraction worker loop")
    
    while True:
        try:
            # Get crawled page from queue
            page_data = get_crawled_page(redis_client, timeout=5)
            
            if page_data is None:
                # No pages in queue, continue
                continue
            
            site = page_data.get('site', 'unknown')
            logger.info(f"[Extractor {worker_id}] Processing crawled page: {site}")
            
            # Process the page (download images and add to batch queue)
            images_processed = asyncio.run(process_crawled_page(page_data, batch_queue_manager, worker_id))
            
            # Update site results
            if images_processed > 0 and site_results_list:
                for site_result in site_results_list:
                    if site_result.url == site:
                        site_result.images_processed = site_result.images_processed + images_processed
                        break
            
        except KeyboardInterrupt:
            logger.info(f"[Extractor {worker_id}] Interrupted, shutting down")
            break
        except Exception as e:
            logger.error(f"[Extractor {worker_id}] Error: {e}", exc_info=True)
            import time
            time.sleep(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True)
    parser.add_argument('--redis-url', type=str, required=True)
    parser.add_argument('--batch-queue', type=str, required=True)  # Would need serialization
    
    args = parser.parse_args()
    # Note: batch_queue would need to be passed differently in actual multiprocessing

