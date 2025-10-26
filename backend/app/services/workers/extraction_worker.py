"""
Extraction Worker Process

Worker process responsible for downloading and validating images from crawled pages,
processing them with face detection, and saving to MinIO storage.
"""

import asyncio
import logging
import sys
import os
import httpx
from queue import Queue
from PIL import Image
import io

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.redis_queues import get_redis_client, get_crawled_page
from app.services.face import detect_faces_batch
from app.services.storage import save_raw_and_thumb_content_addressed_async
from app.core.config import get_settings

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


def crop_face_from_image(image_bytes: bytes, face_bbox: list, margin: float = 0.2) -> bytes:
    """
    Crop a face from an image using the bounding box.
    
    Args:
        image_bytes: Original image bytes
        face_bbox: Face bounding box [x1, y1, x2, y2]
        margin: Margin around face as fraction of face size
        
    Returns:
        Cropped face image bytes
    """
    try:
        # Open image
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = face_bbox
        
        # Convert to pixel coordinates
        x1 = int(x1 * width)
        y1 = int(y1 * height)
        x2 = int(x2 * width)
        y2 = int(y2 * height)
        
        # Add margin
        face_width = x2 - x1
        face_height = y2 - y1
        margin_x = int(face_width * margin)
        margin_y = int(face_height * margin)
        
        # Expand bounding box with margin
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(width, x2 + margin_x)
        y2 = min(height, y2 + margin_y)
        
        # Crop the face
        face_image = image.crop((x1, y1, x2, y2))
        
        # Convert to bytes
        output = io.BytesIO()
        face_image.save(output, format='JPEG', quality=90)
        return output.getvalue()
        
    except Exception as e:
        logger.error(f"Error cropping face: {e}")
        return image_bytes  # Return original if cropping fails


async def process_crawled_page(page_data: dict, max_images_per_site: int, site_results_dict, batch_queue_manager, worker_id: int) -> int:
    """
    Process a crawled page by downloading images, detecting faces, and saving to MinIO.
    
    Args:
        page_data: Crawled page data dict
        max_images_per_site: Maximum images to process per site
        site_results_dict: Shared dict of SiteResult objects to update
        worker_id: Worker ID for logging
        
    Returns:
        Number of images processed
    """
    site = page_data.get('site', '')
    images = page_data.get('images', [])
    
    if not images:
        return 0
    
    logger.info(f"[Extractor {worker_id}] Processing {len(images)} images from {site}")
    logger.info(f"[DATAFLOW] Extractor {worker_id} ← Redis: Site={site}, Images={len(images)}")
    
    # Check current thumbnail count for this site (for limiting thumbnail saves only)
    current_thumbnails = 0
    if site_results_dict and site in site_results_dict:
        current_thumbnails = site_results_dict[site]['thumbnails_saved']
    
    # Process all images regardless of thumbnail limit - only limit thumbnail saves
    logger.info(f"[Extractor {worker_id}] Processing all {len(images)} images from {site} (current thumbnails: {current_thumbnails}, limit: {max_images_per_site})")
    
    # Process images in batches - download and add to batch queue immediately
    images_processed = 0
    raw_images_saved = 0
    thumbnails_saved = 0
    
    # Process images in batches for better coordination
    batch_size = 16  # Smaller batches for better coordination
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        
        # Download batch
        batch_results = []
        for img in batch_images:
            try:
                # Create a simple semaphore for sequential downloads
                semaphore = asyncio.Semaphore(1)
                result = await download_image(img['url'], semaphore)
                batch_results.append(result)
            except Exception as e:
                logger.error(f"[Extractor {worker_id}] Error downloading {img['url']}: {e}")
                batch_results.append((None, None))
        
        # Filter out failed downloads
        valid_images = [(img_bytes, img_info) for img_bytes, img_info in batch_results if img_bytes and img_info]
        
        if not valid_images:
            logger.warning(f"[Extractor {worker_id}] No valid images in batch {i//batch_size + 1}")
            continue
        
        batch_num = i//batch_size + 1
        logger.info(f"[Extractor {worker_id}] Processing batch {batch_num} with {len(valid_images)} images")
        logger.info(f"[DATAFLOW] Extractor {worker_id} → BatchQueue: Site={site}, BatchSize={len(valid_images)}, BatchNum={batch_num}")
        
        # Add images to batch queue for GPU processing
        try:
            # Add each image individually to the batch queue
            for img_bytes, img_info in valid_images:
                # Add site information to img_info
                img_info['site'] = site
                batch_queue_manager.add_image(img_bytes, img_info)
            
            logger.info(f"[Extractor {worker_id}] Added batch of {len(valid_images)} images to GPU queue")
            
            # Update processed count
            images_processed += len(valid_images)
            
            # Update site results immediately for better coordination
            if site_results_dict and site in site_results_dict:
                stats = site_results_dict[site]
                stats['images_processed'] += len(valid_images)
                site_results_dict[site] = stats  # Explicit reassignment for sync
                
                logger.info(f"[DATAFLOW] Extractor {worker_id} → Stats Updated: Site={site}, Processed={images_processed}, RawSaved={raw_images_saved}, ThumbsSaved={thumbnails_saved}")
            
        except Exception as e:
            logger.error(f"[Extractor {worker_id}] Error adding batch to queue: {e}")
    
    return images_processed


def extraction_worker(worker_id: int, redis_url: str, batch_queue_manager, max_images_per_site: int = None, site_results_dict = None):
    """
    Extraction worker process main loop.
    
    Args:
        worker_id: Worker ID
        redis_url: Redis connection URL
        batch_queue_manager: BatchQueueManager instance (unused in new implementation)
        max_images_per_site: Maximum images to process per site (None = unlimited)
        site_results_dict: Shared dict of SiteResult objects to update
    """
    logger.info(f"[Extractor {worker_id}] Starting process")
    
    # Get Redis client
    redis_client = get_redis_client(redis_url)
    
    logger.info(f"[Extractor {worker_id}] Starting extraction worker loop")
    
    # Create new event loop for this process (don't close it in the loop)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        while True:
            try:
                # Get crawled page from queue
                page_data = get_crawled_page(redis_client, timeout=5)
                
                if page_data is None:
                    # No pages in queue, continue
                    continue
                
                site = page_data.get('site', 'unknown')
                logger.info(f"[Extractor {worker_id}] Processing crawled page: {site}")
                
                # Process the page (download images, detect faces, save to MinIO)
                images_processed = loop.run_until_complete(
                    process_crawled_page(page_data, max_images_per_site, site_results_dict, batch_queue_manager, worker_id)
                )
                
            except KeyboardInterrupt:
                logger.info(f"[Extractor {worker_id}] Interrupted, shutting down")
                break
            except Exception as e:
                logger.error(f"[Extractor {worker_id}] Error: {e}", exc_info=True)
                import time
                time.sleep(1)
    finally:
        # Close loop only when worker is shutting down
        loop.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True)
    parser.add_argument('--redis-url', type=str, required=True)
    parser.add_argument('--batch-queue', type=str, required=True)  # Would need serialization
    
    args = parser.parse_args()
    # Note: batch_queue would need to be passed differently in actual multiprocessing

