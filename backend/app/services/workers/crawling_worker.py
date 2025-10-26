"""
Crawling Worker Process

Worker process responsible for fetching HTML, parsing, and selector mining.
Pushes crawled page data to Redis queue for extraction workers.
"""

import asyncio
import logging
import sys
import os
from urllib.parse import urljoin

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.redis_queues import get_redis_client, get_site_from_queue, push_crawled_page
from app.services.http_service import fetch_html_with_redirects
from app.services.selector_mining import SelectorMiningService
from bs4 import BeautifulSoup

# Configure logging for multiprocessing
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - Crawler-%(process)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


async def crawl_site(site_url: str, use_3x3_mining: bool = False, max_pages: int = None) -> dict:
    """
    Crawl a single site.
    
    Args:
        site_url: URL to crawl
        use_3x3_mining: Whether to enable 3x3 mining
        max_pages: Maximum pages to crawl (None = unlimited)
        
    Returns:
        Dict with site, html, and images found
    """
    try:
        # Create HTTP client
        import httpx
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Fetch HTML
            html, errors = await fetch_html_with_redirects(site_url, client)
            if not html:
                logger.error(f"Failed to fetch {site_url}: {errors}")
                return None
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract images
            images = []
            if use_3x3_mining:
                # Use 3x3 mining for better selector discovery
                # Note: SelectorMiningService doesn't have mine_images method
                # We'll use simple extraction for now, but with proper URL conversion
                logger.warning("3x3 mining requested but mine_images method not available, using simple extraction")
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        # Convert relative URLs to absolute URLs
                        absolute_url = urljoin(site_url, src)
                        # Only include http/https URLs
                        if absolute_url.startswith(('http://', 'https://')):
                            images.append({
                                'url': absolute_url,
                                'alt': img.get('alt', ''),
                                'width': img.get('width'),
                                'height': img.get('height')
                            })
            else:
                # Simple extraction
                for img in soup.find_all('img'):
                    src = img.get('src') or img.get('data-src') or img.get('data-lazy-src')
                    if src:
                        # Convert relative URLs to absolute URLs
                        absolute_url = urljoin(site_url, src)
                        # Only include http/https URLs
                        if absolute_url.startswith(('http://', 'https://')):
                            images.append({
                                'url': absolute_url,
                                'alt': img.get('alt', ''),
                                'width': img.get('width'),
                                'height': img.get('height')
                            })
            
            # Apply page limit if specified
            pages_crawled = 1  # We crawled the main page
            if max_pages and pages_crawled >= max_pages:
                logger.info(f"Reached page limit ({max_pages}) for {site_url}")
                # Truncate images if we hit page limit
                if len(images) > 0:
                    logger.info(f"Limiting to first page images only for {site_url}")
            
            return {
                'site': site_url,
                'page_url': site_url,
                'html': html,
                'images': images,
                'pages_crawled': pages_crawled
            }
        
    except Exception as e:
        logger.error(f"Error crawling {site_url}: {e}")
        return None


def crawling_worker(worker_id: int, redis_url: str, use_3x3_mining: bool = False, max_pages: int = None, site_results_list = None):
    """
    Crawling worker process main loop.
    
    Args:
        worker_id: Worker ID
        redis_url: Redis connection URL
        use_3x3_mining: Whether to enable 3x3 mining
        max_pages: Maximum pages to crawl per site (None = unlimited)
        site_results_list: Shared list of SiteResult objects to update
    """
    logger.info(f"[Crawler {worker_id}] Starting process")
    
    # Get Redis client
    redis_client = get_redis_client(redis_url)
    
    logger.info(f"[Crawler {worker_id}] Starting crawling worker loop")
    
    while True:
        try:
            # Get site from queue
            site = get_site_from_queue(redis_client, timeout=5)
            
            if site is None:
                # No sites in queue, continue
                continue
            
            logger.info(f"[Crawler {worker_id}] Processing site: {site}")
            
            # Crawl the site with page limit
            result = asyncio.run(crawl_site(site, use_3x3_mining, max_pages))
            
            if result:
                # Update site results
                if site_results_list:
                    for site_result in site_results_list:
                        if site_result.url == site:
                            site_result.images_found = len(result['images'])
                            site_result.pages_crawled = result.get('pages_crawled', 1)
                            site_result.processing_time = 0.0  # Will be updated by extraction worker
                            break
                
                # Push to crawled_pages queue
                push_crawled_page(redis_client, result)
                logger.info(f"[Crawler {worker_id}] Crawled site: {site} ({len(result['images'])} images)")
            else:
                logger.warning(f"[Crawler {worker_id}] Failed to crawl: {site}")
                
        except KeyboardInterrupt:
            logger.info(f"[Crawler {worker_id}] Interrupted, shutting down")
            break
        except Exception as e:
            logger.error(f"[Crawler {worker_id}] Error: {e}", exc_info=True)
            import time
            time.sleep(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker-id', type=int, required=True)
    parser.add_argument('--redis-url', type=str, required=True)
    parser.add_argument('--use-3x3-mining', action='store_true')
    
    args = parser.parse_args()
    crawling_worker(args.worker_id, args.redis_url, args.use_3x3_mining)
