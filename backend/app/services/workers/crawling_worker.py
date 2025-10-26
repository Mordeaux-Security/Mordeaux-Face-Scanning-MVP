"""
Crawling Worker Process

Worker process responsible for fetching HTML, parsing, and selector mining.
Pushes crawled page data to Redis queue for extraction workers.
"""

import asyncio
import logging
import sys
import os
from urllib.parse import urljoin, urlparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.services.redis_queues import get_redis_client, get_site_from_queue, push_crawled_page
from app.services.http_service import fetch_html_with_redirects, fetch_html_with_js_rendering
from app.services.selector_mining import SelectorMiningService
from bs4 import BeautifulSoup

# Configure logging for multiprocessing
logging.basicConfig(
    level=logging.INFO,
    format=f'%(asctime)s - Crawler-%(process)d - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)


def discover_page_urls(soup: BeautifulSoup, base_url: str, max_urls: int = 20) -> list:
    """
    Discover additional page URLs to crawl from the current page.
    Looks for pagination links and internal links.
    
    Args:
        soup: BeautifulSoup parsed HTML
        base_url: Base URL for resolving relative links
        max_urls: Maximum URLs to return
        
    Returns:
        List of discovered URLs
    """
    discovered_urls = []
    parsed_base = urlparse(base_url)
    
    # Find all links
    for link in soup.find_all('a', href=True):
        href = link.get('href', '').strip()
        if not href or href.startswith('#'):
            continue
        
        # Convert to absolute URL
        absolute_url = urljoin(base_url, href)
        parsed_url = urlparse(absolute_url)
        
        # Only include same-domain URLs with http/https
        if (parsed_url.scheme in ('http', 'https') and 
            parsed_url.netloc == parsed_base.netloc and
            absolute_url not in discovered_urls):
            
            # Prioritize pagination and gallery links
            link_text = link.get_text().lower()
            link_classes = ' '.join(link.get('class', [])).lower()
            
            is_pagination = any(pattern in link_text or pattern in link_classes 
                              for pattern in ['next', 'page', 'more', 'gallery', 'album', 'category'])
            
            if is_pagination or len(discovered_urls) < max_urls // 2:
                discovered_urls.append(absolute_url)
                
                if len(discovered_urls) >= max_urls:
                    break
    
    return discovered_urls[:max_urls]




async def crawl_site(site_url: str, use_3x3_mining: bool = False, max_pages: int = None) -> dict:
    """
    Crawl multiple pages on a site.
    
    Args:
        site_url: Starting URL to crawl
        use_3x3_mining: Whether to enable 3x3 mining
        max_pages: Maximum pages to crawl (None = unlimited, default 1)
        
    Returns:
        Dict with site, pages crawled, and all images found
    """
    if max_pages is None:
        max_pages = 1
    
    try:
        # Create HTTP client
        import httpx
        timeout = httpx.Timeout(30.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            
            # Initialize crawling state
            visited_urls = set()
            urls_to_visit = [site_url]
            all_images = []
            pages_crawled = 0
            
            while urls_to_visit and pages_crawled < max_pages:
                current_url = urls_to_visit.pop(0)
                
                # Skip if already visited
                if current_url in visited_urls:
                    continue
                
                visited_urls.add(current_url)
                pages_crawled += 1
                
                logger.info(f"Crawling page {pages_crawled}/{max_pages}: {current_url}")
                
                # Fetch HTML
                html, errors = await fetch_html_with_redirects(current_url, client)
                
                # If HTML fetch failed or returned very little content, try JS rendering
                if not html or len(html) < 500:
                    logger.info(f"Standard fetch failed or returned minimal content, trying JS rendering for {current_url}")
                    html, js_errors = await fetch_html_with_js_rendering(
                        current_url, 
                        timeout=10.0,
                        wait_for_network_idle=True
                    )
                    if html:
                        logger.info(f"JS rendering successful for {current_url}")
                    else:
                        logger.error(f"JS rendering also failed for {current_url}: {js_errors}")
                        continue
                
                if not html:
                    logger.error(f"Failed to fetch {current_url}: {errors}")
                    continue
                
                # Parse HTML
                soup = BeautifulSoup(html, 'html.parser')
                
                # Extract images from this page
                page_images = []
                if use_3x3_mining:
                    # Use 3x3 mining for better selector discovery
                    try:
                        mining_service = SelectorMiningService()
                        mined_result = await mining_service.mine_selectors_for_page(
                            html=html,
                            url=current_url,
                            client=client
                        )
                        
                        if mined_result.candidates:
                            for candidate in mined_result.candidates:
                                for img_url in candidate.images:
                                    page_images.append({
                                        'url': img_url,
                                        'alt': '',
                                        'width': None,
                                        'height': None
                                    })
                        
                        logger.info(f"3x3 mining found {len(page_images)} images from {len(mined_result.candidates)} candidates")
                        
                        # Fallback to simple extraction if mining found nothing
                        if len(page_images) == 0:
                            logger.warning("3x3 mining found no images, falling back to simple extraction")
                            for img in soup.find_all('img', src=True):
                                src = img.get('src', '').strip()
                                if src:
                                    absolute_url = urljoin(current_url, src)
                                    if absolute_url.startswith(('http://', 'https://')):
                                        page_images.append({
                                            'url': absolute_url,
                                            'alt': img.get('alt', ''),
                                            'width': img.get('width'),
                                            'height': img.get('height')
                                        })
                            logger.info(f"Simple extraction found {len(page_images)} images")
                    
                    except Exception as e:
                        logger.error(f"Error during 3x3 mining for {current_url}: {e}")
                        # Fallback to simple extraction
                        for img in soup.find_all('img', src=True):
                            src = img.get('src', '').strip()
                            if src:
                                absolute_url = urljoin(current_url, src)
                                if absolute_url.startswith(('http://', 'https://')):
                                    page_images.append({
                                        'url': absolute_url,
                                        'alt': img.get('alt', ''),
                                        'width': img.get('width'),
                                        'height': img.get('height')
                                    })
                else:
                    # Simple extraction
                    for img in soup.find_all('img', src=True):
                        src = img.get('src', '').strip()
                        if src:
                            absolute_url = urljoin(current_url, src)
                            if absolute_url.startswith(('http://', 'https://')):
                                page_images.append({
                                    'url': absolute_url,
                                    'alt': img.get('alt', ''),
                                    'width': img.get('width'),
                                    'height': img.get('height')
                                })
                
                all_images.extend(page_images)
                logger.info(f"Page {pages_crawled} yielded {len(page_images)} images (total: {len(all_images)})")
                
                # Discover more URLs if we haven't reached the page limit
                if pages_crawled < max_pages:
                    discovered = discover_page_urls(soup, current_url, max_urls=10)
                    for url in discovered:
                        if url not in visited_urls and url not in urls_to_visit:
                            urls_to_visit.append(url)
                    logger.info(f"Discovered {len(discovered)} new URLs to visit")
            
            logger.info(f"Crawl complete: {pages_crawled} pages, {len(all_images)} total images")
            
            return {
                'site': site_url,
                'page_url': site_url,
                'html': html if pages_crawled > 0 else '',
                'images': all_images,
                'pages_crawled': pages_crawled
            }
        
    except Exception as e:
        error_msg = str(e)
        if '403' in error_msg:
            logger.warning(f"Site {site_url} returned 403 Forbidden - may require authentication or be blocking crawlers")
        else:
            logger.error(f"Error crawling {site_url}: {e}")
        return None


def crawling_worker(worker_id: int, redis_url: str, use_3x3_mining: bool = False, max_pages: int = None, site_results_dict = None):
    """
    Crawling worker process main loop.
    
    Args:
        worker_id: Worker ID
        redis_url: Redis connection URL
        use_3x3_mining: Whether to enable 3x3 mining
        max_pages: Maximum pages to crawl per site (None = unlimited)
        site_results_dict: Shared dict of SiteResult objects to update
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
                logger.info(f"[DATAFLOW] Crawler {worker_id} → Redis: Site={site}, Pages={result.get('pages_crawled', 0)}, Images={len(result.get('images', []))}")
                
                # Update site results
                if site_results_dict and site in site_results_dict:
                    stats = site_results_dict[site]
                    stats['images_found'] = len(result['images'])
                    stats['pages_crawled'] = result.get('pages_crawled', 1)
                    stats['processing_time'] = 0.0  # Will be updated by extraction worker
                    site_results_dict[site] = stats  # Explicit reassignment for sync
                
                # Push to crawled_pages queue
                push_crawled_page(redis_client, result)
                logger.info(f"[Crawler {worker_id}] Crawled site: {site} ({len(result['images'])} images)")
                
                logger.info(f"[DATAFLOW] Crawler {worker_id} → Stats Updated: Site={site}, ImagesFound={stats['images_found']}, PagesCrawled={stats['pages_crawled']}")
            else:
                # Track failure
                if site_results_dict and site in site_results_dict:
                    stats = site_results_dict[site]
                    stats['errors'].append(f"Failed to crawl: site returned error or no content")
                    site_results_dict[site] = stats
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
