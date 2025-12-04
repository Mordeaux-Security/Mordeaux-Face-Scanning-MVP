#!/usr/bin/env python3
"""
Simple test script to verify HTML crawling and saving to MinIO.
Fetches HTML from URLs and saves raw HTML to MinIO raw-images bucket.
"""

import asyncio
import hashlib
import logging
import sys
import re
from pathlib import Path
from typing import Optional, Tuple

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from new_crawler.config import get_config
from new_crawler.http_utils import HTTPUtils
from new_crawler.storage_manager import StorageManager
from new_crawler.selector_miner import SelectorMiner
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_post_sections(html: str, url: str) -> str:
    """Extract only the HTML sections that contain posts/discussions."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find post container patterns (similar to selector miner)
        post_containers = []
        
        # Strategy 1: Table-based forum listings (vBulletin-style)
        threadbits = soup.select('tbody[id*="threadbits"]')
        if threadbits:
            for tbody in threadbits:
                post_containers.append(tbody)
        
        # Strategy 2: Activity stream patterns (Mayo Clinic style)
        activity_rows = soup.select('.ch-activity-simple-row, .activity-simple-row')
        if activity_rows:
            # Get parent container
            if activity_rows:
                parent = activity_rows[0].find_parent('div', class_=re.compile(r'activity.*stream|stream.*activity', re.I))
                if parent:
                    post_containers.append(parent)
                else:
                    # Just use the rows themselves
                    container = soup.new_tag('div', attrs={'class': 'extracted-posts'})
                    for row in activity_rows:
                        container.append(row.extract())
                    post_containers.append(container)
        
        # Strategy 3: Generic repeated post containers
        # Find divs with classes containing discussion/thread/post keywords that appear multiple times
        discussion_divs = soup.find_all('div', class_=re.compile(r'discussion|thread|post|topic|message|activity.*row', re.I))
        if discussion_divs:
            # Group by class to find repeated patterns
            class_groups = {}
            for div in discussion_divs:
                classes = ' '.join(div.get('class', []))
                if classes:
                    if classes not in class_groups:
                        class_groups[classes] = []
                    class_groups[classes].append(div)
            
            # Take classes that appear 3+ times (likely post containers)
            for classes, divs in class_groups.items():
                if len(divs) >= 3:
                    # Get parent container or create one
                    if divs:
                        parent = divs[0].find_parent(['div', 'section', 'main', 'article'])
                        if parent and parent not in post_containers:
                            post_containers.append(parent)
                    break
        
        # Strategy 4: Posts container (vBulletin-style detail pages)
        posts_div = soup.select_one('div#posts, div[id*="posts"]')
        if posts_div:
            post_containers.append(posts_div)
        
        # Strategy 5: Thread post containers
        threadposts = soup.select('div.threadpost, div[id*="edit"][id*="post"]')
        if threadposts:
            # Get parent container
            if threadposts:
                parent = threadposts[0].find_parent(['div', 'section'])
                if parent:
                    post_containers.append(parent)
        
        # If we found containers, extract them
        if post_containers:
            # Create a new document with just the post sections
            extracted = BeautifulSoup('<!DOCTYPE html><html><head><title>Extracted Posts</title></head><body></body></html>', 'html.parser')
            body = extracted.body
            
            for container in post_containers:
                # Clone the container to avoid modifying original
                cloned = BeautifulSoup(str(container), 'html.parser')
                body.append(cloned)
            
            return str(extracted)
        else:
            # No post containers found, return original HTML
            logger.warning(f"No post containers found in {url}, returning full HTML")
            return html
            
    except Exception as e:
        logger.error(f"Error extracting post sections from {url}: {e}")
        # Return original HTML on error
        return html


async def save_html_to_minio(storage: StorageManager, html: str, url: str) -> Tuple[Optional[str], Optional[str]]:
    """Save HTML content to BOTH MinIO raw-images bucket AND local file."""
    html_bytes = html.encode('utf-8')
    content_hash = hashlib.sha256(html_bytes).hexdigest()
    
    # Generate key: html/{first2}/{hash}.html
    key = f"html/{content_hash[:2]}/{content_hash}.html"
    
    config = storage.config
    minio_success = False
    local_success = False
    minio_key = None
    local_path = None
    
    # Save to MinIO (CRITICAL - must succeed)
    if config.s3_endpoint:
        try:
            client = storage._get_client()
            
            # Ensure bucket exists
            try:
                from minio.error import S3Error
                # Check if bucket exists
                found = client.bucket_exists(config.s3_bucket_raw)
                if not found:
                    logger.info(f"Creating bucket: {config.s3_bucket_raw}")
                    client.make_bucket(config.s3_bucket_raw)
                    logger.info(f"Bucket {config.s3_bucket_raw} created")
            except S3Error as e:
                if e.code != 'NoSuchBucket':
                    raise
                logger.info(f"Creating bucket: {config.s3_bucket_raw}")
                client.make_bucket(config.s3_bucket_raw)
            except Exception as e:
                logger.warning(f"Bucket check/create failed (may already exist): {e}")
            
            import io
            # MinIO
            client.put_object(
                bucket_name=config.s3_bucket_raw,
                object_name=key,
                data=io.BytesIO(html_bytes),
                length=len(html_bytes),
                content_type='text/html'
            )
            minio_key = key
            minio_success = True
            url_str = f"{config.s3_endpoint}/{config.s3_bucket_raw}/{key}"
            logger.info(f"✓ Saved HTML to MinIO: {url} -> {key} ({len(html_bytes)} bytes)")
        except ImportError as e:
            logger.error(f"✗ MinIO library not available: {e}. Install with: pip install minio")
            raise  # Critical - fail if MinIO library missing
        except Exception as e:
            logger.error(f"✗ MinIO save FAILED for {url}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise  # Critical - fail if MinIO save fails
    else:
        logger.error("✗ MinIO endpoint not configured! Cannot save to MinIO.")
        raise ValueError("MinIO endpoint not configured")
    
    # ALWAYS save to local file as well (for verification)
    try:
        output_dir = Path(__file__).parent / "crawl_output" / "html"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create filename from URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        domain = parsed.netloc.replace('.', '_').replace(':', '_')
        filename = f"{domain}_{content_hash[:8]}.html"
        filepath = output_dir / filename
        
        with open(filepath, 'wb') as f:
            f.write(html_bytes)
        
        local_path = str(filepath)
        local_success = True
        logger.info(f"✓ Saved HTML to local file: {url} -> {filepath} ({len(html_bytes)} bytes)")
    except Exception as e:
        logger.error(f"✗ Local file save FAILED for {url}: {e}")
        # Don't raise - local is secondary, but log the error
    
    # Return MinIO key as primary, local path as secondary
    if minio_success:
        return minio_key, local_path if local_success else minio_key
    else:
        raise Exception("MinIO save failed - this is a critical error")


def discover_forum_links(html: str, base_url: str) -> list[str]:
    """Discover forum/board/discussion links from HTML."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        from urllib.parse import urljoin, urlparse
        
        discovered = set()
        base_domain = urlparse(base_url).netloc
        
        # Forum/board keywords to look for in links
        forum_keywords = ['forum', 'board', 'discussion', 'community', 'thread', 'post', 
                         'message', 'chat', 'discuss', 'topic']
        
        # Find links with forum keywords in href or text
        all_links = soup.find_all('a', href=True)
        for link in all_links:
            href = link.get('href', '')
            link_text = link.get_text(strip=True).lower()
            href_lower = href.lower()
            
            # Check if link contains forum keywords
            if any(keyword in href_lower or keyword in link_text for keyword in forum_keywords):
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                
                # Only same-domain links
                if parsed.netloc == base_domain or not parsed.netloc:
                    discovered.add(full_url)
        
        return list(discovered)[:20]  # Limit to 20 links
        
    except Exception as e:
        logger.error(f"Error discovering forum links: {e}")
        return []


def discover_pagination_links(html: str, base_url: str) -> list[str]:
    """Discover pagination links (next, page 2, etc.) from HTML."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        from urllib.parse import urljoin, urlparse
        
        discovered = set()
        base_domain = urlparse(base_url).netloc
        
        # Pagination keywords
        pagination_keywords = ['next', 'page', 'more', 'older', 'newer', '»', '›', '→']
        
        # Find pagination links
        pagination_selectors = [
            '.pagination a', '.pager a', '.page-nav a', '[class*="pagination"] a',
            '[class*="pager"] a', '.next', '.page-next', 'a[rel="next"]',
            '.chPagination a', '.chCorePaginate'
        ]
        
        for selector in pagination_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href', '')
                link_text = link.get_text(strip=True).lower()
                href_lower = href.lower()
                
                # Check if it's a pagination link
                if any(keyword in href_lower or keyword in link_text for keyword in pagination_keywords):
                    # Skip if it's just "#" or "javascript:"
                    if href and not href.startswith(('#', 'javascript:')):
                        full_url = urljoin(base_url, href)
                        parsed = urlparse(full_url)
                        
                        # Only same-domain links
                        if parsed.netloc == base_domain or not parsed.netloc:
                            discovered.add(full_url)
        
        # Also look for numbered page links (page 2, page 3, etc.)
        page_links = soup.find_all('a', href=True, string=re.compile(r'^\d+$|page\s*\d+|p\.?\s*\d+', re.I))
        for link in page_links:
            href = link.get('href', '')
            if href and not href.startswith(('#', 'javascript:')):
                full_url = urljoin(base_url, href)
                parsed = urlparse(full_url)
                if parsed.netloc == base_domain or not parsed.netloc:
                    discovered.add(full_url)
        
        return list(discovered)[:10]  # Limit to 10 pagination links
        
    except Exception as e:
        logger.error(f"Error discovering pagination links: {e}")
        return []


async def crawl_site_multiple_pages(base_url: str, http_utils: HTTPUtils, storage: StorageManager, max_pages: int = 5) -> list[tuple[str, bool, str]]:
    """Crawl a site through multiple pages, discovering forum areas and following pagination."""
    from urllib.parse import urlparse
    from collections import deque
    
    results = []
    visited = set()
    to_visit = deque([base_url])
    pages_crawled = 0
    
    logger.info(f"Starting multi-page crawl for {base_url} (max {max_pages} pages)")
    
    while to_visit and pages_crawled < max_pages:
        current_url = to_visit.popleft()
        
        # Skip if already visited
        if current_url in visited:
            continue
        
        visited.add(current_url)
        pages_crawled += 1
        
        try:
            logger.info(f"[{pages_crawled}/{max_pages}] Fetching: {current_url}")
            
            # Fetch HTML
            html, status, _ = await http_utils.fetch_html(current_url)
            
            if not html:
                logger.warning(f"No HTML retrieved for {current_url}: {status}")
                results.append((current_url, False, f"No HTML: {status}"))
                continue
            
            logger.info(f"Retrieved {len(html)} chars from {current_url}")
            
            # Extract post sections
            extracted_html = extract_post_sections(html, current_url)
            reduction_pct = 100 * (1 - len(extracted_html) / len(html)) if html else 0
            logger.info(f"Extracted {len(extracted_html)} chars ({reduction_pct:.1f}% reduction)")
            
            # Save to MinIO and local
            try:
                key, local_path = await save_html_to_minio(storage, extracted_html, current_url)
                results.append((current_url, True, f"MinIO: {key}"))
                logger.info(f"✓ Saved page {pages_crawled}: {key}")
            except Exception as e:
                results.append((current_url, False, f"Save failed: {e}"))
                logger.error(f"✗ Failed to save {current_url}: {e}")
                continue
            
            # Discover new links to crawl
            if pages_crawled < max_pages:
                # First, try to find pagination links (if we're on a post listing page)
                pagination_links = discover_pagination_links(html, current_url)
                if pagination_links:
                    logger.info(f"Found {len(pagination_links)} pagination links")
                    for link in pagination_links[:3]:  # Limit to 3 pagination links
                        if link not in visited and link not in to_visit:
                            to_visit.append(link)
                            logger.info(f"  Added pagination link: {link}")
                
                # Also discover forum/board links (to find forum areas)
                if pages_crawled == 1:  # Only on first page
                    forum_links = discover_forum_links(html, current_url)
                    if forum_links:
                        logger.info(f"Found {len(forum_links)} forum/board links")
                        for link in forum_links[:5]:  # Limit to 5 forum links
                            if link not in visited and link not in to_visit:
                                to_visit.append(link)
                                logger.info(f"  Added forum link: {link}")
            
        except Exception as e:
            logger.error(f"Error crawling {current_url}: {e}", exc_info=True)
            results.append((current_url, False, str(e)))
    
    logger.info(f"Completed crawl: {pages_crawled} pages, {len(results)} results")
    return results


async def test_crawl_and_save(urls: list[str], max_pages_per_site: int = 5):
    """Test crawling URLs and saving HTML to MinIO, with multi-page crawling."""
    import os
    import importlib
    
    # Skip Redis validation for this test
    os.environ['SKIP_REDIS_VALIDATION'] = '1'
    
    # Set MinIO configuration if not already set
    if not os.environ.get('S3_ENDPOINT'):
        # Default MinIO endpoint (local or Docker)
        # Try localhost first (for local testing), fallback to Docker hostname
        os.environ['S3_ENDPOINT'] = 'http://localhost:9000'
    # Ensure endpoint has protocol
    s3_endpoint = os.environ.get('S3_ENDPOINT', '')
    if s3_endpoint and not s3_endpoint.startswith(('http://', 'https://')):
        os.environ['S3_ENDPOINT'] = f'http://{s3_endpoint}'
    
    if not os.environ.get('S3_ACCESS_KEY'):
        os.environ['S3_ACCESS_KEY'] = os.environ.get('MINIO_ROOT_USER', 'minioadmin')
    if not os.environ.get('S3_SECRET_KEY'):
        os.environ['S3_SECRET_KEY'] = os.environ.get('MINIO_ROOT_PASSWORD', 'minioadmin')
    if not os.environ.get('S3_USE_SSL'):
        os.environ['S3_USE_SSL'] = 'false'
    
    # Reload config to pick up environment variables
    import new_crawler.config
    importlib.reload(new_crawler.config)
    from new_crawler.config import get_config
    
    config = get_config()
    http_utils = HTTPUtils()
    storage = StorageManager()
    
    logger.info(f"Testing multi-page crawl for {len(urls)} sites (max {max_pages_per_site} pages per site)")
    logger.info(f"MinIO endpoint: {config.s3_endpoint}")
    logger.info(f"MinIO access key: {'***' if config.s3_access_key else 'None'}")
    logger.info(f"Raw bucket: {config.s3_bucket_raw}")
    
    # Test MinIO connection
    if config.s3_endpoint:
        try:
            client = storage._get_client()
            # Try to list buckets to verify connection
            buckets = client.list_buckets()
            bucket_names = [b.name for b in buckets] if hasattr(buckets, '__iter__') else []
            logger.info(f"MinIO connection successful. Buckets: {bucket_names}")
            
            # Ensure raw-images bucket exists
            if config.s3_bucket_raw not in bucket_names:
                logger.info(f"Creating bucket: {config.s3_bucket_raw}")
                client.make_bucket(config.s3_bucket_raw)
                logger.info(f"Bucket {config.s3_bucket_raw} created")
        except Exception as e:
            logger.warning(f"MinIO connection test failed: {e}. Will try to save anyway.")
    else:
        logger.warning("MinIO endpoint not configured. Will save to local files.")
    
    all_results = []
    
    for base_url in urls:
        try:
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting crawl for site: {base_url}")
            logger.info(f"{'='*60}")
            
            # Crawl multiple pages from this site
            site_results = await crawl_site_multiple_pages(base_url, http_utils, storage, max_pages_per_site)
            all_results.extend(site_results)
            
        except Exception as e:
            logger.error(f"Error processing site {base_url}: {e}", exc_info=True)
            all_results.append((base_url, False, str(e)))
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    successful = sum(1 for _, success, _ in all_results if success)
    logger.info(f"Successful: {successful}/{len(all_results)} pages saved")
    for url, success, info in all_results:
        status = "✓" if success else "✗"
        logger.info(f"{status} {url} -> {info}")
    
    # Cleanup
    await http_utils.close()
    
    return all_results


async def main():
    """Main entry point."""
    # Test URLs from sites.txt
    test_urls = [
        "https://forum.diabetes.org.uk",
        "https://www.diabetes.co.uk/forum",
        "https://www.healthboards.com/boards/index.php",
        "https://connect.mayoclinic.org",
        "https://www.reddit.com/r/diabetes/",
    ]
    
    # Or read from sites.txt if it exists
    sites_file = Path(__file__).parent / "sites.txt"
    if sites_file.exists():
        logger.info(f"Reading URLs from {sites_file}")
        with open(sites_file, 'r') as f:
            urls = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            if urls:
                test_urls = urls[:5]  # Limit to first 5 for testing
    
    await test_crawl_and_save(test_urls, max_pages_per_site=5)


if __name__ == "__main__":
    asyncio.run(main())

