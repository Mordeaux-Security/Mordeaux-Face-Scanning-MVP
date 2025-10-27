"""
Selector Miner for New Crawler System

Cleaned up 3x3 selector mining logic with core patterns only.
Performs 3x3 crawl (3 category pages × 3 content pages) for better structure diversity.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import httpx

from .config import get_config
from .http_utils import get_http_utils
from .data_structures import CandidateImage

logger = logging.getLogger(__name__)

# Singleton pattern per process
_selector_miner_instance = None


class SelectorMiner:
    """Selector miner with 3x3 crawling approach."""
    
    def __init__(self):
        self.config = get_config()
        self.http_utils = get_http_utils()
        self._semaphore = asyncio.Semaphore(3)  # Max 3 concurrent page fetches
        
        # Core selector patterns (cleaned up from bloated original)
        self.core_patterns = [
            # Video/media patterns
            '.video img', '.video-item img', '.video-thumb img', '.video-card img',
            '.media img', '.media-item img', '.media-thumb img', '.media-card img',
            
            # Gallery patterns
            '.gallery img', '.gallery-item img', '.gallery-thumb img', '.gallery-card img',
            '.grid-item img', '.masonry-item img', '.photo-item img', '.photo-card img',
            
            # Thumbnail patterns
            '.thumb img', '.thumbnail img', '.thumb-block img', '.thumb-wrapper img',
            '.thumb-inside img', '.thumb-container img', '.thumb-holder img',
            
            # Generic container patterns
            '.item img', '.card img', '.post img', '.entry img', '.content-item img',
            '.list-item img', '.feed-item img', '.tile img', '.cell img', '.box img',
            
            # Framework-specific patterns
            '[class*="thumb"] img', '[class*="grid"] img', '[class*="card"] img',
            '[class*="video"] img', '[class*="media"] img', '[class*="gallery"] img',
            
            # List structures
            'li img', 'li.item img', 'li.card img', 'li.thumb img',
            
            # Legacy patterns (preserve existing)
            '.list-global__item img', '.post-item img', '.video-thumb img',
            '.media-thumb img', '.content-thumb img'
        ]
    
    async def mine_selectors(self, html: str, base_url: str, site_id: str) -> List[CandidateImage]:
        """Mine selectors from HTML content."""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            candidates = []
            
            # Find images using core patterns
            for pattern in self.core_patterns[:self.config.nc_max_selector_patterns]:
                try:
                    images = soup.select(pattern)
                    if len(images) >= 2:  # Only consider patterns with 2+ matches
                        for img in images:
                            candidate = self._create_candidate(img, base_url, pattern, site_id)
                            if candidate:
                                candidates.append(candidate)
                except Exception as e:
                    logger.debug(f"Error with pattern {pattern}: {e}")
                    continue
            
            # Remove duplicates based on image URL
            seen_urls = set()
            unique_candidates = []
            for candidate in candidates:
                if candidate.img_url not in seen_urls:
                    seen_urls.add(candidate.img_url)
                    unique_candidates.append(candidate)
            
            logger.info(f"Found {len(unique_candidates)} unique image candidates from {base_url}")
            return unique_candidates
            
        except Exception as e:
            logger.error(f"Error mining selectors from {base_url}: {e}")
            return []
    
    def _create_candidate(self, img_element, base_url: str, selector: str, site_id: str) -> Optional[CandidateImage]:
        """Create candidate image from img element."""
        try:
            # Get image source
            src = img_element.get('src') or img_element.get('data-src') or img_element.get('data-lazy-src')
            if not src:
                return None
            
            # Convert to absolute URL
            img_url = urljoin(base_url, src)
            
            # Validate URL
            parsed = urlparse(img_url)
            if not parsed.scheme or not parsed.netloc:
                return None
            
            # Get additional attributes
            alt_text = img_element.get('alt', '')
            width = img_element.get('width') or img_element.get('Width')
            height = img_element.get('height') or img_element.get('Height')
            
            # Convert width/height to integers if possible
            try:
                width = int(width) if width else None
                height = int(height) if height else None
            except (ValueError, TypeError):
                width = height = None
            
            # Infer content type from URL extension
            content_type = None
            img_url_lower = img_url.lower()
            if img_url_lower.endswith(('.jpg', '.jpeg')):
                content_type = 'image/jpeg'
            elif img_url_lower.endswith('.png'):
                content_type = 'image/png'
            elif img_url_lower.endswith('.webp'):
                content_type = 'image/webp'
            elif img_url_lower.endswith('.gif'):
                content_type = 'image/gif'
            elif img_url_lower.endswith('.bmp'):
                content_type = 'image/bmp'
            elif img_url_lower.endswith('.svg'):
                content_type = 'image/svg+xml'
            
            # Estimate file size from dimensions (rough approximation)
            estimated_size = None
            if width and height:
                # Rough estimate: width * height * 3 bytes (RGB) * compression factor
                estimated_size = int(width * height * 3 * 0.3)  # 30% compression factor
            
            # Check for srcset
            has_srcset = bool(img_element.get('srcset'))
            
            return CandidateImage(
                page_url=base_url,
                img_url=img_url,
                selector_hint=selector,
                site_id=site_id,
                alt_text=alt_text,
                width=width,
                height=height,
                content_type=content_type,
                estimated_size=estimated_size,
                has_srcset=has_srcset
            )
            
        except Exception as e:
            logger.debug(f"Error creating candidate: {e}")
            return None
    
    async def mine_with_3x3_crawl(self, base_url: str, site_id: str, max_pages: int = 5) -> List[CandidateImage]:
        """Perform 3x3 crawl: 3 category pages × 3 content pages."""
        try:
            # Import redis manager at function level to avoid circular imports
            from .redis_manager import get_redis_manager
            redis = get_redis_manager()
            
            logger.info(f"Starting 3x3 crawl for {base_url} (max_pages={max_pages})")
            all_candidates = []
            checked_urls = {base_url}
            pages_crawled = 0
            
            # Step 1: Fetch base page
            logger.debug(f"[3x3-CRAWL] Fetching page {pages_crawled + 1}/{max_pages}: {base_url}")
            async with self._semaphore:
                html, error = await self.http_utils.fetch_html(base_url)
            if not html:
                logger.warning(f"Failed to fetch base page {base_url}: {error}")
                return []
            
            pages_crawled += 1
            
            # Mine from base page
            base_candidates = await self.mine_selectors(html, base_url, site_id)
            all_candidates.extend(base_candidates)
            logger.debug(f"[3x3-CRAWL] Page yielded {len(base_candidates)} candidates")
            
            # Check if we've reached the page limit
            if pages_crawled >= max_pages:
                logger.info(f"[3x3-CRAWL] Reached max pages limit ({max_pages}), stopping")
                return all_candidates
            
            # NEW: Check if thumbnail limit reached before continuing
            site_stats = await asyncio.to_thread(redis.get_site_stats, site_id)
            thumbnails_saved = site_stats.get('images_saved_thumbs', 0) if site_stats else 0
            
            if thumbnails_saved >= self.config.nc_max_images_per_site:
                logger.info(f"[3x3-CRAWL] Site {site_id} has {thumbnails_saved} thumbnails (limit: {self.config.nc_max_images_per_site}), stopping crawl")
                return all_candidates
            
            # Step 2: Find 3 category pages
            soup = BeautifulSoup(html, 'html.parser')
            category_urls = await self._discover_category_pages(soup, base_url)
            logger.info(f"Found {len(category_urls)} category pages")
            
            # Process up to 3 category pages
            for i, category_url in enumerate(category_urls[:3]):
                # Check thumbnail limit before fetching next page
                site_stats = await asyncio.to_thread(redis.get_site_stats, site_id)
                thumbnails_saved = site_stats.get('images_saved_thumbs', 0) if site_stats else 0
                
                if thumbnails_saved >= self.config.nc_max_images_per_site:
                    logger.info(f"[3x3-CRAWL] Site {site_id} has {thumbnails_saved} thumbnails (limit: {self.config.nc_max_images_per_site}), stopping crawl")
                    return all_candidates
                
                if category_url in checked_urls:
                    continue
                
                if pages_crawled >= max_pages:
                    logger.info(f"[3x3-CRAWL] Reached max pages limit ({max_pages}), stopping")
                    break
                
                logger.debug(f"[3x3-CRAWL] Fetching page {pages_crawled + 1}/{max_pages}: {category_url}")
                checked_urls.add(category_url)
                
                try:
                    async with self._semaphore:
                        cat_html, cat_error = await self.http_utils.fetch_html(category_url)
                    if cat_html:
                        pages_crawled += 1
                        cat_candidates = await self.mine_selectors(cat_html, category_url, site_id)
                        all_candidates.extend(cat_candidates)
                        logger.debug(f"[3x3-CRAWL] Page yielded {len(cat_candidates)} candidates")
                        
                        # Step 3: Find 3 content pages from this category
                        cat_soup = BeautifulSoup(cat_html, 'html.parser')
                        content_urls = await self._discover_content_pages(cat_soup, category_url)
                        
                        # Process up to 3 content pages from this category
                        for j, content_url in enumerate(content_urls[:3]):
                            # Check thumbnail limit before fetching next page
                            site_stats = await asyncio.to_thread(redis.get_site_stats, site_id)
                            thumbnails_saved = site_stats.get('images_saved_thumbs', 0) if site_stats else 0
                            
                            if thumbnails_saved >= self.config.nc_max_images_per_site:
                                logger.info(f"[3x3-CRAWL] Site {site_id} has {thumbnails_saved} thumbnails (limit: {self.config.nc_max_images_per_site}), stopping crawl")
                                return all_candidates
                            
                            if content_url in checked_urls:
                                continue
                            
                            if pages_crawled >= max_pages:
                                logger.info(f"[3x3-CRAWL] Reached max pages limit ({max_pages}), stopping")
                                break
                            
                            logger.debug(f"[3x3-CRAWL] Fetching page {pages_crawled + 1}/{max_pages}: {content_url}")
                            checked_urls.add(content_url)
                            
                            try:
                                async with self._semaphore:
                                    content_html, content_error = await self.http_utils.fetch_html(content_url)
                                if content_html:
                                    pages_crawled += 1
                                    content_candidates = await self.mine_selectors(content_html, content_url, site_id)
                                    all_candidates.extend(content_candidates)
                                    logger.debug(f"[3x3-CRAWL] Page yielded {len(content_candidates)} candidates")
                            except Exception as e:
                                logger.debug(f"Error processing content page {content_url}: {e}")
                                continue
                    else:
                        logger.warning(f"Failed to fetch category page {category_url}: {cat_error}")
                except Exception as e:
                    logger.debug(f"Error processing category page {category_url}: {e}")
                    continue
            
            # Remove duplicates
            seen_urls = set()
            unique_candidates = []
            for candidate in all_candidates:
                if candidate.img_url not in seen_urls:
                    seen_urls.add(candidate.img_url)
                    unique_candidates.append(candidate)
            
            logger.info(f"3x3 crawl completed: {pages_crawled} pages crawled, {len(checked_urls)} URLs checked, {len(unique_candidates)} unique candidates found")
            return unique_candidates
            
        except Exception as e:
            logger.error(f"Error in 3x3 crawl for {base_url}: {e}")
            return []
    
    async def _discover_category_pages(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Discover category/listing pages."""
        try:
            same_host = urlparse(base_url).netloc
            category_urls = []
            
            # Look for common category link patterns
            category_selectors = [
                'a[href*="/category"]', 'a[href*="/categories"]', 'a[href*="/gallery"]',
                'a[href*="/galleries"]', 'a[href*="/videos"]', 'a[href*="/video"]',
                'a[href*="/new"]', 'a[href*="/latest"]', 'a[href*="/trending"]',
                'a[href*="/hot"]', 'a[href*="/popular"]', 'a[href*="/top"]',
                '.category a', '.categories a', '.gallery a', '.galleries a',
                '.nav a', '.menu a', '.sidebar a'
            ]
            
            for selector in category_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(base_url, href)
                        parsed_url = urlparse(absolute_url)
                        
                        # Only include same-host URLs
                        if parsed_url.netloc == same_host:
                            # Filter out obvious non-category pages
                            if not any(skip in parsed_url.path.lower() for skip in ['/user', '/profile', '/login', '/register', '/admin']):
                                category_urls.append(absolute_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in category_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            logger.info(f"Discovered {len(unique_urls)} category pages")
            return unique_urls[:10]  # Return top 10 for selection
            
        except Exception as e:
            logger.error(f"Error discovering category pages: {e}")
            return []
    
    async def _discover_content_pages(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Discover content pages from a category page."""
        try:
            same_host = urlparse(base_url).netloc
            content_urls = []
            
            # Look for common content link patterns
            content_selectors = [
                'a[href*="/t/"]', 'a[href*="/thread"]', 'a[href*="/post"]',
                'a[href*="/video"]', 'a[href*="/watch"]', 'a[href*="/view"]',
                'a[href*="/item"]', 'a[href*="/article"]', 'a[href*="/story"]',
                '.topic-title a', '.thread-title a', '.post-title a',
                '.item-title a', '.content-title a', '.title a',
                '.video-title a', '.media-title a', '.gallery-title a'
            ]
            
            for selector in content_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(base_url, href)
                        parsed_url = urlparse(absolute_url)
                        
                        # Only include same-host URLs
                        if parsed_url.netloc == same_host:
                            # Filter out obvious non-content pages
                            if not any(skip in parsed_url.path.lower() for skip in ['/category', '/admin', '/user', '/profile', '/settings']):
                                content_urls.append(absolute_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in content_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            logger.info(f"Discovered {len(unique_urls)} content pages")
            return unique_urls[:10]  # Return top 10 for selection
            
        except Exception as e:
            logger.error(f"Error discovering content pages: {e}")
            return []
    
    async def mine_site(self, site_url: str, site_id: str) -> List[CandidateImage]:
        """Mine selectors from a site using 3x3 approach if enabled."""
        try:
            if self.config.nc_use_3x3_mining:
                logger.info(f"Using 3x3 mining for {site_url}")
                return await self.mine_with_3x3_crawl(site_url, site_id)
            else:
                logger.info(f"Using simple mining for {site_url}")
                html, error = await self.http_utils.fetch_html(site_url)
                if html:
                    return await self.mine_selectors(html, site_url, site_id)
                else:
                    logger.warning(f"Failed to fetch {site_url}: {error}")
                    return []
        except Exception as e:
            logger.error(f"Error mining site {site_url}: {e}")
            return []




def get_selector_miner() -> SelectorMiner:
    """Get singleton selector miner instance."""
    global _selector_miner_instance
    if _selector_miner_instance is None:
        _selector_miner_instance = SelectorMiner()
    return _selector_miner_instance


def close_selector_miner():
    """Close singleton selector miner."""
    global _selector_miner_instance
    if _selector_miner_instance:
        _selector_miner_instance = None
