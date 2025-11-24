"""
Selector Miner for New Crawler System

Cleaned up 3x3 selector mining logic with core patterns only.
Performs 3x3 crawl (3 category pages × 3 content pages) for better structure diversity.
"""

import asyncio
import logging
import time
import re
from typing import List, Dict, Any, Optional, Set, AsyncIterator, Tuple
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
        self._semaphore = asyncio.Semaphore(12)  # Max 12 concurrent page fetches
        
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
        """Mine selectors from HTML content with script/noscript/JSON-LD extraction."""
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

            # Broad fallback: if no candidates yet, scan all <img> tags on page
            if not candidates:
                try:
                    img_tags = soup.find_all('img')
                    for img in img_tags:
                        candidate = self._create_candidate(img, base_url, 'img', site_id)
                        if candidate:
                            candidates.append(candidate)
                    logger.info(f"Broad image fallback found {len(candidates)} candidates on {base_url}")
                except Exception as e:
                    logger.debug(f"Error during broad image fallback: {e}")
            
            # Extract images from noscript blocks if enabled
            if self.config.nc_extract_noscript_images:
                noscript_candidates = self._extract_noscript_images(soup, base_url, site_id)
                candidates.extend(noscript_candidates)
            
            # Extract images from JSON-LD if enabled
            if self.config.nc_extract_jsonld_images:
                jsonld_candidates = self._extract_jsonld_images(soup, base_url, site_id)
                candidates.extend(jsonld_candidates)
            
            # Extract images from script blocks if enabled
            if self.config.nc_extract_script_images:
                script_candidates = self._extract_script_images(soup, base_url, site_id)
                candidates.extend(script_candidates)
            
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
        """Create candidate image from img element with comprehensive attribute extraction."""
        try:
            # Comprehensive image source extraction
            img_url = self._extract_image_url(img_element, base_url)
            if not img_url:
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
            content_type = self._infer_content_type(img_url)
            
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
    
    def _extract_image_url(self, img_element, base_url: str) -> Optional[str]:
        """Extract image URL with comprehensive attribute checking."""
        # Primary attributes (in order of preference)
        primary_attrs = ['src', 'data-src', 'data-lazy-src', 'data-original', 'data-medium', 'data-large']
        
        # Additional lazy-load attributes
        lazy_attrs = ['data-lazy', 'data-highres', 'data-mediumthumb', 'data-thumb', 'data-image', 'data-tn']
        
        # Check primary attributes first
        for attr in primary_attrs:
            src = img_element.get(attr)
            if src and src.strip():
                img_url = urljoin(base_url, src.strip())
                if self._is_valid_image_url(img_url):
                    return img_url
        
        # Check lazy-load attributes if enabled
        if self.config.nc_extract_data_attributes:
            for attr in lazy_attrs:
                src = img_element.get(attr)
                if src and src.strip():
                    img_url = urljoin(base_url, src.strip())
                    if self._is_valid_image_url(img_url):
                        return img_url
        
        # Check srcset if enabled
        if self.config.nc_extract_srcset_images:
            srcset_url = self._extract_from_srcset(img_element, base_url)
            if srcset_url:
                return srcset_url
        
        # Check parent element data attributes
        if self.config.nc_extract_data_attributes:
            parent_url = self._extract_from_parent(img_element, base_url)
            if parent_url:
                return parent_url
        
        # Check background images in style attributes
        if self.config.nc_extract_background_images:
            bg_url = self._extract_background_image(img_element, base_url)
            if bg_url:
                return bg_url
        
        return None
    
    def _extract_from_srcset(self, img_element, base_url: str) -> Optional[str]:
        """Extract best quality image from srcset attribute."""
        srcset = img_element.get('srcset')
        if not srcset:
            return None
        
        try:
            # Parse srcset format: "url1 width1, url2 width2, ..."
            candidates = []
            for entry in srcset.split(','):
                entry = entry.strip()
                if not entry:
                    continue
                    
                parts = entry.split()
                if len(parts) < 2:
                    continue
                    
                url = parts[0].strip()
                descriptor = parts[1].strip()
                
                # Parse width descriptor (e.g., "640w")
                if descriptor.endswith('w'):
                    try:
                        width = int(descriptor[:-1])
                        absolute_url = urljoin(base_url, url)
                        if self._is_valid_image_url(absolute_url):
                            candidates.append((width, absolute_url))
                    except ValueError:
                        continue
            
            if not candidates:
                return None
            
            # Return the highest quality image (largest width)
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]
            
        except Exception as e:
            logger.debug(f"Error parsing srcset: {e}")
            return None
    
    def _extract_from_parent(self, img_element, base_url: str) -> Optional[str]:
        """Extract image URL from parent element data attributes."""
        try:
            parent = img_element.parent
            if not parent:
                return None
            
            # Check parent data attributes
            parent_attrs = ['data-src', 'data-image', 'data-thumb', 'data-medium', 'data-large']
            for attr in parent_attrs:
                src = parent.get(attr)
                if src and src.strip():
                    img_url = urljoin(base_url, src.strip())
                    if self._is_valid_image_url(img_url):
                        return img_url
            
            return None
        except Exception as e:
            logger.debug(f"Error extracting from parent: {e}")
            return None
    
    def _extract_background_image(self, img_element, base_url: str) -> Optional[str]:
        """Extract background image URL from style attribute."""
        style = img_element.get('style')
        if not style:
            return None
        
        try:
            # Look for background-image: url(...) patterns
            patterns = [
                r'background-image:\s*url\(["\']?([^"\']+)["\']?\)',
                r'background:\s*[^;]*url\(["\']?([^"\']+)["\']?\)',
                r'background-image:\s*url\(([^)]+)\)',
                r'background:\s*[^;]*url\(([^)]+)\)'
            ]
            
            for pattern in patterns:
                match = re.search(pattern, style, re.IGNORECASE)
                if match:
                    url = match.group(1).strip()
                    if url and not url.startswith('data:'):
                        img_url = urljoin(base_url, url)
                        if self._is_valid_image_url(img_url):
                            return img_url
            
            return None
        except Exception as e:
            logger.debug(f"Error extracting background image: {e}")
            return None
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL."""
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Skip data URLs and non-HTTP protocols
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Skip obvious non-image URLs
            url_lower = url.lower()
            skip_patterns = ['javascript:', 'mailto:', 'tel:', '#', '.pdf', '.doc', '.zip']
            if any(pattern in url_lower for pattern in skip_patterns):
                return False
            
            return True
        except Exception:
            return False
    
    def _infer_content_type(self, url: str) -> Optional[str]:
        """Infer content type from URL extension."""
        url_lower = url.lower()
        if url_lower.endswith(('.jpg', '.jpeg')):
            return 'image/jpeg'
        elif url_lower.endswith('.png'):
            return 'image/png'
        elif url_lower.endswith('.webp'):
            return 'image/webp'
        elif url_lower.endswith('.gif'):
            return 'image/gif'
        elif url_lower.endswith('.bmp'):
            return 'image/bmp'
        elif url_lower.endswith('.svg'):
            return 'image/svg+xml'
        return None
    
    async def mine_with_3x3_crawl(self, base_url: str, site_id: str, max_pages: int = 5) -> AsyncIterator[Tuple[str, List[CandidateImage]]]:
        """Perform 3x3 crawl: 1 base + 3 category + 3 content pages with strategy learning."""
        try:
            from .redis_manager import get_redis_manager
            from urllib.parse import urlparse
            redis = get_redis_manager()
            domain = urlparse(base_url).netloc
            
            logger.info(f"Starting 3x3 crawl for {base_url} (max_pages={max_pages})")
            checked_urls = {base_url}
            pages_crawled = 0
            
            # Track comparison stats for strategy learning
            total_http_candidates = 0
            total_js_candidates = 0
            sample_pages_count = 0
            max_sample_pages = 7 if max_pages == -1 else min(7, max_pages)  # 1+3+3 = 7 pages for sampling
            
            # PHASE 1: Sample crawl - fetch ALL sample pages with comparison
            
            # Step 1: Fetch base page with comparison
            logger.info(f"[3x3-SAMPLE] Fetching base page: {base_url}")
            async with self._semaphore:
                html, error, comparison_stats = await self.http_utils.fetch_html(
                    base_url, use_js_fallback=True, force_compare_first_visit=False
                )
            
            if not html:
                logger.warning(f"Failed to fetch base page {base_url}: {error}")
                return
            
            # Track stats
            if comparison_stats:
                total_http_candidates += comparison_stats['http_count']
                total_js_candidates += comparison_stats['js_count']
                sample_pages_count += 1
            
            pages_crawled += 1
            base_candidates = await self.mine_selectors(html, base_url, site_id)
            yield base_url, base_candidates
            
            # Get category/content URLs for sampling
            soup = BeautifulSoup(html, 'html.parser')
            category_urls = await self._discover_category_pages(soup, base_url)
            if not category_urls:
                category_urls = await self._discover_random_same_domain_links(soup, base_url, limit=3)
            
            # Step 2: Fetch up to 6 more sample pages in PARALLEL with comparison
            sample_tasks = []
            # Track URLs being processed to avoid duplicates, but don't mark as checked until success
            in_progress_urls = set()
            # Store URL-task pairs to track which URL corresponds to which task
            url_task_map = {}
            for category_url in category_urls[:6]:  # Take up to 6 more (for total of 7)
                if category_url not in checked_urls and category_url not in in_progress_urls and sample_pages_count < max_sample_pages:
                    in_progress_urls.add(category_url)  # Track in progress
                    task = asyncio.create_task(self._fetch_and_compare_page(category_url, site_id))
                    sample_tasks.append(task)
                    url_task_map[task] = category_url  # Map task to URL
            
            # Process sample pages as they complete
            for task in asyncio.as_completed(sample_tasks):
                original_url = url_task_map.get(task)  # Get original URL from task map
                url = original_url  # Initialize with original
                try:
                    candidates, returned_url, comparison_stats = await task
                    # Use returned_url in case it differs from mapped url
                    url = returned_url
                    pages_crawled += 1
                    sample_pages_count += 1
                    
                    if comparison_stats:
                        total_http_candidates += comparison_stats['http_count']
                        total_js_candidates += comparison_stats['js_count']
                    
                    # Mark as checked ONLY after successful processing
                    checked_urls.add(url)
                    # Clean up BOTH original and returned URLs from in_progress
                    in_progress_urls.discard(url)  # Remove returned URL
                    if url != original_url and original_url:
                        in_progress_urls.discard(original_url)  # Also remove original if different
                    
                    logger.info(f"[3x3-SAMPLE] Page {sample_pages_count}/{max_sample_pages} yielded {len(candidates)} candidates")
                    yield url, candidates
                    
                except Exception as e:
                    logger.debug(f"Error processing sample page: {e}")
                    # Don't mark failed URLs as checked - they can be retried in BFS phase
                    # Always clean up the original URL (it was the one we added to in_progress)
                    if original_url:
                        in_progress_urls.discard(original_url)
            
            # PHASE 2: Aggregate and store strategy
            logger.info(f"[3x3-SAMPLE] Completed {sample_pages_count} sample pages")
            logger.info(f"[3x3-AGGREGATE] Total HTTP candidates: {total_http_candidates}")
            logger.info(f"[3x3-AGGREGATE] Total JS candidates: {total_js_candidates}")
            
            # Determine winning strategy (aggressive HTTP-first if enabled)
            use_aggressive = getattr(self.config, 'nc_js_aggressive_http', True)
            if use_aggressive:
                # Aggressive: Require 3x more images or at least 10 for JS to win
                js_threshold = max(10, 3 * total_http_candidates)
                # Prefer HTTP if it found images (unless JS is significantly better)
                use_js = total_js_candidates >= js_threshold and (total_http_candidates == 0 or total_js_candidates >= js_threshold)
            else:
                # Original strategy: 2x more images or at least 5
                use_js = total_js_candidates >= max(5, 2 * total_http_candidates)
            
            await redis.set_domain_rendering_strategy_async(
                domain, use_js, total_http_candidates, total_js_candidates
            )
            logger.info(f"[3x3-STRATEGY] Stored strategy for {domain}: use_js={use_js} (HTTP={total_http_candidates}, JS={total_js_candidates})")
            
            # PHASE 3: BFS crawl using learned strategy
            if pages_crawled >= max_pages and max_pages != -1:
                return
            
            # Check thumbnail limit
            site_stats = await asyncio.to_thread(redis.get_site_stats, site_id)
            thumbnails_saved = site_stats.get('images_saved_thumbs', 0) if site_stats else 0
            
            if thumbnails_saved >= self.config.nc_max_images_per_site:
                logger.info(f"[3x3-BFS] Site {site_id} reached image limit: {thumbnails_saved}")
                return
            
            # BFS queue: start with remaining category URLs from initial discovery
            url_queue = [url for url in category_urls if url not in checked_urls]
            
            # If queue is empty or small, discover additional links from base page to seed BFS
            # This ensures we have URLs to process even if all category URLs were sampled
            if len(url_queue) < 5:  # Threshold: if less than 5 URLs, discover more
                try:
                    # Use the base page HTML we already fetched (from line 373)
                    # Parse it and discover all same-domain links
                    base_soup = BeautifulSoup(html, 'html.parser')
                    additional_urls = await self._discover_all_same_domain_links(base_soup, base_url)
                    
                    # Add new URLs that haven't been checked
                    for new_url in additional_urls:
                        if new_url not in checked_urls and new_url not in url_queue and new_url != base_url:
                            url_queue.append(new_url)
                    
                    if additional_urls:
                        discovered_count = len([u for u in additional_urls if u not in checked_urls and u != base_url])
                        if discovered_count > 0:
                            logger.debug(f"[3x3-BFS] Discovered {discovered_count} additional URLs from base page")
                except Exception as e:
                    logger.debug(f"Error discovering additional links from base page: {e}")
            
            logger.info(f"[3x3-BFS] Starting BFS with {len(url_queue)} initial URLs")
            
            while url_queue and (max_pages == -1 or pages_crawled < max_pages):
                # Check limits before each batch
                if thumbnails_saved >= self.config.nc_max_images_per_site:
                    logger.info(f"[3x3-BFS] Image limit reached, stopping")
                    break
                
                # Pop next URL
                current_url = url_queue.pop(0)
                if current_url in checked_urls:
                    continue
                
                checked_urls.add(current_url)
                
                # Fetch and mine page using learned strategy
                candidates, html = await self._fetch_and_mine_page_with_html(current_url, site_id)
                pages_crawled += 1
                yield current_url, candidates
                
                # Discover new links from this page
                if html and (max_pages == -1 or pages_crawled < max_pages):
                    soup = BeautifulSoup(html, 'html.parser')
                    new_urls = await self._discover_all_same_domain_links(soup, current_url)
                    
                    # Add undiscovered URLs to queue
                    for new_url in new_urls:
                        if new_url not in checked_urls and new_url not in url_queue:
                            url_queue.append(new_url)
                    
                    logger.debug(f"[3x3-BFS] Discovered {len(new_urls)} links, queue size: {len(url_queue)}")
                
                # Re-check stats
                site_stats = await asyncio.to_thread(redis.get_site_stats, site_id)
                thumbnails_saved = site_stats.get('images_saved_thumbs', 0) if site_stats else 0
            
            logger.info(f"[3x3-BFS] Completed: {pages_crawled} pages crawled, {len(checked_urls)} URLs checked")
            
        except Exception as e:
            logger.error(f"Error in 3x3 crawl for {base_url}: {e}")
            return

    async def _fetch_and_compare_page(self, url: str, site_id: str) -> Tuple[List[CandidateImage], str, Optional[Dict[str, int]]]:
        """Fetch and mine a page with HTTP vs JS comparison (for sample phase)."""
        try:
            async with self._semaphore:
                html, error, comparison_stats = await self.http_utils.fetch_html(
                    url, use_js_fallback=True, force_compare_first_visit=False
                )
            
            if html:
                candidates = await self.mine_selectors(html, url, site_id)
                return candidates, url, comparison_stats
            else:
                logger.warning(f"Failed to fetch page {url}: {error}")
                return [], url, None
        except Exception as e:
            logger.debug(f"Error fetching page {url}: {e}")
            return [], url, None

    async def _fetch_and_mine_page(self, url: str, site_id: str) -> Tuple[List[CandidateImage], str]:
        """Fetch and mine a page using learned strategy (for remaining pages after sample)."""
        try:
            from urllib.parse import urlparse
            from .redis_manager import get_redis_manager
            domain = urlparse(url).netloc
            redis = get_redis_manager()
            
            # Get stored strategy
            use_js = await redis.get_domain_rendering_strategy_async(domain)
            
            async with self._semaphore:
                if use_js is True:
                    # FORCE JS rendering directly
                    logger.info(f"[3x3-APPLY] Using JS rendering for {url} (domain strategy)")
                    html, error = await self.http_utils._fetch_with_js(url)
                elif use_js is False:
                    # Use HTTP only
                    logger.debug(f"[3x3-APPLY] Using HTTP for {url} (domain strategy)")
                    html, error = await self.http_utils._fetch_with_redirects(url)
                else:
                    # No strategy yet - shouldn't happen in remaining pages
                    logger.warning(f"[3x3-APPLY] No strategy for {domain}, using standard fetch")
                    html, error, _ = await self.http_utils.fetch_html(url)
            
            if html:
                candidates = await self.mine_selectors(html, url, site_id)
                return candidates, url
            else:
                logger.warning(f"Failed to fetch page {url}: {error}")
                return [], url
        except Exception as e:
            logger.error(f"Error fetching page {url}: {e}")
            return [], url
    
    async def _fetch_and_mine_page_with_html(self, url: str, site_id: str) -> Tuple[List[CandidateImage], Optional[str]]:
        """Fetch and mine a page, returning both candidates and HTML for link discovery."""
        try:
            from urllib.parse import urlparse
            from .redis_manager import get_redis_manager
            domain = urlparse(url).netloc
            redis = get_redis_manager()
            
            # Get stored strategy
            use_js = await redis.get_domain_rendering_strategy_async(domain)
            
            html = None
            async with self._semaphore:
                if use_js is True:
                    # FORCE JS rendering directly
                    html, error = await self.http_utils._fetch_with_js(url)
                elif use_js is False:
                    # Use HTTP only
                    html, error = await self.http_utils._fetch_with_redirects(url)
                else:
                    # No strategy yet - shouldn't happen in remaining pages
                    html, error, _ = await self.http_utils.fetch_html(url)
            
            if html:
                candidates = await self.mine_selectors(html, url, site_id)
                return candidates, html
            else:
                logger.warning(f"Failed to fetch page {url}: {error}")
                return [], None
        except Exception as e:
            logger.error(f"Error fetching page {url}: {e}")
            return [], None

    async def _discover_random_same_domain_links(self, soup: BeautifulSoup, base_url: str, limit: int = 3) -> List[str]:
        """Pick random same-domain links when categories are not detected."""
        try:
            import random as _random
            same_host = urlparse(base_url).netloc
            links = []
            for a in soup.find_all('a', href=True):
                absolute = urljoin(base_url, a['href'])
                parsed = urlparse(absolute)
                if parsed.netloc == same_host:
                    if not any(skip in parsed.path.lower() for skip in ['/login', '/register', '/user', '/profile', '/admin']):
                        links.append(absolute)
            # Unique and shuffle
            unique = []
            seen = set()
            for u in links:
                if u not in seen:
                    seen.add(u)
                    unique.append(u)
            _random.shuffle(unique)
            return unique[:limit]
        except Exception as e:
            logger.debug(f"Error discovering random links: {e}")
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
    
    async def _discover_all_same_domain_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Discover all same-domain links from a page (for BFS crawling)."""
        try:
            same_host = urlparse(base_url).netloc
            discovered_urls = []
            
            # Find all links
            for a in soup.find_all('a', href=True):
                href = a['href']
                absolute_url = urljoin(base_url, href)
                parsed_url = urlparse(absolute_url)
                
                # Same-domain only
                if parsed_url.netloc == same_host:
                    # Filter out auth/admin pages
                    if not any(skip in parsed_url.path.lower() for skip in 
                              ['/login', '/register', '/user', '/profile', '/admin', '/settings']):
                        discovered_urls.append(absolute_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in discovered_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            return unique_urls
        except Exception as e:
            logger.debug(f"Error discovering links: {e}")
            return []
    
    async def mine_site(self, site_url: str, site_id: str) -> List[CandidateImage]:
        """Mine selectors from a site using 3x3 approach if enabled."""
        try:
            if self.config.nc_use_3x3_mining:
                logger.info(f"Using 3x3 mining for {site_url}")
                all_candidates = []
                async for page_candidates in self.mine_with_3x3_crawl(site_url, site_id):
                    all_candidates.extend(page_candidates)
                return all_candidates
            else:
                logger.info(f"Using simple mining for {site_url}")
                html, error, _ = await self.http_utils.fetch_html(site_url)
                if html:
                    return await self.mine_selectors(html, site_url, site_id)
                else:
                    logger.warning(f"Failed to fetch {site_url}: {error}")
                    return []
        except Exception as e:
            logger.error(f"Error mining site {site_url}: {e}")
            return []

    def _extract_noscript_images(self, soup: BeautifulSoup, base_url: str, site_id: str) -> List[CandidateImage]:
        """Extract images from noscript blocks."""
        candidates = []
        try:
            noscript_tags = soup.find_all('noscript')
            for noscript in noscript_tags:
                # Parse the inner HTML of noscript as HTML
                inner_soup = BeautifulSoup(noscript.string or '', 'html.parser')
                img_tags = inner_soup.find_all('img')
                for img in img_tags:
                    candidate = self._create_candidate(img, base_url, 'noscript img', site_id)
                    if candidate:
                        candidates.append(candidate)
        except Exception as e:
            logger.debug(f"Error extracting noscript images: {e}")
        return candidates
    
    def _extract_jsonld_images(self, soup: BeautifulSoup, base_url: str, site_id: str) -> List[CandidateImage]:
        """Extract images from JSON-LD structured data."""
        candidates = []
        try:
            import json
            jsonld_scripts = soup.find_all('script', type='application/ld+json')
            for script in jsonld_scripts:
                try:
                    json_data = json.loads(script.string or '{}')
                    image_urls = self._extract_images_from_json(json_data)
                    for img_url in image_urls:
                        if self._is_valid_image_url(img_url):
                            full_url = urljoin(base_url, img_url)
                            candidate = CandidateImage(
                                page_url=base_url,
                                img_url=full_url,
                                selector_hint='json-ld',
                                site_id=site_id,
                                alt_text='',
                                width=None,
                                height=None,
                                content_type=self._infer_content_type(full_url),
                                estimated_size=None,
                                has_srcset=False
                            )
                            candidates.append(candidate)
                except json.JSONDecodeError:
                    continue
        except Exception as e:
            logger.debug(f"Error extracting JSON-LD images: {e}")
        return candidates
    
    def _extract_script_images(self, soup: BeautifulSoup, base_url: str, site_id: str) -> List[CandidateImage]:
        """Extract images from script blocks (HTML fragments and JSON)."""
        candidates = []
        try:
            script_tags = soup.find_all('script')
            img_tag_pattern = re.compile(r'<img[^>]+>', re.IGNORECASE)
            url_pattern = re.compile(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png|webp|gif)', re.IGNORECASE)
            
            for script in script_tags:
                script_content = script.string or ''
                
                # Extract HTML img tags from script content
                img_matches = img_tag_pattern.findall(script_content)
                for img_html in img_matches:
                    try:
                        img_soup = BeautifulSoup(img_html, 'html.parser')
                        img_tag = img_soup.find('img')
                        if img_tag:
                            candidate = self._create_candidate(img_tag, base_url, 'script img', site_id)
                            if candidate:
                                candidates.append(candidate)
                    except Exception:
                        continue
                
                # Extract image URLs from script content
                url_matches = url_pattern.findall(script_content)
                for img_url in url_matches:
                    if self._is_valid_image_url(img_url):
                        full_url = urljoin(base_url, img_url)
                        candidate = CandidateImage(
                            page_url=base_url,
                            img_url=full_url,
                            selector_hint='script url',
                            site_id=site_id,
                            alt_text='',
                            width=None,
                            height=None,
                            content_type=self._infer_content_type(full_url),
                            estimated_size=None,
                            has_srcset=False
                        )
                        candidates.append(candidate)
        except Exception as e:
            logger.debug(f"Error extracting script images: {e}")
        return candidates
    
    def _extract_images_from_json(self, json_data: Any) -> List[str]:
        """Recursively extract image URLs from JSON data."""
        image_urls = []
        try:
            if isinstance(json_data, dict):
                for key, value in json_data.items():
                    if key.lower() in ['image', 'thumbnail', 'src', 'srcset', 'url', 'photo', 'picture']:
                        if isinstance(value, str) and any(value.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                            image_urls.append(value)
                        elif isinstance(value, list):
                            for item in value:
                                if isinstance(item, str) and any(item.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                                    image_urls.append(item)
                    else:
                        image_urls.extend(self._extract_images_from_json(value))
            elif isinstance(json_data, list):
                for item in json_data:
                    image_urls.extend(self._extract_images_from_json(item))
        except Exception:
            pass
        return image_urls


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
