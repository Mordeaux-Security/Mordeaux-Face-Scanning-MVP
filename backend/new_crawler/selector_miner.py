"""
Selector Miner for New Crawler System

Cleaned up 3x3 selector mining logic with core patterns only.
Performs 3x3 crawl (3 category pages × 3 content pages) for better structure diversity.
"""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, AsyncIterator, Tuple
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import httpx

from .config import get_config
from .http_utils import get_http_utils
from .data_structures import CandidateImage, CandidatePost
from .extraction_tracer import get_extraction_tracer

logger = logging.getLogger(__name__)

# Singleton pattern per process
_selector_miner_instance = None


class SelectorMiner:
    """Selector miner with 3x3 crawling approach."""
    
    def __init__(self):
        self.config = get_config()
        self.http_utils = get_http_utils()
        self._semaphore = asyncio.Semaphore(12)  # Max 12 concurrent page fetches
        self.tracer = get_extraction_tracer()
        
        # Track visited URLs to prevent infinite loops and excessive navigation
        # Limit size to prevent unbounded memory growth (LRU-style: keep most recent 10,000)
        self._visited_urls: Set[str] = set()
        self._visited_urls_max_size: int = 10000
        
        # Strategy metrics tracking (cleared after each site to prevent unbounded growth)
        self._strategy_metrics: Dict[str, Dict[str, Any]] = {}
    
    def _update_strategy_metrics(self, site_id: str, strategy: str, success: bool, content_length: int, duration: float):
        """Update strategy-level metrics for tracking."""
        key = f"{site_id}:{strategy}"
        if key not in self._strategy_metrics:
            self._strategy_metrics[key] = {
                'site_id': site_id,
                'strategy': strategy,
                'attempts': 0,
                'successes': 0,
                'failures': 0,
                'total_content_length': 0,
                'total_duration': 0.0,
                'avg_content_length': 0,
                'avg_duration': 0.0
            }
        
        metrics = self._strategy_metrics[key]
        metrics['attempts'] += 1
        if success:
            metrics['successes'] += 1
            metrics['total_content_length'] += content_length
            metrics['total_duration'] += duration
        else:
            metrics['failures'] += 1
        
        # Update averages
        if metrics['successes'] > 0:
            metrics['avg_content_length'] = metrics['total_content_length'] / metrics['successes']
            metrics['avg_duration'] = metrics['total_duration'] / metrics['successes']
    
    async def _save_strategy_metrics_to_redis(self, site_id: str):
        """Save strategy metrics to Redis for cross-worker visibility."""
        try:
            from .redis_manager import get_redis_manager
            redis_manager = get_redis_manager()
            
            # Get all metrics for this site
            site_metrics = {k: v for k, v in self._strategy_metrics.items() if k.startswith(f"{site_id}:")}
            
            if site_metrics:
                metrics_key = f"extraction_metrics:{site_id}"
                # Convert to JSON-serializable format
                metrics_data = {k: v for k, v in site_metrics.items()}
                client = await redis_manager._get_async_client()
                await client.set(metrics_key, json.dumps(metrics_data, default=str))
                logger.info(f"[METRICS] Saved strategy metrics for site {site_id}")
        except Exception as e:
            logger.warning(f"[METRICS] Failed to save strategy metrics to Redis: {e}")
        
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

    async def mine_posts_with_3x3_crawl(self, base_url: str, site_id: str, max_pages: int = 5) -> AsyncIterator[Tuple[str, List[CandidatePost]]]:
        """Perform 3x3 crawl for diabetes-related posts: 1 base + 3 category + 3 content pages."""
        try:
            from .redis_manager import get_redis_manager
            from urllib.parse import urlparse
            redis = get_redis_manager()
            domain = urlparse(base_url).netloc

            logger.info(f"Starting 3x3 diabetes post crawl for {base_url} (max_pages={max_pages})")
            checked_urls = {base_url}
            pages_crawled = 0

            # PHASE 1: Sample crawl for diabetes posts

            # Step 1: Fetch base page
            logger.info(f"[3x3-POST-SAMPLE] Fetching base page: {base_url}")
            async with self._semaphore:
                html, error, _ = await self.http_utils.fetch_html(
                    base_url, use_js_fallback=True, force_compare_first_visit=False
                )

            if not html:
                logger.warning(f"Failed to fetch base page {base_url}: {error}")
                return

            pages_crawled += 1
            base_candidates = await self.mine_posts_for_diabetes(html, base_url, site_id)
            yield base_url, base_candidates

            # Get category/content URLs for sampling
            soup = BeautifulSoup(html, 'html.parser')
            category_urls = await self._discover_category_pages(soup, base_url)
            if not category_urls:
                category_urls = await self._discover_random_same_domain_links(soup, base_url, limit=3)

            # Step 2: Fetch up to 6 more sample pages in PARALLEL
            sample_tasks = []
            in_progress_urls = set()
            url_task_map = {}
            max_sample_pages = 7 if max_pages == -1 else min(7, max_pages)

            for category_url in category_urls[:6]:  # Take up to 6 more
                if category_url not in checked_urls and category_url not in in_progress_urls and pages_crawled < max_sample_pages:
                    in_progress_urls.add(category_url)
                    task = asyncio.create_task(self._fetch_post_page(category_url, site_id))
                    sample_tasks.append(task)
                    url_task_map[task] = category_url

            # Process sample pages as they complete
            for task in asyncio.as_completed(sample_tasks):
                original_url = url_task_map.get(task)
                url = original_url
                try:
                    candidates, returned_url = await task
                    url = returned_url
                    pages_crawled += 1

                    checked_urls.add(url)
                    in_progress_urls.discard(url)
                    if url != original_url and original_url:
                        in_progress_urls.discard(original_url)

                    logger.info(f"[3x3-POST-SAMPLE] Page yielded {len(candidates)} diabetes post candidates")
                    yield url, candidates

                except Exception as e:
                    logger.debug(f"Error processing post sample page: {e}")
                    if original_url:
                        in_progress_urls.discard(original_url)

            # PHASE 2: Continue BFS crawl if more pages needed
            if max_pages == -1 or pages_crawled < max_pages:
                remaining_pages = max_pages - pages_crawled if max_pages != -1 else 1000

                # Use discovered URLs for BFS expansion
                discovered_urls = set()
                for url in checked_urls:
                    if url != base_url:  # Don't rediscover base
                        discovered_urls.add(url)

                # Convert to list and sort for deterministic order
                bfs_queue = sorted(list(discovered_urls))

                logger.info(f"[3x3-POST-BFS] Starting BFS with {len(bfs_queue)} URLs, remaining_pages={remaining_pages}")

                for url in bfs_queue:
                    if url in checked_urls:
                        continue
                    if max_pages != -1 and pages_crawled >= max_pages:
                        break

                    try:
                        candidates, _ = await self._fetch_post_page(url, site_id)
                        pages_crawled += 1
                        checked_urls.add(url)

                        logger.info(f"[3x3-POST-BFS] BFS page yielded {len(candidates)} diabetes post candidates")
                        yield url, candidates

                    except Exception as e:
                        logger.debug(f"Error in BFS crawl for {url}: {e}")
                        continue

            logger.info(f"[3x3-POST] Completed diabetes crawl: {pages_crawled} pages, {len(checked_urls)} URLs checked")

        except Exception as e:
            logger.error(f"Error in 3x3 diabetes post crawl for {base_url}: {e}")

    async def _fetch_post_page(self, url: str, site_id: str) -> Tuple[List[CandidatePost], str]:
        """Fetch a page and mine diabetes-related posts."""
        async with self._semaphore:
            html, error, _ = await self.http_utils.fetch_html(
                url, use_js_fallback=True, force_compare_first_visit=False
            )

        if not html:
            logger.debug(f"Failed to fetch post page {url}: {error}")
            return [], url

        candidates = await self.mine_posts_for_diabetes(html, url, site_id)
        return candidates, url

    def _detect_page_type(self, soup: BeautifulSoup, url: Optional[str] = None) -> str:
        """
        Detect if page is a thread listing page or individual post page.
        
        Args:
            soup: BeautifulSoup parsed HTML
            url: Optional URL to use for URL-based heuristics
        
        Returns:
            'listing' - Thread listing page (multiple posts/threads)
            'detail' - Individual post/thread page (single post with replies)
        """
        try:
            logger.info(f"[PAGE-TYPE] Starting detection for URL: {url}")
            # URL-based heuristics (strong signal, check first)
            url_indicators = {'listing': 0, 'detail': 0}
            if url:
                url_lower = url.lower()
                logger.info(f"[PAGE-TYPE] URL lower: {url_lower}")
                # Detail page URL patterns (check for /posts/ plural and /boards/threads/)
                detail_patterns = ['/comments/', '/thread/', '/threads/', '/post/', '/posts/', '/topic/', '/p/', '/t/']
                if any(pattern in url_lower for pattern in detail_patterns):
                    url_indicators['detail'] += 2  # Strong indicator
                    logger.info(f"[PAGE-TYPE] URL matches detail pattern")
                # Listing page URL patterns
                elif any(pattern in url_lower for pattern in ['/forum/', '/boards/', '/r/']):
                    # Reddit: /r/subreddit/ without /comments/ is listing
                    if '/r/' in url_lower and '/comments/' not in url_lower:
                        url_indicators['listing'] += 2  # Strong indicator
                        logger.info(f"[PAGE-TYPE] URL matches Reddit listing pattern")
                    # Other forums: /forum/ or /boards/ without detail path is listing
                    # But /boards/threads/ is detail, /boards/forums/ is listing
                    elif '/boards/threads/' in url_lower:
                        url_indicators['detail'] += 2  # Strong indicator
                        logger.info(f"[PAGE-TYPE] URL matches boards/threads detail pattern")
                    elif ('/forum/' in url_lower or '/boards/' in url_lower) and not any(detail in url_lower for detail in ['/thread/', '/threads/', '/post/', '/posts/', '/topic/']):
                        url_indicators['listing'] += 1
                        logger.info(f"[PAGE-TYPE] URL matches forum listing pattern")
                logger.info(f"[PAGE-TYPE] URL indicators: {url_indicators}")
            else:
                logger.info(f"[PAGE-TYPE] No URL provided for detection")
            
            # Listing page indicators - use generic heuristics
            listing_indicators = []
            
            # Check HTML/data attributes for forum template types (XenForo, etc.)
            html_tag = soup.find('html')
            if html_tag:
                data_template = html_tag.get('data-template', '')
                if 'forum_view' in data_template or 'forum_list' in data_template or 'thread_list' in data_template:
                    listing_indicators.append(True)
                    listing_indicators.append(True)  # Strong indicator
                # Note: 'thread_view' or 'post_view' would indicate detail page, checked later
            
            # Count repeated similar structures (key indicator of listings)
            # Look for containers that appear multiple times with similar structure
            all_divs = soup.find_all('div', class_=True)
            class_counts = {}
            for div in all_divs:
                classes = ' '.join(div.get('class', []))
                if classes:
                    class_counts[classes] = class_counts.get(classes, 0) + 1
            
            # If we have 3+ elements with the same class, likely a listing
            repeated_classes = [count for count in class_counts.values() if count >= 3]
            if repeated_classes:
                listing_indicators.append(True)
            
            # Table-based forum listings (vBulletin-style)
            listing_indicators.extend([
                len(soup.select('tbody[id*="threadbits"]')) > 0,
                len(soup.select('tbody[id*="threadbits"] tr')) >= 3,
                len(soup.select('td[id*="threadtitle"], td[id*="td_threadtitle"]')) >= 3,
            ])
            
            # Activity stream patterns (Mayo Clinic style)
            listing_indicators.extend([
                len(soup.select('.activity-stream, .ch-activity-stream')) > 0,
                len(soup.select('.ch-activity-simple-row, .activity-simple-row')) >= 3,
                len(soup.select('a.discussion-title, a[class*="discussion"]')) >= 3,
            ])
            
            # Multiple post containers with similar structure
            listing_indicators.extend([
                len(soup.select('.thread-list, .post-list, .topic-list')) > 0,
                len(soup.select('li.thread, li.post, li.topic')) >= 3,
            ])
            
            # XenForo-style structItem patterns (common in forum listings)
            listing_indicators.extend([
                len(soup.select('.structItem, [class*="structItem"]')) >= 3,
                len(soup.select('.structItem-title, [class*="structItem-title"]')) >= 3,
                len(soup.select('[data-last], [data-author]')) >= 3,  # XenForo thread metadata
            ])
            
            # Generic repeated structures with links (likely listing items)
            all_elements_with_links = soup.find_all(True, recursive=False)
            link_counts = {}
            for elem in all_elements_with_links:
                if elem.find('a', href=True):
                    tag_class = f"{elem.name}.{'.'.join(elem.get('class', [])[:2])}"
                    link_counts[tag_class] = link_counts.get(tag_class, 0) + 1
            repeated_link_structures = [count for count in link_counts.values() if count >= 3]
            if repeated_link_structures:
                listing_indicators.append(True)
            
            # Pagination controls
            listing_indicators.append(len(soup.select('.pagination, .pager, .page-nav, [class*="pagination"], .chPagination')) > 0)
            
            # List structures with multiple items
            listing_indicators.append(len(soup.select('ul.thread-list, ol.thread-list, ul.post-list, ol.post-list')) > 0)
            
            # Thread titles without full content (common in listings)
            listing_indicators.append(len(soup.select('.thread-title, .post-title, .topic-title, a[id*="thread_title"], a.discussion-title, a[class*="discussion-title"], .structItem-title')) >= 3)
            
            # Detail page indicators - look for multiple instances of comment/reply/response elements
            # This is a strong signal that we're on an individual post page, not a listing
            
            # Count comment/reply/response elements (tags, classes, IDs, attributes)
            comment_keywords = ['comment', 'reply', 'response', 'replies', 'responses', 'comments']
            comment_count = 0
            
            # Count custom elements (e.g., <shreddit-comment>, <comment-tree>)
            for keyword in comment_keywords:
                # Use lambda to match tag names containing keyword
                comment_count += len(soup.find_all(lambda tag: tag.name and re.search(keyword, tag.name, re.I)))
            
            # Count elements with comment keywords in class/id/attributes
            for keyword in comment_keywords:
                # Classes
                comment_count += len(soup.select(f'[class*="{keyword}" i]'))
                # IDs
                comment_count += len(soup.select(f'[id*="{keyword}" i]'))
                # Data attributes
                comment_count += len(soup.select(f'[data-*="{keyword}" i]'))
                # Slot attributes
                comment_count += len(soup.select(f'[slot*="{keyword}" i]'))
            
            # Comment composer/input fields (usually only on post pages)
            comment_inputs = len(soup.select('textarea[placeholder*="comment" i], textarea[placeholder*="reply" i], textarea[placeholder*="thought" i], [class*="comment-composer" i], [class*="reply-composer" i], [id*="comment-composer" i]'))
            
            # Comment trees/threads
            comment_trees = len(soup.select('[class*="comment-tree" i], [id*="comment-tree" i], [class*="reply-tree" i], [id*="reply-tree" i]'))
            
            # Comment sorting dropdowns (usually only on post pages)
            comment_sort = len(soup.select('[class*="comment-sort" i], [id*="comment-sort" i], [class*="sort-comment" i]'))
            
            # Multiple comment metadata elements (author, timestamp repeated)
            comment_metadata = len(soup.select('[class*="comment-meta" i], [class*="reply-meta" i], [class*="comment-author" i], [class*="reply-author" i]'))
            
            # Nested comment structures (depth indicators)
            comment_depth = len(soup.select('[depth], [data-depth], [class*="depth" i][class*="comment" i], [class*="depth" i][class*="reply" i]'))
            
            # "More replies" or similar links
            more_replies = len(soup.find_all('a', href=True, string=re.compile(r'more\s+(replies?|comments?|responses?)', re.I)))
            
            # Detail page indicators
            detail_indicators = [
                # Multiple comment/reply elements (3+ is a strong signal)
                comment_count >= 3,
                # Comment composer/input present
                comment_inputs > 0,
                # Comment trees present
                comment_trees > 0,
                # Comment sorting present
                comment_sort > 0,
                # Multiple comment metadata elements (3+)
                comment_metadata >= 3,
                # Nested comment structures
                comment_depth > 0,
                # "More replies" links
                more_replies > 0,
                # Posts container (common in vBulletin-style forums)
                len(soup.select('div#posts, div[id*="posts"]')) > 0,
                # Thread post containers (vBulletin-style)
                len(soup.select('div.threadpost, div[id*="edit"][id*="post"]')) > 0,
                # Post message divs (vBulletin-style)
                len(soup.select('div[id*="post_message"]')) > 0,
                # Single main post with full content
                len(soup.select('.post-content, .message-content, .post-body, .message-body')) == 1,
                # Reply/comment sections (legacy patterns)
                len(soup.select('.replies, .reply-list, .comment-list, .comments')) > 0,
                # Breadcrumb navigation
                len(soup.select('.breadcrumb, .breadcrumbs, [class*="breadcrumb"]')) > 0,
                # Single post container with replies nested inside
                len(soup.select('.post .reply, .message .reply, .thread .reply')) > 0,
            ]
            
            listing_score = sum(listing_indicators)
            detail_score = sum(detail_indicators)
            
            # Count multiple similar post/thread structures (strong listing indicator)
            # IMPORTANT: Require that elements have links (not just any article/post)
            post_elements = soup.select('.post, .thread, .topic, .message, .structItem, article')
            # Filter to only elements that contain links (more specific for listings)
            post_elements_with_links = [elem for elem in post_elements if elem.find('a', href=True)]
            multiple_posts = len(post_elements_with_links) >= 3
            
            # Reddit-specific detection
            reddit_listing_indicators = 0
            reddit_detail_indicators = 0
            if url and '/r/' in url.lower():
                # Reddit listing: many <article> elements with links, no shreddit-comment
                if len(post_elements_with_links) >= 3 and len(soup.select('shreddit-comment')) == 0:
                    reddit_listing_indicators += 2
                # Reddit detail: shreddit-comment elements present
                if len(soup.select('shreddit-comment')) >= 3:
                    reddit_detail_indicators += 2
                # Reddit detail: shreddit-post with nested comments
                if len(soup.select('shreddit-post')) == 1 and len(soup.select('shreddit-comment')) >= 1:
                    reddit_detail_indicators += 1
            
            # Improve comment counting: only count actual comment/reply elements, not generic "comment" in class names
            # Count actual comment elements more precisely
            precise_comment_count = 0
            # Custom comment elements (Reddit, etc.)
            precise_comment_count += len(soup.select('shreddit-comment, comment-tree, comment-item'))
            # Comment containers with actual structure
            precise_comment_count += len(soup.select('[class*="comment"][class*="tree"], [class*="reply"][class*="tree"], [id*="comment"][id*="tree"]'))
            # Comment metadata elements (indicates actual comments, not just class names)
            precise_comment_count += len(soup.select('[class*="comment-meta"], [class*="reply-meta"], [class*="comment-author"], [class*="reply-author"]'))
            # Nested comment structures
            precise_comment_count += len(soup.select('[depth], [data-depth], [class*="depth"][class*="comment"], [class*="depth"][class*="reply"]'))
            
            # Use precise comment count for detail detection
            precise_comment_count_for_detail = precise_comment_count
            
            logger.info(f"[PAGE-TYPE] Listing score: {listing_score}, Detail score: {detail_score}, multiple_posts={multiple_posts}, precise_comment_count={precise_comment_count_for_detail}, url_indicators={url_indicators}")
            
            # URL-based classification (strongest signal, check first)
            if url_indicators['detail'] >= 2:
                logger.info(f"[PAGE-TYPE] Classified as DETAIL: URL pattern indicates detail page")
                return 'detail'
            if url_indicators['listing'] >= 2:
                logger.info(f"[PAGE-TYPE] Classified as LISTING: URL pattern indicates listing page")
                return 'listing'
            
            # Reddit-specific classification
            if reddit_detail_indicators >= 2:
                logger.info(f"[PAGE-TYPE] Classified as DETAIL: Reddit detail page indicators")
                return 'detail'
            if reddit_listing_indicators >= 2:
                logger.info(f"[PAGE-TYPE] Classified as LISTING: Reddit listing page indicators")
                return 'listing'
            
            # STRONG listing indicators (take precedence):
            # 1. Multiple similar post structures with links (3+)
            # 2. Strong listing score (3+)
            if multiple_posts or listing_score >= 3:
                logger.info(f"[PAGE-TYPE] Classified as LISTING: multiple_posts={multiple_posts}, listing_score={listing_score}")
                return 'listing'
            
            # STRONG detail indicators (require multiple signals):
            # Need at least 3 detail indicators AND precise_comment_count >= 5 (to avoid false positives)
            if detail_score >= 3 and precise_comment_count_for_detail >= 5:
                logger.info(f"[PAGE-TYPE] Classified as DETAIL: detail_score={detail_score}, precise_comment_count={precise_comment_count_for_detail}")
                return 'detail'
            
            # MEDIUM listing indicators:
            # Listing score >= 2 OR multiple posts with links (2+)
            if listing_score >= 2 or (len(post_elements_with_links) >= 2):
                logger.info(f"[PAGE-TYPE] Classified as LISTING: listing_score={listing_score}, post_elements_with_links={len(post_elements_with_links)}")
                return 'listing'
            
            # MEDIUM detail indicators:
            # Detail score >= 2 AND precise_comment_count >= 3
            if detail_score >= 2 and precise_comment_count_for_detail >= 3:
                logger.info(f"[PAGE-TYPE] Classified as DETAIL: detail_score={detail_score}, precise_comment_count={precise_comment_count_for_detail}")
                return 'detail'
            
            # URL as tiebreaker
            if url_indicators['detail'] >= 1:
                logger.info(f"[PAGE-TYPE] Classified as DETAIL: URL pattern suggests detail (weak)")
                return 'detail'
            if url_indicators['listing'] >= 1:
                logger.info(f"[PAGE-TYPE] Classified as LISTING: URL pattern suggests listing (weak)")
                return 'listing'
            
            # WEAK indicators - prefer listing (forums are usually listing pages):
            if listing_score >= 1:
                logger.info(f"[PAGE-TYPE] Classified as LISTING: listing_score={listing_score} (weak indicator)")
                return 'listing'
            
            # Default to listing if unclear (most forum pages are listings)
            logger.info(f"[PAGE-TYPE] Classified as LISTING: default (unclear, assuming listing)")
            return 'listing'
                
        except Exception as e:
            logger.error(f"[PAGE-TYPE] Error detecting page type: {e}", exc_info=True)
            # Default to listing on error (most forum pages are listings)
            # But log the error so we can debug
            if url:
                # Try URL-based fallback
                url_lower = url.lower()
                if any(pattern in url_lower for pattern in ['/comments/', '/thread/', '/threads/', '/post/', '/posts/', '/boards/threads/']):
                    logger.info(f"[PAGE-TYPE] Error fallback: URL suggests detail page")
                    return 'detail'
                elif '/r/' in url_lower and '/comments/' not in url_lower:
                    logger.info(f"[PAGE-TYPE] Error fallback: URL suggests listing page")
                    return 'listing'
            logger.info(f"[PAGE-TYPE] Error fallback: defaulting to listing")
            return 'listing'  # Default to listing on error (most forum pages are listings)
    
    def _find_post_listing_items(self, soup: BeautifulSoup) -> List:
        """Find HTML elements that contain discussion/post/thread keywords and have links.
        
        Uses loose matching to find elements by:
        - Custom element names (e.g., <shreddit-post>, <forum-thread>)
        - Class names (exact or partial matches)
        - ID attributes (exact or partial matches)
        - Other attributes (data-*, slot, name, etc.)
        - Element tag names containing keywords
        """
        post_items = []
        
        # Keywords to look for in classes, ids, attributes, tag names, or text
        keywords = ['discussion', 'post', 'thread', 'topic', 'message', 'forum', 'board', 'comment', 'reply']
        
        # Strategy 1: Table-based forum patterns (vBulletin-style) - keep specific patterns
        table_rows = soup.select('tbody[id*="threadbits"] tr, tbody tr[id*="thread"], tr[id*="thread"]')
        for row in table_rows:
            if row.find('a', href=True):
                post_items.append(row)
        
        # Strategy 2: Activity stream patterns (Mayo Clinic style)
        activity_rows = soup.select('.ch-activity-simple-row, .activity-simple-row, div[class*="activity"][class*="row"]')
        for row in activity_rows:
            if row.find('a', href=True):
                post_items.append(row)
        
        # Strategy 3: XenForo structItem patterns (very common)
        struct_items = soup.select('.structItem, [class*="structItem"]:not([class*="structItem--minimal"])')
        for item in struct_items:
            if item.find('a', href=True):
                post_items.append(item)
        
        # Strategy 4: List-based patterns
        list_items = soup.select('.thread-list .thread, .post-list .post, .topic-list .topic, li.thread, li.post, li.topic, li.message')
        for item in list_items:
            if item.find('a', href=True):
                post_items.append(item)
        
        # Strategy 4: Custom elements (e.g., <shreddit-post>, <forum-thread>, <discussion-item>)
        for keyword in keywords:
            # Find custom elements with keyword in tag name (case-insensitive)
            # e.g., <shreddit-post>, <reddit-post>, <forum-post>, <thread-item>
            custom_elements = soup.find_all(lambda tag: tag.name and keyword.lower() in tag.name.lower())
            for elem in custom_elements:
                if elem.find('a', href=True):
                    if elem not in post_items:
                        post_items.append(elem)
        
        # Strategy 5: Find elements by ID containing keywords (loose matching)
        for keyword in keywords:
            # Find elements with keyword in ID attribute
            elements_by_id = soup.find_all(True, id=re.compile(keyword, re.I))
            for elem in elements_by_id:
                if elem.find('a', href=True):
                    if elem not in post_items:
                        post_items.append(elem)
        
        # Strategy 6: Find elements by class containing keywords (loose matching)
        for keyword in keywords:
            # Find elements with keyword in class (partial match)
            elements_by_class = soup.find_all(True, class_=re.compile(keyword, re.I))
            for elem in elements_by_class:
                if elem.find('a', href=True):
                    if elem not in post_items:
                        post_items.append(elem)
        
        # Strategy 7: Find elements by other attributes (data-*, slot, name, role, etc.)
        attribute_patterns = ['data-', 'slot', 'name', 'role', 'itemtype', 'itemscope']
        for keyword in keywords:
            for attr_pattern in attribute_patterns:
                # Find elements with keyword in any attribute
                elements = soup.find_all(True, attrs={attr_pattern: re.compile(keyword, re.I)})
                # Also check all attributes for keyword
                all_elements = soup.find_all(True)
                for elem in all_elements:
                    # Check all attributes for keyword
                    for attr_name, attr_value in elem.attrs.items():
                        if isinstance(attr_value, (list, tuple)):
                            attr_value = ' '.join(str(v) for v in attr_value)
                        if attr_value and keyword.lower() in str(attr_value).lower():
                            if elem.find('a', href=True):
                                if elem not in post_items:
                                    post_items.append(elem)
                            break  # Found keyword in this element, move to next
        
        # Strategy 8: Generic loose matching - check element name, classes, ids, and attributes
        if not post_items:
            all_elements = soup.find_all(True)  # Find all elements
            for elem in all_elements:
                # Skip if already found or doesn't have links
                if elem in post_items or not elem.find('a', href=True):
                    continue
                
                # Check tag name
                tag_name = elem.name.lower() if elem.name else ''
                
                # Check classes
                classes = ' '.join(elem.get('class', [])).lower()
                
                # Check ID
                elem_id = elem.get('id', '').lower()
                
                # Check all attributes
                attrs_text = ' '.join([f"{k}={v}" for k, v in elem.attrs.items() if isinstance(v, str)]).lower()
                
                # Check if any keyword appears in tag name, classes, id, or attributes
                element_text = f"{tag_name} {classes} {elem_id} {attrs_text}"
                if any(kw in element_text for kw in keywords):
                    post_items.append(elem)
        
        # Strategy 9: Fallback - find any element with keyword in text content (last resort)
        if not post_items:
            for keyword in keywords:
                # Find elements containing keyword in text, but also check if they have structure suggesting posts
                elements = soup.find_all(True, string=re.compile(keyword, re.I))
                for elem in elements:
                    # Get parent or the element itself if it's a container
                    container = elem.parent if elem.parent else elem
                    # Check if it looks like a post container (has link, has some structure)
                    if container.find('a', href=True) and len(container.get_text(strip=True)) > 20:
                        if container not in post_items:
                            post_items.append(container)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_items = []
        for item in post_items:
            item_id = id(item)
            if item_id not in seen:
                seen.add(item_id)
                unique_items.append(item)
        
        logger.info(f"[POST-FIND] Found {len(unique_items)} post listing items using loose matching")
        if unique_items:
            # Log sample of found items for debugging
            sample_items = unique_items[:3]
            for i, item in enumerate(sample_items):
                tag_name = item.name if item.name else 'unknown'
                classes = ' '.join(item.get('class', []))[:50] if item.get('class') else 'no-class'
                elem_id = item.get('id', '')[:50] if item.get('id') else 'no-id'
                logger.info(f"[POST-FIND] Sample item {i+1}: <{tag_name}> class='{classes}' id='{elem_id}'")
            
            # Write found items to file
            debug_dir = Path("backend/crawl_output/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            items_file = debug_dir / f"post_items_{id(soup)}.json"
            items_data = []
            for item in unique_items[:10]:  # First 10 items
                items_data.append({
                    'tag': item.name if item.name else 'unknown',
                    'classes': list(item.get('class', []))[:5],
                    'id': item.get('id', '')[:100],
                    'text_preview': item.get_text(strip=True)[:200],
                    'has_links': bool(item.find('a', href=True))
                })
            with open(items_file, 'w', encoding='utf-8') as f:
                json.dump(items_data, f, indent=2)
            logger.info(f"[POST-FIND] Wrote {len(items_data)} post items to {items_file}")
        else:
            # Write why no items were found
            debug_dir = Path("backend/crawl_output/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            no_items_file = debug_dir / f"no_post_items_{id(soup)}.json"
            with open(no_items_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_elements': len(soup.find_all(True)),
                    'total_links': len(soup.find_all('a', href=True)),
                    'has_threadbits': len(soup.select('tbody[id*="threadbits"]')) > 0,
                    'has_activity_rows': len(soup.select('.ch-activity-simple-row')) > 0
                }, f, indent=2)
            logger.info(f"[POST-FIND] Wrote no-items analysis to {no_items_file}")
        return unique_items
    
    def _extract_post_links(self, post_items: List, base_url: str) -> List[tuple]:
        """Extract navigation links from post listing items, prioritize by keywords."""
        post_links = []
        discussion_keywords = ['discussion', 'thread', 'post', 'topic', 'message', 'comment', '/t/', '/p/', '/comment/']
        diabetes_keywords = [
            'diabetes', 'diabetic', 'insulin', 'blood sugar', 'glucose',
            'type 1 diabetes', 'type 2 diabetes', 'gestational diabetes',
            'diabetes mellitus', 'hyperglycemia', 'hypoglycemia',
            'a1c', 'hba1c', 'blood glucose', 'sugar levels'
        ]
        
        for post_item in post_items:
            # Find all links within the post item (descendant search)
            links = post_item.find_all('a', href=True)
            
            if not links:
                continue
            
            # Prefer links with discussion/post/thread keywords in href
            best_link = None
            best_link_text = ""
            
            for link in links:
                href = link.get('href', '').lower()
                link_text = link.get_text(strip=True).lower()
                
                # Filter out non-post links (user profiles, categories, search, etc.)
                non_post_patterns = ['/user/', '/profile/', '/member/', '/category/', '/search', '/tag/', '/author/']
                if any(pattern in href for pattern in non_post_patterns):
                    continue
                
                # Check if link looks like a discussion/post link
                is_discussion_link = any(kw in href for kw in discussion_keywords)
                
                # Validate link matches discussion URL patterns
                discussion_url_patterns = ['/thread/', '/post/', '/topic/', '/t/', '/p/', '/comments/', '/discussion/']
                matches_discussion_pattern = any(pattern in href for pattern in discussion_url_patterns)
                
                # Check if title contains diabetes keywords
                has_diabetes_in_title = any(kw in link_text for kw in diabetes_keywords)
                
                # Prefer discussion links with meaningful text and valid patterns
                if (is_discussion_link or matches_discussion_pattern) and len(link_text) > 5:
                    if not best_link or has_diabetes_in_title:
                        best_link = link
                        best_link_text = link_text
                        if has_diabetes_in_title:
                            break  # Found diabetes keyword, use this one
            
            # Fallback to first link if no discussion link found (but still filter non-post links)
            if not best_link and links:
                for link in links:
                    href = link.get('href', '').lower()
                    # Skip non-post links even in fallback
                    non_post_patterns = ['/user/', '/profile/', '/member/', '/category/', '/search', '/tag/', '/author/']
                    if not any(pattern in href for pattern in non_post_patterns):
                        best_link = link
                        best_link_text = best_link.get_text(strip=True).lower()
                        break
            
            if best_link:
                href = best_link.get('href')
                if href:
                    link_url = urljoin(base_url, href)
                    has_keywords = any(kw in best_link_text for kw in diabetes_keywords)
                    post_links.append((post_item, link_url, has_keywords))
        
        logger.info(f"[POST-LINKS] Extracted {len(post_links)} post links from {len(post_items)} post items")
        if post_links:
            # Log sample links
            sample_links = post_links[:3]
            for i, (item, link_url, has_keywords) in enumerate(sample_links):
                logger.info(f"[POST-LINKS] Sample link {i+1}: {link_url[:80]}... (has_keywords={has_keywords})")
        return post_links
    
    async def _navigate_and_read_post(self, post_url: str, page_url: str, site_id: str) -> Optional[CandidatePost]:
        """Navigate to post link, fetch HTML, extract full content, check keywords.
        
        IMPORTANT: This should only be called for individual post pages, not listing pages.
        If the fetched page is detected as a listing page, we skip it to prevent infinite recursion.
        """
        try:
            # Check if already visited (in-memory check first, then Redis for cross-worker dedup)
            normalized_url = post_url.split('#')[0].rstrip('/')
            
            # Check Redis for cross-worker deduplication (in-memory check already done in mine_posts_for_diabetes)
            try:
                from .redis_manager import get_redis_manager
                redis_manager = get_redis_manager()
                redis_key = f"nc:visited_url:{normalized_url}"
                client = await redis_manager._get_async_client()
                exists = await client.exists(redis_key)
                if exists:
                    logger.debug(f"[POST-NAV] Skipping already visited URL (Redis): {normalized_url}")
                    return None
            except Exception as e:
                logger.warning(f"[POST-NAV] Failed to check Redis for URL dedup: {e}")
                # Continue anyway - in-memory dedup in mine_posts_for_diabetes still works
            
            logger.info(f"[POST-NAV] Navigating to post: {post_url[:80]}...")
            # Fetch HTML from post link
            async with self._semaphore:
                html, error, _ = await self.http_utils.fetch_html(
                    post_url, use_js_fallback=True, force_compare_first_visit=False
                )
            
            if not html:
                logger.info(f"[POST-NAV] Failed to fetch post page {post_url}: {error}")
                return None
            
            logger.info(f"[POST-NAV] Fetched {len(html)} chars from {post_url}")
            
            # Extract full post and replies
            soup = BeautifulSoup(html, 'html.parser')
            
            # CRITICAL: Check if this is actually a listing page (prevent infinite recursion)
            page_type = self._detect_page_type(soup, post_url)
            if page_type == 'listing':
                logger.warning(f"[POST-NAV] Post URL {post_url} is actually a listing page, skipping to prevent infinite recursion")
                return None
            
            post_data = await self._extract_full_post_and_replies(soup, post_url, site_id)
            
            if not post_data:
                logger.info(f"[POST-NAV] Could not extract post data from {post_url}")
                return None
            
            logger.info(f"[POST-NAV] Extracted post data: title={bool(post_data.get('title'))}, content_len={len(post_data.get('content', ''))}")
            
            # Check if post contains diabetes keywords (but return all posts)
            has_keywords = self._contains_diabetes_keyword_in_content(post_data)
            if has_keywords:
                logger.info(f"[POST-NAV] Post contains diabetes keywords: {post_url}")
            else:
                logger.info(f"[POST-NAV] Post does not contain diabetes keywords (will save to raw-images only): {post_url}")
            
            # Mark as visited in Redis after successful extraction (for cross-worker dedup)
            try:
                from .redis_manager import get_redis_manager
                redis_manager = get_redis_manager()
                redis_key = f"nc:visited_url:{normalized_url}"
                client = await redis_manager._get_async_client()
                await client.setex(redis_key, 86400, "1")
            except Exception as e:
                logger.warning(f"[POST-NAV] Failed to mark URL as visited in Redis: {e}")
                # Continue anyway - in-memory dedup still works
            
            # Create candidate from post data (return all posts, not just keyword-filtered)
            return self._create_post_candidate_from_data(post_data, post_url, page_url, site_id)
            
        except Exception as e:
            logger.info(f"[POST-NAV] Error navigating to post {post_url}: {e}")
            return None
    
    async def _extract_full_post_and_replies(self, soup: BeautifulSoup, post_url: str, site_id: str) -> Optional[dict]:
        """Extract main post and all replies from detail page."""
        strategy_order = []
        strategy_results = {}
        extraction_start = time.time()
        
        try:
            # Find main post
            main_post = None
            main_title = None
            main_author = None
            main_date = None
            main_content = ""
            
            # Strategy 0: Reddit-specific extraction (<shreddit-post> and <shreddit-comment>)
            shreddit_posts = soup.select('shreddit-post')
            if shreddit_posts:
                main_post = shreddit_posts[0]
                logger.info(f"[POST-EXTRACT] Found Reddit post using shreddit-post element")
                
                # Extract title from h1 or [slot="title"]
                title_elem = main_post.select_one('h1, [slot="title"], h2[slot="title"]')
                if title_elem:
                    main_title = title_elem.get_text(strip=True)
                else:
                    # Fallback: look for title in any heading
                    for heading in main_post.find_all(['h1', 'h2', 'h3']):
                        title_text = heading.get_text(strip=True)
                        if title_text and len(title_text) > 5:
                            main_title = title_text
                            break
                
                # Extract content from [slot="text"] or .md elements
                content_elem = main_post.select_one('[slot="text"], .md, [class*="md"]')
                if content_elem:
                    main_content = content_elem.get_text(strip=True)
                else:
                    # Fallback: get all text from post, excluding nested comments
                    # Clone to avoid modifying original
                    post_clone = BeautifulSoup(str(main_post), 'html.parser')
                    # Remove comment elements
                    for comment in post_clone.select('shreddit-comment'):
                        comment.decompose()
                    main_content = post_clone.get_text(strip=True)
                
                # Extract author from [slot="author"] or similar
                author_elem = main_post.select_one('[slot="author"], [data-author], .author, [class*="author"]')
                if author_elem:
                    main_author = author_elem.get_text(strip=True)
                    # Reddit format: u/username
                    if main_author.startswith('u/'):
                        main_author = main_author[2:]
                
                # Extract date from time element or [slot="time"]
                date_elem = main_post.select_one('time[datetime], [slot="time"], time')
                if date_elem:
                    main_date = self._extract_date_from_element(date_elem)
            
            # Strategy 1: vBulletin-style thread posts (first one is usually main)
            if not main_post:
                thread_posts = soup.select('div.threadpost, div[id*="edit"][id*="post"]')
                if thread_posts:
                    main_post = thread_posts[0]
                    # Extract title, author, date, content from first post
                    main_title = self._extract_title_from_element(main_post)
                    main_author = self._extract_author_from_element(main_post)
                    main_date = self._extract_date_from_element(main_post)
                    main_content = self._extract_content_from_element(main_post)
            
            # Strategy 2: Post message divs (vBulletin-style)
            if not main_post:
                post_messages = soup.select('div[id*="post_message"]')
                logger.info(f"[POST-EXTRACT] Found {len(post_messages)} post message divs")
                if post_messages:
                    main_post = post_messages[0]
                    # Find parent container for title/author/date
                    parent = main_post.find_parent(['div', 'table'], class_=re.compile(r'threadpost|post', re.I))
                    if parent:
                        main_title = self._extract_title_from_element(parent)
                        main_author = self._extract_author_from_element(parent)
                        main_date = self._extract_date_from_element(parent)
                    main_content = self._extract_content_from_element(main_post)
                    logger.info(f"[POST-EXTRACT] Extracted from post_message: title={bool(main_title)}, content_len={len(main_content)}")
            
            # Strategy 3: Standard post patterns with loose matching (similar to _find_post_listing_items)
            if not main_post:
                logger.info(f"[POST-EXTRACT] Trying loose matching strategies...")
                keywords = ['post', 'message', 'thread', 'discussion', 'topic', 'comment', 'reply']
                
                # Strategy 3a: Find elements by class containing keywords (loose matching with regex)
                for keyword in keywords:
                    elements_by_class = soup.find_all(True, class_=re.compile(keyword, re.I))
                    for elem in elements_by_class:
                        # Skip if it looks like navigation, header, or error messages
                        elem_classes = ' '.join(elem.get('class', [])).lower()
                        elem_id = elem.get('id', '').lower()
                        if any(skip in elem_classes or skip in elem_id for skip in ['nav', 'header', 'footer', 'menu', 'sidebar', 'error', 'warning', 'alert', 'notification', 'empty', 'no-result', 'not-found']):
                            continue
                        candidate_content = self._extract_content_from_element(elem)
                        if candidate_content and len(candidate_content) > 20:
                            # Validate content quality
                            is_valid, reason = self._validate_post_content(elem, min_length=20)
                            if is_valid:
                                logger.info(f"[POST-EXTRACT] Found post using loose class match: keyword='{keyword}', class={elem.get('class', [])}")
                                main_post = elem
                                main_title = self._extract_title_from_element(main_post)
                                main_author = self._extract_author_from_element(main_post)
                                main_date = self._extract_date_from_element(main_post)
                                main_content = candidate_content
                                break
                            else:
                                logger.debug(f"[POST-EXTRACT] Rejected candidate: {reason}")
                    if main_post:
                        break
                
                # Strategy 3b: Find elements by ID containing keywords (loose matching)
                if not main_post:
                    for keyword in keywords:
                        elements_by_id = soup.find_all(True, id=re.compile(keyword, re.I))
                        for elem in elements_by_id:
                            candidate_content = self._extract_content_from_element(elem)
                            if candidate_content and len(candidate_content) > 20:
                                logger.info(f"[POST-EXTRACT] Found post using loose ID match: keyword='{keyword}', id={elem.get('id', '')}")
                                main_post = elem
                                main_title = self._extract_title_from_element(main_post)
                                main_author = self._extract_author_from_element(main_post)
                                main_date = self._extract_date_from_element(main_post)
                                main_content = candidate_content
                                break
                        if main_post:
                            break
                
                # Strategy 3c: Find custom elements with keywords in tag name (loose matching)
                if not main_post:
                    for keyword in keywords:
                        custom_elements = soup.find_all(lambda tag: tag.name and keyword in tag.name.lower())
                        for elem in custom_elements:
                            candidate_content = self._extract_content_from_element(elem)
                            if candidate_content and len(candidate_content) > 20:
                                logger.info(f"[POST-EXTRACT] Found post using custom element: {elem.name}")
                                main_post = elem
                                main_title = self._extract_title_from_element(main_post)
                                main_author = self._extract_author_from_element(main_post)
                                main_date = self._extract_date_from_element(main_post)
                                main_content = candidate_content
                                break
                        if main_post:
                            break
                
                # Strategy 3d: Find elements by attributes containing keywords (data-*, slot, role, etc.)
                if not main_post:
                    attribute_patterns = ['data-', 'slot', 'name', 'role', 'itemtype', 'itemscope']
                    for keyword in keywords:
                        for attr_pattern in attribute_patterns:
                            # Find elements with keyword in any attribute (limit to common content containers)
                            all_elements = soup.find_all(['div', 'section', 'article', 'main', 'li'])
                            for elem in all_elements:
                                # SKIP error messages, navigation, etc.
                                elem_id = elem.get('id', '').lower()
                                elem_classes = ' '.join(elem.get('class', [])).lower()
                                if any(skip in elem_id or skip in elem_classes for skip in ['error', 'warning', 'alert', 'notification', 'empty', 'no-result', 'not-found', 'nav', 'header', 'footer', 'sidebar', 'menu']):
                                    continue
                                
                                # Check all attributes for keyword
                                for attr_name, attr_value in elem.attrs.items():
                                    if isinstance(attr_value, (list, tuple)):
                                        attr_value = ' '.join(str(v) for v in attr_value)
                                    if attr_value and keyword.lower() in str(attr_value).lower():
                                        candidate_content = self._extract_content_from_element(elem)
                                        if candidate_content and len(candidate_content) > 20:
                                            # VALIDATE before accepting
                                            is_valid, reason = self._validate_post_content(elem, min_length=20)
                                            if is_valid:
                                                logger.info(f"[POST-EXTRACT] Found post using attribute match: keyword='{keyword}', attr={attr_name}={attr_value}")
                                                main_post = elem
                                                main_title = self._extract_title_from_element(main_post)
                                                main_author = self._extract_author_from_element(main_post)
                                                main_date = self._extract_date_from_element(main_post)
                                                main_content = candidate_content
                                                break
                                            else:
                                                logger.debug(f"[POST-EXTRACT] Rejected candidate in Strategy 3d: {reason}")
                                if main_post:
                                    break
                            if main_post:
                                break
                        if main_post:
                            break
                
                # Strategy 3e: Generic loose matching - check element name, classes, ids, and attributes
                if not main_post:
                    all_elements = soup.find_all(['div', 'section', 'article', 'main'])
                    for elem in all_elements:
                        # Skip navigation, headers, footers
                        elem_id = elem.get('id', '').lower()
                        elem_classes = ' '.join(elem.get('class', [])).lower()
                        if any(skip in elem_id or skip in elem_classes for skip in ['nav', 'header', 'footer', 'sidebar', 'menu', 'ad', 'advertisement', 'error', 'warning', 'alert', 'notification', 'empty', 'no-result', 'not-found']):
                            continue
                        
                        # Check tag name
                        tag_name = elem.name.lower() if elem.name else ''
                        
                        # Check all attributes
                        attrs_text = ' '.join([f"{k}={v}" for k, v in elem.attrs.items() if isinstance(v, str)]).lower()
                        
                        # Check if any keyword appears in tag name, classes, id, or attributes
                        element_text = f"{tag_name} {elem_classes} {elem_id} {attrs_text}"
                        if any(kw in element_text for kw in keywords):
                            candidate_content = self._extract_content_from_element(elem)
                            if candidate_content and len(candidate_content) > 20:
                                logger.info(f"[POST-EXTRACT] Found post using generic loose match: tag={tag_name}, classes={elem.get('class', [])}")
                                main_post = elem
                                main_title = self._extract_title_from_element(main_post)
                                main_author = self._extract_author_from_element(main_post)
                                main_date = self._extract_date_from_element(main_post)
                                main_content = candidate_content
                                break
            
            # Strategy 4: Generic fallback - try to find ANY substantial content container on detail pages
            if not main_post:
                logger.info(f"[POST-EXTRACT] Trying generic fallback strategies...")
                # Look for main content areas
                main_content_selectors = [
                    'main', 'main article', '[role="main"]', '[role="article"]',
                    '.content', '.main-content', '.entry-content', '.post-content',
                    '#content', '#main', '#post', '#article'
                ]
                for selector in main_content_selectors:
                    containers = soup.select(selector)
                    if containers:
                        container = containers[0]
                        candidate_content = self._extract_content_from_element(container)
                        if candidate_content and len(candidate_content) > 50:  # Need substantial content for fallback
                            logger.info(f"[POST-EXTRACT] Found content using fallback selector: {selector}, content_len={len(candidate_content)}")
                            main_post = container
                            main_title = self._extract_title_from_element(main_post)
                            main_author = self._extract_author_from_element(main_post)
                            main_date = self._extract_date_from_element(main_post)
                            main_content = candidate_content
                            break
                
                # Last resort: find largest text container (likely the main post)
                if not main_post:
                    # Find all divs and sections, get the one with most text
                    all_containers = soup.find_all(['div', 'section', 'article'])
                    best_container = None
                    max_text_length = 0
                    for container in all_containers:
                        # Skip navigation, headers, footers
                        container_id = container.get('id', '').lower()
                        container_class = ' '.join(container.get('class', [])).lower()
                        if any(skip in container_id or skip in container_class for skip in ['nav', 'header', 'footer', 'sidebar', 'menu', 'ad', 'advertisement']):
                            continue
                        text = container.get_text(strip=True)
                        if len(text) > max_text_length and len(text) > 100:  # Must have substantial content
                            max_text_length = len(text)
                            best_container = container
                    
                    if best_container:
                        logger.info(f"[POST-EXTRACT] Using largest text container as fallback, content_len={max_text_length}")
                        main_post = best_container
                        main_title = self._extract_title_from_element(main_post)
                        main_author = self._extract_author_from_element(main_post)
                        main_date = self._extract_date_from_element(main_post)
                        main_content = self._extract_content_from_element(main_post)
            
            if not main_post:
                # Log diagnostic information about what was found
                logger.info(f"[POST-EXTRACT] Could not find main post element on {post_url}")
                logger.info(f"[POST-EXTRACT] Diagnostic: Found {len(soup.select('div'))} divs, {len(soup.select('article'))} articles, {len(soup.select('section'))} sections")
                # Use variables to avoid backslashes in f-string expressions
                post_selector = '[class*="post"]'
                message_selector = '[class*="message"]'
                content_selector = '[class*="content"]'
                logger.info(f"[POST-EXTRACT] Diagnostic: Found {len(soup.select(post_selector))} elements with 'post' in class")
                logger.info(f"[POST-EXTRACT] Diagnostic: Found {len(soup.select(message_selector))} elements with 'message' in class")
                logger.info(f"[POST-EXTRACT] Diagnostic: Found {len(soup.select(content_selector))} elements with 'content' in class")
                
                # Log to tracer
                self.tracer.log_attempt(
                    url=post_url,
                    page_type="detail",
                    strategy_used=None,
                    success=False,
                    failure_reason="No main post found after all strategies",
                    content_length=0,
                    strategy_order=strategy_order,
                    strategy_results=strategy_results,
                    html_sample=str(soup)[:10000]
                )
                
                # Try to find the largest text block as absolute last resort
                all_text_elements = soup.find_all(['div', 'section', 'article', 'main'])
                if all_text_elements:
                    largest = max(all_text_elements, key=lambda e: len(e.get_text(strip=True)))
                    largest_text = largest.get_text(strip=True)
                    logger.info(f"[POST-EXTRACT] Diagnostic: Largest text block has {len(largest_text)} characters, tag={largest.name}, class={largest.get('class', [])}")
                return None
            
            # Find all replies
            replies_content = []
            
            # Strategy 0: Reddit comments (<shreddit-comment> elements, recursively)
            if shreddit_posts:
                # Extract comments recursively from the main post container or entire page
                def extract_reddit_comments(container, depth=0, max_depth=10):
                    """Recursively extract Reddit comments."""
                    if depth > max_depth:
                        return []
                    comments = []
                    # Find all shreddit-comment elements within this container
                    comment_elements = container.select('shreddit-comment')
                    for comment_elem in comment_elements:
                        # Extract comment content from [slot="text"] or .md
                        comment_content_elem = comment_elem.select_one('[slot="text"], .md, [class*="md"]')
                        if comment_content_elem:
                            comment_text = comment_content_elem.get_text(strip=True)
                            if comment_text and len(comment_text) > 10:  # Minimum length
                                comments.append(comment_text)
                        else:
                            # Fallback: get text, excluding nested comments
                            comment_clone = BeautifulSoup(str(comment_elem), 'html.parser')
                            for nested in comment_clone.select('shreddit-comment'):
                                nested.decompose()
                            comment_text = comment_clone.get_text(strip=True)
                            if comment_text and len(comment_text) > 10:
                                comments.append(comment_text)
                        
                        # Recursively extract nested comments
                        nested_comments = extract_reddit_comments(comment_elem, depth + 1, max_depth)
                        comments.extend(nested_comments)
                    return comments
                
                # Extract comments from the entire page (Reddit comments are often outside the main post element)
                reddit_comments = extract_reddit_comments(soup)
                replies_content.extend(reddit_comments)
                logger.info(f"[POST-EXTRACT] Extracted {len(reddit_comments)} Reddit comments")
            
            # Strategy 1: vBulletin-style thread posts (skip first one if it's the main post)
            thread_posts_for_replies = soup.select('div.threadpost, div[id*="edit"][id*="post"]')
            if thread_posts_for_replies and len(thread_posts_for_replies) > 1:
                # Skip first one if it's the main post we already extracted
                start_idx = 1 if (main_post and main_post in thread_posts_for_replies) else 0
                for reply_post in thread_posts_for_replies[start_idx:]:
                    reply_content = self._extract_content_from_element(reply_post)
                    if reply_content:
                        replies_content.append(reply_content)
            
            # Strategy 2: Reply patterns (with nested structure support)
            def extract_nested_replies(container, depth=0, max_depth=10, exclude_element=None):
                """Recursively extract replies from nested structures."""
                if depth > max_depth:
                    return []
                extracted = []
                
                # Find direct reply/comment children
                reply_patterns = [
                    '.reply', '.comment', '.reply-content', '.comment-content',
                    '.reply-body', '.comment-body', '.post .reply', '.message .reply',
                    '[class*="reply"]', '[class*="comment"]'
                ]
                
                for pattern in reply_patterns:
                    replies = container.select(pattern)
                    for reply in replies:
                        # Skip if this is the main post or excluded element
                        if reply == exclude_element or reply == main_post:
                            continue
                        # Skip if already extracted (check by content)
                        reply_content = self._extract_content_from_element(reply)
                        if reply_content and len(reply_content) > 10:  # Minimum length
                            # Check if similar content already extracted
                            is_duplicate = any(
                                abs(len(reply_content) - len(existing)) < 5 and 
                                reply_content[:50] == existing[:50] 
                                for existing in extracted
                            )
                            if not is_duplicate:
                                extracted.append(reply_content)
                                # Recursively extract nested replies within this reply
                                nested = extract_nested_replies(reply, depth + 1, max_depth, exclude_element)
                                extracted.extend(nested)
                
                return extracted
            
            # Extract replies from the entire page (not just main_post container)
            nested_replies = extract_nested_replies(soup, exclude_element=main_post)
            replies_content.extend(nested_replies)
            logger.info(f"[POST-EXTRACT] Extracted {len(nested_replies)} replies (including nested)")
            
            # Combine main post content with all replies
            all_content_parts = []
            if main_title:
                all_content_parts.append(main_title)
            if main_content:
                all_content_parts.append(main_content)
            all_content_parts.extend(replies_content)
            combined_content = " ".join(all_content_parts)
            
            # Extract raw HTML of the main post element (and replies if available)
            raw_html = None
            if main_post:
                # Get the HTML of the main post element
                raw_html = str(main_post)
                
                # If there are replies, try to find and include their HTML
                if replies_content:
                    reply_elements = []
                    # Try multiple strategies to find reply containers
                    # Strategy 1: Find reply/comment containers (excluding main post)
                    reply_containers = soup.select('.reply, .comment, .threadpost, [class*="reply"], [class*="comment"]')
                    seen_containers = set()
                    for container in reply_containers:
                        # Skip the main post if it's in the list
                        if container == main_post:
                            continue
                            
                        # Check if this container is a child of main_post (if so, already included in main_post HTML)
                        is_child_of_main = False
                        try:
                            parent = container.parent
                            while parent and parent != soup:
                                if parent == main_post:
                                    is_child_of_main = True
                                    break
                                parent = getattr(parent, 'parent', None)
                                if not parent:
                                    break
                        except:
                            pass
                        
                        if not is_child_of_main:
                            container_id = id(container)
                            if container_id not in seen_containers:
                                seen_containers.add(container_id)
                                reply_elements.append(str(container))
                    
                    # Strategy 2: For Reddit, find shreddit-comment elements
                    if shreddit_posts:
                        comment_elements = soup.select('shreddit-comment')
                        for comment_elem in comment_elements:
                            comment_id = id(comment_elem)
                            if comment_id not in seen_containers:
                                seen_containers.add(comment_id)
                                reply_elements.append(str(comment_elem))
                    
                    # Strategy 3: Find nested reply structures
                    if main_post.parent:
                        # Look for sibling reply containers
                        parent = main_post.parent
                        for sibling in parent.find_all(['div', 'article', 'section'], recursive=False):
                            if sibling == main_post:
                                continue
                            sibling_id = id(sibling)
                            if sibling_id not in seen_containers:
                                # Check if it looks like a reply
                                sibling_classes = str(sibling.get('class', [])).lower()
                                if any(keyword in sibling_classes for keyword in ['reply', 'comment', 'response']):
                                    seen_containers.add(sibling_id)
                                    reply_elements.append(str(sibling))
                    
                    if reply_elements:
                        raw_html += "\n<!-- REPLIES -->\n" + "\n".join(reply_elements)
                
                # Limit raw_html size to prevent unbounded memory growth (max 500KB)
                max_raw_html_size = 500 * 1024  # 500KB
                if raw_html and len(raw_html) > max_raw_html_size:
                    raw_html = raw_html[:max_raw_html_size] + "\n<!-- TRUNCATED -->\n"
                    logger.debug(f"[POST-EXTRACT] Truncated raw_html from {len(str(main_post))} to {max_raw_html_size} chars")
            
            # Log successful extraction to tracer
            extraction_duration = time.time() - extraction_start
            successful_strategy = strategy_order[-1] if strategy_order else "unknown"
            
            self.tracer.log_attempt(
                url=post_url,
                page_type="detail",
                strategy_used=successful_strategy,
                success=True,
                content_length=len(combined_content),
                title_found=bool(main_title),
                author_found=bool(main_author),
                date_found=bool(main_date),
                strategy_order=strategy_order,
                strategy_results=strategy_results,
                html_sample=str(main_post)[:10000] if main_post else None
            )
            
            # Update strategy metrics
            self._update_strategy_metrics(site_id, successful_strategy, True, len(combined_content), extraction_duration)
            
            return {
                'title': main_title,
                'content': combined_content,
                'author': main_author,
                'date': main_date,
                'post_url': post_url,
                'raw_html': raw_html
            }
            
        except Exception as e:
            logger.error(f"[POST-EXTRACT] Error extracting post from {post_url}: {e}")
            # Log exception to tracer
            self.tracer.log_attempt(
                url=post_url,
                page_type="detail",
                strategy_used=None,
                success=False,
                failure_reason=f"Exception: {str(e)}",
                strategy_order=strategy_order,
                strategy_results=strategy_results
            )
            return None
    
    def _validate_post_content(self, element, min_length: int = 50) -> tuple[bool, str]:
        """
        Validate that an element is actually a post, not an error/nav/menu.
        Returns (is_valid, reason).
        """
        if not element:
            return False, "Element is None"
        
        text = element.get_text(strip=True)
        
        # Check minimum length
        if len(text) < min_length:
            return False, f"Content too short: {len(text)} chars"
        
        # Check for error messages
        error_keywords = ['error', 'not found', '404', '500', 'forbidden', 'unauthorized', 
                         'page not found', 'access denied', 'try again', 'something went wrong']
        text_lower = text.lower()
        for keyword in error_keywords:
            if keyword in text_lower:
                return False, f"Contains error keyword: {keyword}"
        
        # Check for navigation/menu (too many links relative to text)
        links = element.find_all('a')
        link_count = len(links)
        text_length = len(text)
        if text_length > 0 and link_count > text_length / 50:  # More than 1 link per 50 chars
            return False, f"Too many links ({link_count}) relative to text ({text_length} chars)"
        
        # Check for post-like structure (has title, content, or author indicators)
        has_title = bool(element.find(['h1', 'h2', 'h3', 'h4', lambda tag: tag.name and ('title' in tag.get('class', []) or 'heading' in tag.get('class', []))]))
        has_author = bool(element.find(['[class*="author"]', '[class*="user"]', '[class*="poster"]']))
        has_date = bool(element.find(['time', '[class*="date"]', '[class*="time"]']))
        
        if not (has_title or has_author or has_date):
            # Still might be valid if it has substantial content
            if len(text) < 200:
                return False, "No post structure indicators and content too short"
        
        return True, "Valid post content"
    
    def _extract_title_from_element(self, element) -> Optional[str]:
        """Extract title from a post element."""
        # Similar logic to _create_post_candidate title extraction
        title = None
        
        # Look for links with discussion/post keywords
        discussion_links = element.find_all('a', href=True)
        for link in discussion_links:
            href = link.get('href', '').lower()
            link_text = link.get_text(strip=True)
            if any(kw in href for kw in ['discussion', 'thread', 'post', 'topic', 'comment', '/t/', '/p/']) and len(link_text) > 10:
                title = link_text
                break
        
        # Check for heading elements
        if not title:
            for heading in element.find_all(['h1', 'h2', 'h3', 'h4']):
                title = heading.get_text(strip=True)
                if title and len(title) > 5:
                    break
        
        # Check for title classes
        if not title:
            title_elem = element.select_one('.title, .subject, .thread-title, .post-title, .topic-title')
            if title_elem:
                title = title_elem.get_text(strip=True)
        
        return title
    
    def _extract_author_from_element(self, element) -> Optional[str]:
        """Extract author from a post element."""
        # Similar logic to _create_post_candidate author extraction
        author = None
        
        author_selectors = [
            '.author', '.username', '.byline', '.poster', '.post-author', '.message-author',
            '.user-name', '.member-name', '.user-link', '.author-name'
        ]
        for selector in author_selectors:
            author_elem = element.select_one(selector)
            if author_elem:
                author = author_elem.get_text(strip=True)
                if author:
                    break
        
        # Check data attributes
        if not author:
            author = element.get('data-author') or element.get('data-username') or element.get('data-poster')
        
        return author
    
    def _extract_date_from_element(self, element) -> Optional[datetime]:
        """Extract date from a post element."""
        # Similar logic to _create_post_candidate date extraction
        from datetime import datetime
        try:
            from dateutil import parser as date_parser
            has_dateutil = True
        except ImportError:
            has_dateutil = False
            date_parser = None
        
        date = None
        
        if has_dateutil:
            # Try <time> element with datetime attribute
            time_elem = element.find('time', datetime=True)
            if time_elem:
                try:
                    date = date_parser.parse(time_elem['datetime'])
                except (ValueError, KeyError):
                    pass
            
            # Try common date class selectors
            if not date:
                date_selectors = ['.date', '.published', '.post-date', '.timestamp', '.created', '.posted']
                for date_selector in date_selectors:
                    date_elem = element.select_one(date_selector)
                    if date_elem:
                        date_text = date_elem.get_text(strip=True)
                        if date_text:
                            try:
                                date = date_parser.parse(date_text)
                                break
                            except (ValueError, TypeError):
                                continue
        else:
            # Fallback: try ISO format
            time_elem = element.find('time', datetime=True)
            if time_elem:
                try:
                    date_str = time_elem['datetime']
                    if 'T' in date_str:
                        date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    else:
                        date = datetime.strptime(date_str, '%Y-%m-%d')
                except (ValueError, KeyError):
                    pass
        
        return date
    
    def _extract_content_from_element(self, element) -> str:
        """Extract content text from a post element."""
        # Clone to avoid modifying original
        content_element = BeautifulSoup(str(element), 'html.parser')
        
        # Remove nested replies/comments/quotes
        for nested in content_element.select('.reply, .comment, .quote, blockquote, .nested-reply'):
            nested.decompose()
        
        # Try specific content selectors
        content_selectors = [
            'div[id*="post_message"]',
            '.post-content', '.message-content', '.post-body', '.message-body',
            '.post-text', '.message-text', '.content', '.text', '.body',
            'td[id*="td_post"]'
        ]
        
        for selector in content_selectors:
            content_elem = content_element.select_one(selector)
            if content_elem:
                # Remove title/header areas
                for title_div in content_elem.select('div.smallfont strong, div.smallfont:has(strong)'):
                    title_div.decompose()
                content = content_elem.get_text(strip=True, separator=' ')
                if content and len(content) > 20:
                    return content
        
        # Fallback to all text
        return content_element.get_text(strip=True, separator=' ')
    
    def _contains_diabetes_keyword_in_content(self, post_data: dict) -> bool:
        """Check if post title, content, or replies contain diabetes keywords."""
        diabetes_keywords = [
            'diabetes', 'diabetic', 'insulin', 'blood sugar', 'glucose',
            'type 1 diabetes', 'type 2 diabetes', 'gestational diabetes',
            'diabetes mellitus', 'hyperglycemia', 'hypoglycemia',
            'a1c', 'hba1c', 'blood glucose', 'sugar levels'
        ]
        
        # Check title
        if post_data.get('title'):
            title_lower = post_data['title'].lower()
            for keyword in diabetes_keywords:
                if keyword in title_lower:
                    return True
        
        # Check content (includes main post + all replies)
        if post_data.get('content'):
            content_lower = post_data['content'].lower()
            for keyword in diabetes_keywords:
                if keyword in content_lower:
                    return True
        
        return False
    
    def _create_post_candidate_from_data(self, post_data: dict, post_url: str, page_url: str, site_id: str) -> CandidatePost:
        """Create CandidatePost from extracted post data."""
        return CandidatePost(
            page_url=page_url,
            post_url=post_url,
            selector_hint='navigated-post',
            site_id=site_id,
            title=post_data.get('title'),
            content=post_data.get('content'),
            author=post_data.get('author'),
            date=post_data.get('date'),
            raw_html=post_data.get('raw_html')
        )
    
    async def mine_posts_for_diabetes(self, html: str, base_url: str, site_id: str) -> List[CandidatePost]:
        """Mine posts from HTML content and check for diabetes mentions."""
        try:
            # Create debug output directory (local, persists after crawl)
            debug_dir = Path("backend/crawl_output/debug")
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Write HTML sample to file
            safe_url = base_url.replace("https://", "").replace("http://", "").replace("/", "_")[:100]
            html_file = debug_dir / f"html_{site_id}_{safe_url}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html[:50000])  # First 50k chars
            logger.info(f"[POST-MINE] Wrote HTML sample to {html_file}")
            
            # Clear strategy metrics for this site to prevent unbounded growth
            site_metrics_keys = [k for k in self._strategy_metrics.keys() if k.startswith(f"{site_id}:")]
            for key in site_metrics_keys:
                del self._strategy_metrics[key]
            if site_metrics_keys:
                logger.debug(f"[POST-MINE] Cleared {len(site_metrics_keys)} strategy metrics entries for site {site_id}")
            
            soup = BeautifulSoup(html, 'html.parser')
            candidates = []
            
            # Detect page type (pass URL for URL-based heuristics)
            page_type = self._detect_page_type(soup, base_url)
            logger.info(f"[POST-MINE] Detected page type: {page_type} for {base_url} (HTML size: {len(html)} chars)")
            
            # Log some diagnostic info about the page
            all_links = soup.find_all('a', href=True)
            logger.info(f"[POST-MINE] Page has {len(all_links)} total links")
            
            # Check for common post indicators
            has_threadbits = len(soup.select('tbody[id*="threadbits"]')) > 0
            has_activity_rows = len(soup.select('.ch-activity-simple-row, .activity-simple-row')) > 0
            has_post_classes = len(soup.find_all(True, class_=re.compile(r'post|thread|discussion', re.I))) > 0
            logger.info(f"[POST-MINE] Page indicators: threadbits={has_threadbits}, activity_rows={has_activity_rows}, post_classes={has_post_classes}")
            
            # Write page analysis to file
            analysis_file = debug_dir / f"analysis_{site_id}_{safe_url}.json"
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'url': base_url,
                    'site_id': site_id,
                    'page_type': page_type,
                    'html_size': len(html),
                    'total_links': len(all_links),
                    'indicators': {
                        'threadbits': has_threadbits,
                        'activity_rows': has_activity_rows,
                        'post_classes': has_post_classes
                    }
                }, f, indent=2)
            logger.info(f"[POST-MINE] Wrote page analysis to {analysis_file}")

            if page_type == 'listing':
                # Phase 1: Find post items with links
                post_items = self._find_post_listing_items(soup)
                
                if not post_items:
                    logger.info(f"[POST-MINE] No post listing items found on {base_url}")
                    return []
                
                # Phase 2: Extract links, prioritize by keywords
                post_links = self._extract_post_links(post_items, base_url)
                
                if not post_links:
                    logger.info(f"[POST-MINE] No post links extracted from {base_url} (found {len(post_items)} items but no links)")
                    return []
                
                # Sort: posts with keywords in title first (True values first)
                post_links.sort(key=lambda x: not x[2])  # has_keywords_in_title (True first)
                
                # Log prioritization
                prioritized_count = sum(1 for _, _, has_kw in post_links if has_kw)
                logger.info(f"[POST-MINE] Prioritized {prioritized_count}/{len(post_links)} posts with keywords in title")
                
                # Phase 3: Navigate to links concurrently (prioritized order, with limits)
                # Limit number of links to prevent resource exhaustion
                max_links = self.config.nc_max_post_links_per_page
                limited_links = post_links[:max_links]
                if len(post_links) > max_links:
                    logger.info(f"[POST-MINE] Limiting navigation to {max_links}/{len(post_links)} post links to prevent resource exhaustion")
                
                # Filter out already visited URLs
                new_links = []
                for post_item, link_url, has_keywords in limited_links:
                    # Normalize URL (remove fragments, trailing slashes)
                    normalized_url = link_url.split('#')[0].rstrip('/')
                    if normalized_url not in self._visited_urls:
                        # Limit _visited_urls size to prevent unbounded memory growth
                        if len(self._visited_urls) >= self._visited_urls_max_size:
                            # Remove oldest entries (convert to list, remove first 1000, convert back)
                            urls_list = list(self._visited_urls)
                            self._visited_urls = set(urls_list[1000:])
                            logger.debug(f"[POST-MINE] Trimmed _visited_urls from {len(urls_list)} to {len(self._visited_urls)} entries")
                        self._visited_urls.add(normalized_url)
                        new_links.append((post_item, link_url, has_keywords))
                    else:
                        logger.debug(f"[POST-MINE] Skipping already visited URL: {link_url}")
                
                if not new_links:
                    logger.info(f"[POST-MINE] All {len(limited_links)} post links were already visited")
                    return []
                
                logger.info(f"[POST-MINE] Navigating to {len(new_links)} new post links from {base_url} (skipped {len(limited_links) - len(new_links)} duplicates)")
                tasks = []
                for post_item, link_url, has_keywords in new_links:
                    task = asyncio.create_task(
                        self._navigate_and_read_post(link_url, base_url, site_id)
                    )
                    tasks.append(task)
                
                # Process concurrently (respects semaphore)
                results = await asyncio.gather(*tasks, return_exceptions=True)
                successful = 0
                for result in results:
                    if isinstance(result, CandidatePost):
                        candidates.append(result)
                        successful += 1
                    elif isinstance(result, Exception):
                        logger.info(f"[POST-MINE] Error processing post link: {result}")
                logger.info(f"[POST-MINE] Successfully navigated and extracted {successful}/{len(new_links)} posts")
                
                # Clear BeautifulSoup from memory after processing listing page
                soup.decompose()
                del html
            
            else:
                # Already on detail page - extract post + replies, check keywords
                logger.info(f"[POST-MINE] Processing as detail page: {base_url}")
                post_data = await self._extract_full_post_and_replies(soup, base_url, site_id)
                
                # Clear BeautifulSoup from memory after extraction
                soup.decompose()
                del html
                
                if post_data:
                    logger.info(f"[POST-MINE] Extracted post data from detail page: title={bool(post_data.get('title'))}, content_len={len(post_data.get('content', ''))}")
                    # Return all posts (not just keyword-filtered)
                    has_keywords = self._contains_diabetes_keyword_in_content(post_data)
                    if has_keywords:
                        logger.info(f"[POST-MINE] Detail page post contains diabetes keywords")
                    else:
                        logger.info(f"[POST-MINE] Detail page post does NOT contain diabetes keywords (will save to raw-images only)")
                    candidate = self._create_post_candidate_from_data(post_data, base_url, base_url, site_id)
                    candidates.append(candidate)
                    
                    # Clear post_data from memory after creating candidate
                    del post_data
                else:
                    logger.info(f"[POST-MINE] Could not extract post data from detail page: {base_url}")

            # Remove duplicates based on post URL (normalize URLs to remove fragments and trailing slashes)
            seen_urls = set()
            unique_candidates = []
            for candidate in candidates:
                # Normalize URL for duplicate detection
                normalized_url = candidate.post_url.split('#')[0].rstrip('/')
                if normalized_url not in seen_urls:
                    seen_urls.add(normalized_url)
                    unique_candidates.append(candidate)
                else:
                    logger.debug(f"[POST-MINE] Skipping duplicate post: {candidate.post_url} (normalized: {normalized_url})")

            logger.info(f"Found {len(unique_candidates)} unique diabetes-related post candidates from {base_url}")
            
            # Write candidates to file
            if unique_candidates:
                candidates_file = debug_dir / f"candidates_{site_id}_{safe_url}.json"
                with open(candidates_file, 'w', encoding='utf-8') as f:
                    candidates_data = []
                    for candidate in unique_candidates:
                        candidates_data.append({
                            'post_url': candidate.post_url,
                            'page_url': candidate.page_url,
                            'title': candidate.title,
                            'content_preview': candidate.content[:200] if candidate.content else None,
                            'author': candidate.author,
                            'date': candidate.date.isoformat() if candidate.date else None,
                            'site_id': candidate.site_id
                        })
                    json.dump(candidates_data, f, indent=2, default=str)
                logger.info(f"[POST-MINE] Wrote {len(unique_candidates)} candidates to {candidates_file}")
            else:
                # Write why no candidates were found
                no_candidates_file = debug_dir / f"no_candidates_{site_id}_{safe_url}.json"
                with open(no_candidates_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'url': base_url,
                        'site_id': site_id,
                        'page_type': page_type,
                        'reason': 'No candidates found - check logs for details'
                    }, f, indent=2)
                logger.info(f"[POST-MINE] Wrote no-candidates info to {no_candidates_file}")
            
            return unique_candidates

        except Exception as e:
            logger.error(f"Error mining posts from {base_url}: {e}")
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

    def _create_post_candidate(self, post_element, base_url: str, selector: str, site_id: str) -> Optional[CandidatePost]:
        """Create candidate post from HTML element (optimized for forum posts)."""
        try:
            from datetime import datetime
            # Try to import dateutil, fallback to basic parsing if not available
            try:
                from dateutil import parser as date_parser
                has_dateutil = True
            except ImportError:
                has_dateutil = False
                date_parser = None
            
            # Extract post URL - use generic heuristics to find discussion/post links
            post_url = base_url  # Default to page URL
            
            # For individual posts on detail pages, try to find post anchor/permalink
            if post_element.name == 'div' and ('threadpost' in post_element.get('class', []) or 'edit' in post_element.get('id', '')):
                # Extract post ID from element ID (e.g., "edit5524647" -> "5524647")
                elem_id = post_element.get('id', '')
                post_id_match = re.search(r'(\d+)', elem_id)
                if post_id_match:
                    post_id = post_id_match.group(1)
                    # Try to find anchor link with this post ID
                    anchor = post_element.find('a', {'name': f'post{post_id}'})
                    if anchor:
                        # Use page URL with anchor
                        post_url = f"{base_url}#post{post_id}"
            
            # Strategy: Find links that look like discussion/post URLs
            # Look for links with discussion/post-related keywords in href or class
            discussion_keywords = ['discussion', 'thread', 'post', 'topic', 'message', 'comment', '/t/', '/p/', '/comment/']
            
            # Priority 1: Links with discussion/post keywords in class or id
            if post_url == base_url:
                for keyword in discussion_keywords:
                    # Check for links with keyword in class
                    link = post_element.find('a', href=True, class_=re.compile(keyword, re.I))
                    if link:
                        href = link.get('href')
                        if href and any(kw in href.lower() for kw in discussion_keywords):
                            post_url = urljoin(base_url, href)
                            break
                    
                    # Check for links with keyword in id
                    link = post_element.find('a', href=True, id=re.compile(keyword, re.I))
                    if link:
                        href = link.get('href')
                        if href:
                            post_url = urljoin(base_url, href)
                            break
            
            # Priority 2: Check for thread title links (common in table-based forums)
            if post_url == base_url:
                thread_title_link = post_element.find('a', id=re.compile(r'thread_title', re.I))
                if thread_title_link and thread_title_link.get('href'):
                    post_url = urljoin(base_url, thread_title_link['href'])
            
            # Priority 3: Look for links with discussion/post keywords in href
            if post_url == base_url:
                all_links = post_element.find_all('a', href=True)
                for link in all_links:
                    href = link.get('href', '').lower()
                    # Check if href contains discussion/post keywords
                    if any(keyword in href for keyword in discussion_keywords):
                        # Prefer links that look like individual posts (have IDs or comment paths)
                        if '/comment/' in href or re.search(r'/\d+', href) or any(kw in href for kw in ['/thread', '/post', '/topic']):
                            post_url = urljoin(base_url, link['href'])
                            break
            
            # Priority 4: Look for links in title areas
            if post_url == base_url:
                title_area = post_element.find(['h1', 'h2', 'h3', 'h4', 'div'], class_=re.compile(r'title|subject|discussion', re.I))
                if title_area:
                    link = title_area.find('a', href=True)
                    if link:
                        post_url = urljoin(base_url, link['href'])
            
            # Priority 5: For table rows, look in the thread title cell
            if post_url == base_url and post_element.name == 'tr':
                title_cell = post_element.find('td', id=re.compile(r'threadtitle|td_threadtitle', re.I))
                if title_cell:
                    link = title_cell.find('a', href=True)
                    if link:
                        post_url = urljoin(base_url, link['href'])
            
            # Priority 6: Fallback to first link that looks like a discussion URL
            if post_url == base_url:
                link = post_element.find('a', href=True)
                if link:
                    href = link['href']
                    # Prefer links that look like thread/post URLs
                    if any(keyword in href.lower() for keyword in discussion_keywords):
                        post_url = urljoin(base_url, href)
                    else:
                        post_url = urljoin(base_url, href)

            # Extract title - use generic heuristics
            title = None
            
            # Strategy 1: Look for links with discussion/post keywords (often contain titles)
            discussion_links = post_element.find_all('a', href=True)
            for link in discussion_links:
                href = link.get('href', '').lower()
                link_text = link.get_text(strip=True)
                # If link looks like a discussion/post link and has meaningful text
                if any(kw in href for kw in ['discussion', 'thread', 'post', 'topic', 'comment', '/t/', '/p/']) and len(link_text) > 10:
                    title = link_text
                    break
            
            # Strategy 2: Check for thread title links (common in table-based forums)
            if not title:
                thread_title_link = post_element.find('a', id=re.compile(r'thread_title', re.I))
                if thread_title_link:
                    title = thread_title_link.get_text(strip=True)
            
            # Strategy 3: Check for discussion-title class (Mayo Clinic style)
            if not title:
                discussion_title = post_element.find('a', class_=re.compile(r'discussion.*title|title.*discussion', re.I))
                if discussion_title:
                    title = discussion_title.get_text(strip=True)
            
            # Strategy 4: Check for vBulletin-style post titles (in div.smallfont strong within post content cell)
            if not title:
                post_content_cell = post_element.find('td', id=re.compile(r'td_post', re.I))
                if post_content_cell:
                    title_elem = post_content_cell.find('div', class_='smallfont')
                    if title_elem:
                        strong_elem = title_elem.find('strong')
                        if strong_elem:
                            title = strong_elem.get_text(strip=True)
            
            # Strategy 5: Look in heading elements and title classes
            if not title:
                title_selectors = [
                    'h1', 'h2', 'h3', 'h4', 
                    '.title', '.subject', '.thread-title', '.post-title', '.topic-title',
                    '.message-title', '.entry-title', 'a.title', 'a.subject',
                    'div.smallfont strong',  # vBulletin-style titles
                    '[class*="title"]', '[class*="subject"]'  # Generic title classes
                ]
                for title_selector in title_selectors:
                    title_elem = post_element.select_one(title_selector)
                    if title_elem:
                        title = title_elem.get_text(strip=True)
                        if title and len(title) > 5:  # Ensure meaningful title
                            break
            
            # Strategy 6: For table rows, check the thread title cell
            if not title and post_element.name == 'tr':
                title_cell = post_element.find('td', id=re.compile(r'threadtitle|td_threadtitle', re.I))
                if title_cell:
                    link = title_cell.find('a', href=True)
                    if link:
                        title = link.get_text(strip=True)
            
            # Strategy 7: Look in discussion-info divs (Mayo Clinic style)
            if not title:
                discussion_info = post_element.find('div', class_=re.compile(r'discussion.*info|info.*discussion', re.I))
                if discussion_info:
                    link = discussion_info.find('a', href=True)
                    if link:
                        title = link.get_text(strip=True)

            # Extract content - exclude nested replies/comments to get just this post's content
            # Clone element to avoid modifying original
            content_element = BeautifulSoup(str(post_element), 'html.parser')
            
            # Remove nested replies/comments/quotes from content
            for nested in content_element.select('.reply, .comment, .quote, blockquote, .nested-reply'):
                nested.decompose()
            
            # For table rows (listing pages), content is often just the title
            # The actual post content is on the individual thread page
            if post_element.name == 'tr':
                # Try to get content from thread title cell
                title_cell = content_element.find('td', id=re.compile(r'threadtitle|td_threadtitle', re.I))
                if title_cell:
                    # Get all text from the title cell (includes title and author info)
                    content = title_cell.get_text(strip=True, separator=' ')
                else:
                    # Fallback to all text in row
                    content = content_element.get_text(strip=True, separator=' ')
            else:
                # Strategy 1: Check for vBulletin-style post message divs
                post_message = content_element.select_one('div[id*="post_message"]')
                if post_message:
                    content = post_message.get_text(strip=True, separator=' ')
                else:
                    # Strategy 2: Look for content in elements with post/message/content keywords
                    content = None
                    content_keywords = ['post', 'message', 'content', 'body', 'text', 'discussion']
                    
                    # Try specific content selectors first
                    content_selectors = [
                        '.post-content', '.message-content', '.post-body', '.message-body',
                        '.post-text', '.message-text', '.content', '.text', '.body',
                        'td[id*="td_post"]',  # vBulletin post content cell
                        '[class*="post"][class*="content"]', '[class*="message"][class*="content"]',
                        '[class*="discussion"][class*="content"]'
                    ]
                    for content_selector in content_selectors:
                        try:
                            content_elem = content_element.select_one(content_selector)
                            if content_elem:
                                # For table cells, get text but exclude title/header areas
                                if content_elem.name == 'td':
                                    # Exclude title divs
                                    for title_div in content_elem.select('div.smallfont strong, div.smallfont:has(strong)'):
                                        title_div.decompose()
                                content = content_elem.get_text(strip=True, separator=' ')
                                if content and len(content) > 20:  # Ensure meaningful content
                                    break
                        except Exception:
                            continue
                    
                    # Strategy 3: Generic search for divs with content-like classes
                    if not content:
                        for keyword in content_keywords:
                            content_divs = content_element.find_all('div', class_=re.compile(keyword, re.I))
                            for div in content_divs:
                                text = div.get_text(strip=True, separator=' ')
                                # Skip if it's too short or looks like metadata
                                if len(text) > 50 and not re.match(r'^\d+$', text.strip()):  # Not just a number
                                    content = text
                                    break
                            if content:
                                break
                    
                    # Fallback to all text if no content area found
                    if not content:
                        content = content_element.get_text(strip=True, separator=' ')

            # Extract author - look for forum-specific author patterns
            author = None
            
            # Check for vBulletin-style post menu divs first
            post_menu = post_element.select_one('div[id*="postmenu"]')
            if post_menu:
                author = post_menu.get_text(strip=True)
            
            if not author:
                author_selectors = [
                    '.author', '.username', '.byline', '.poster', '.post-author', '.message-author',
                    '.user-name', '.member-name', '.user-link', '.author-name',
                    '[class*="author"]', '[class*="user"]', '[class*="poster"]'
                ]
                for author_selector in author_selectors:
                    author_elem = post_element.select_one(author_selector)
                    if author_elem:
                        author = author_elem.get_text(strip=True)
                        if author:
                            break
            
            # For table-based forums, check div.smallfont (common pattern)
            if not author and post_element.name in ['tr', 'td']:
                # Look for smallfont divs which often contain author info
                smallfont_divs = post_element.find_all('div', class_='smallfont')
                for div in smallfont_divs:
                    text = div.get_text(strip=True)
                    # Skip if it looks like date/time info
                    if not re.search(r'\d{1,2}-\d{1,2}-\d{4}|\d{1,2}:\d{2}', text):
                        # Check if it contains "by" keyword (common pattern)
                        if 'by' in text.lower():
                            parts = text.split('by', 1)
                            if len(parts) > 1:
                                author = parts[1].strip()
                                break
                        else:
                            # Might be just the author name
                            author = text.strip()
                            if author and len(author) < 50:  # Reasonable author name length
                                break
            
            # For threadpost divs, check the post menu area
            if not author and post_element.name == 'div' and ('threadpost' in post_element.get('class', []) or 'edit' in post_element.get('id', '')):
                # Look in the table structure for author info
                post_table = post_element.find('table', id=re.compile(r'post\d+', re.I))
                if post_table:
                    post_menu_cell = post_table.find('td', class_='alt2')
                    if post_menu_cell:
                        post_menu_div = post_menu_cell.find('div', id=re.compile(r'postmenu', re.I))
                        if post_menu_div:
                            author = post_menu_div.get_text(strip=True)
            
            # If still no author, try data attributes
            if not author:
                author = post_element.get('data-author') or post_element.get('data-username') or post_element.get('data-poster')

            # Extract date from various HTML patterns
            date = None
            
            # Check for vBulletin-style dates in thead cells (e.g., "08-08-2025, 04:52 PM")
            if not date:
                thead_cell = post_element.find('td', class_='thead')
                if thead_cell:
                    date_text = thead_cell.get_text(strip=True)
                    # Look for date patterns like "08-08-2025, 04:52 PM" or "MM-DD-YYYY, HH:MM AM/PM"
                    date_match = re.search(r'(\d{1,2}-\d{1,2}-\d{4}[,\s]+\d{1,2}:\d{2}\s*(?:AM|PM)?)', date_text)
                    if date_match:
                        date_str = date_match.group(1)
                        try:
                            if has_dateutil:
                                date = date_parser.parse(date_str)
                            else:
                                # Try to parse manually: "MM-DD-YYYY, HH:MM AM/PM"
                                date_str_clean = date_str.replace(',', '').strip()
                                if 'AM' in date_str_clean or 'PM' in date_str_clean:
                                    date = datetime.strptime(date_str_clean, '%m-%d-%Y %I:%M %p')
                                else:
                                    date = datetime.strptime(date_str_clean, '%m-%d-%Y %H:%M')
                        except (ValueError, TypeError):
                            pass
            
            if has_dateutil:
                # Try <time> element with datetime attribute
                if not date:
                    time_elem = post_element.find('time', datetime=True)
                    if time_elem:
                        try:
                            date = date_parser.parse(time_elem['datetime'])
                        except (ValueError, KeyError):
                            pass
                
                # Try <time> element with text content
                if not date:
                    time_elem = post_element.find('time')
                    if time_elem:
                        time_text = time_elem.get_text(strip=True)
                        if time_text:
                            try:
                                date = date_parser.parse(time_text)
                            except (ValueError, TypeError):
                                pass
                
                # Try common date class selectors
                if not date:
                    date_selectors = ['.date', '.published', '.post-date', '.timestamp', '.created', '.posted']
                    for date_selector in date_selectors:
                        date_elem = post_element.select_one(date_selector)
                        if date_elem:
                            date_text = date_elem.get_text(strip=True)
                            if date_text:
                                try:
                                    date = date_parser.parse(date_text)
                                    break
                                except (ValueError, TypeError):
                                    continue
                
                # Try datetime attribute on the element itself
                if not date and post_element.get('datetime'):
                    try:
                        date = date_parser.parse(post_element['datetime'])
                    except (ValueError, KeyError):
                        pass
            else:
                # Fallback: try to parse ISO format dates manually
                if not date:
                    time_elem = post_element.find('time', datetime=True)
                    if time_elem:
                        try:
                            date_str = time_elem['datetime']
                            # Try ISO format: YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS
                            if 'T' in date_str:
                                date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            else:
                                date = datetime.strptime(date_str, '%Y-%m-%d')
                        except (ValueError, KeyError):
                            pass

            # Allow posts with minimal or no content (listing pages often have just titles)
            # Only reject if content is completely None (shouldn't happen, but safety check)
            if content is None:
                content = ""  # Use empty string instead of None

            # Extract raw HTML of the post element
            raw_html = str(post_element) if post_element else None
            
            return CandidatePost(
                page_url=base_url,
                post_url=post_url,
                selector_hint=selector,
                site_id=site_id,
                title=title,
                content=content,
                author=author,
                date=date,
                raw_html=raw_html
            )

        except Exception as e:
            logger.debug(f"Error creating post candidate: {e}")
            return None

    def _contains_diabetes_keyword(self, candidate: CandidatePost) -> bool:
        """Check if post content contains diabetes-related keywords."""
        if not candidate.content:
            return False

        content_lower = candidate.content.lower()
        title_lower = candidate.title.lower() if candidate.title else ""

        # Diabetes-related keywords
        diabetes_keywords = [
            'diabetes', 'diabetic', 'insulin', 'blood sugar', 'glucose',
            'type 1 diabetes', 'type 2 diabetes', 'gestational diabetes',
            'diabetes mellitus', 'hyperglycemia', 'hypoglycemia',
            'a1c', 'hba1c', 'blood glucose', 'sugar levels'
        ]

        # Check content and title for keywords
        for keyword in diabetes_keywords:
            if keyword in content_lower or keyword in title_lower:
                return True

        return False
    
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
        """Discover forum/board navigation pages (prioritize forum links over generic categories)."""
        try:
            same_host = urlparse(base_url).netloc
            forum_urls = []
            fallback_urls = []
            
            # Forum/board-specific link patterns (high priority)
            forum_selectors = [
                'a[href*="/forum"]', 'a[href*="/forums"]', 'a[href*="/board"]', 'a[href*="/boards"]',
                'a[href*="/community"]', 'a[href*="/threads"]', 'a[href*="/thread"]',
                'a[href*="/messages"]', 'a[href*="/message"]', 'a[href*="/chat"]',
                'a[href*="/discuss"]', 'a[href*="/discussion"]', 'a[href*="/discussions"]',
                '.forum-link a', '.board-link a', '.community-link a', '.thread-link a',
                '.forum-nav a', '.board-nav a', '.community-nav a'
            ]
            
            # Text-based detection: links containing forum-related keywords
            forum_keywords = ['forum', 'board', 'community', 'thread', 'message', 'chat', 'discuss']
            
            # First pass: Find forum-specific links
            for selector in forum_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get('href')
                    if href:
                        absolute_url = urljoin(base_url, href)
                        parsed_url = urlparse(absolute_url)
                        
                        if parsed_url.netloc == same_host:
                            path_lower = parsed_url.path.lower()
                            # Filter out non-forum pages
                            if not any(skip in path_lower for skip in ['/user', '/profile', '/login', '/register', '/admin', '/settings']):
                                forum_urls.append(absolute_url)
            
            # Second pass: Find links by text content and URL
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                if not href:
                    continue
                    
                link_text = link.get_text(strip=True).lower()
                absolute_url = urljoin(base_url, href)
                parsed_url = urlparse(absolute_url)
                
                if parsed_url.netloc != same_host:
                    continue
                
                path_lower = parsed_url.path.lower()
                
                # Check if link text or URL contains forum keywords
                is_forum_link = any(keyword in path_lower for keyword in forum_keywords) or \
                               any(keyword in link_text for keyword in forum_keywords)
                
                # Filter out non-forum pages
                if any(skip in path_lower for skip in ['/user', '/profile', '/login', '/register', '/admin', '/settings', 
                                                       '/category', '/gallery', '/video', '/hot', '/trending', '/popular']):
                    continue
                
                if is_forum_link:
                    forum_urls.append(absolute_url)
                else:
                    # Keep as fallback if no forum links found
                    fallback_urls.append(absolute_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_forum_urls = []
            for url in forum_urls:
                if url not in seen:
                    seen.add(url)
                    unique_forum_urls.append(url)
            
            # If we found forum links, return them; otherwise use fallbacks
            if unique_forum_urls:
                logger.info(f"Discovered {len(unique_forum_urls)} forum/board pages")
                return unique_forum_urls[:10]
            else:
                # Remove duplicates from fallbacks
                seen = set()
                unique_fallbacks = []
                for url in fallback_urls:
                    if url not in seen:
                        seen.add(url)
                        unique_fallbacks.append(url)
                logger.info(f"Discovered {len(unique_fallbacks)} fallback pages (no forum links found)")
                return unique_fallbacks[:10]
            
        except Exception as e:
            logger.error(f"Error discovering forum pages: {e}")
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
        """Discover all same-domain links from a page (for BFS crawling), prioritizing forum/board links."""
        try:
            same_host = urlparse(base_url).netloc
            forum_urls = []
            other_urls = []
            
            # Forum/board keywords for prioritization
            forum_keywords = ['forum', 'board', 'community', 'thread', 'message', 'chat', 'discuss']
            
            # Find all links
            for a in soup.find_all('a', href=True):
                href = a['href']
                absolute_url = urljoin(base_url, href)
                parsed_url = urlparse(absolute_url)
                
                # Same-domain only
                if parsed_url.netloc != same_host:
                    continue
                
                path_lower = parsed_url.path.lower()
                
                # Filter out auth/admin pages and non-forum categories
                skip_patterns = ['/login', '/register', '/user', '/profile', '/admin', '/settings',
                                '/category', '/gallery', '/video', '/hot', '/trending', '/popular', '/top']
                if any(skip in path_lower for skip in skip_patterns):
                    continue
                
                # Check if this is a forum/board link
                is_forum_link = any(keyword in path_lower for keyword in forum_keywords) or \
                               any(keyword in a.get_text(strip=True).lower() for keyword in forum_keywords)
                
                if is_forum_link:
                    forum_urls.append(absolute_url)
                else:
                    other_urls.append(absolute_url)
            
            # Prioritize forum links, then add others
            all_urls = forum_urls + other_urls
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in all_urls:
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
