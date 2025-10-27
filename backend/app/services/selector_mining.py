"""
Selector Mining Service

Service interface for selector mining that integrates with the crawler.
Uses the shared selector_core library for common functionality.

This service mines CSS selectors for image extraction from HTML content,
with evidence-based scoring and validation, including 3x3 depth mining.
"""

import logging
import os
import json
from typing import List, Dict, Set, Tuple, Optional, Union, Any
from urllib.parse import urljoin, urlparse
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

from .http_service import HttpService, fetch_html_with_redirects, fetch_html_with_js_rendering
from .crawler_modules.selector_core import (
    MinerNetworkError, MinerSchemaError, emit_recipe_yaml_block,
    Limits, MinedResult, CandidateSelector,
    resolve_image_url, validate_image_request, extract_extra_sources,
    stable_selector, gather_evidence, score_candidate, discover_listing_links
)

logger = logging.getLogger(__name__)


async def _discover_category_pages(soup: BeautifulSoup, base_url: str, client: httpx.AsyncClient) -> List[str]:
    """Discover category/listing pages that likely contain video thumbnails."""
    same_host = urlparse(base_url).netloc
    category_urls = discover_listing_links(soup, base_url, same_host)
    
    # Also try common category URL patterns as fallback
    parsed_base = urlparse(base_url)
    common_paths = ['/new', '/latest', '/trending', '/hot', '/videos', '/category']
    for path in common_paths:
        fallback_url = f"{parsed_base.scheme}://{parsed_base.netloc}{path}"
        if fallback_url not in category_urls:
            category_urls.append(fallback_url)
    
    # Test up to 3 category URLs to see if they have video thumbnails
    tested_urls = []
    for url in category_urls[:5]:  # Test first 5 discovered URLs
        try:
            response = await client.get(url, timeout=10.0)
            if response.status_code == 200:
                tested_urls.append(url)
                if len(tested_urls) >= 3:  # Stop after finding 3 working category pages
                    break
        except Exception:
            continue
    
    logger.info(f"Discovered {len(tested_urls)} working category pages: {tested_urls}")
    return tested_urls


def _add_structural_analysis(soup: BeautifulSoup, candidate_groups: dict, base_url: str) -> None:
    """
    Enhanced structural analysis to detect repeated parent elements and common ancestor containers.
    This helps find grid/gallery structures that might not match standard patterns.
    """
    # Find all images first
    all_images = soup.find_all(['img', 'source', 'video'])
    
    # Group images by their parent elements
    parent_groups = {}
    for img in all_images:
        parent = img.parent
        if parent:
            parent_tag = parent.name
            parent_classes = ' '.join(parent.get('class', []))
            parent_id = parent.get('id', '')
            
            # Create a key that identifies similar parent structures
            parent_key = f"{parent_tag}.{parent_classes}.{parent_id}"
            
            if parent_key not in parent_groups:
                parent_groups[parent_key] = []
            parent_groups[parent_key].append(img)
    
    # Find parent groups with multiple images (potential grids)
    for parent_key, images in parent_groups.items():
        if len(images) >= 3:  # Only consider parents with 3+ images
            # Find common ancestor containers
            common_ancestors = _find_common_ancestors(images)
            
            for ancestor in common_ancestors:
                # Generate selector for this ancestor
                ancestor_selector = _generate_ancestor_selector(ancestor)
                if ancestor_selector:
                    # Add images to candidate groups
                    for img in images:
                        if ancestor_selector not in candidate_groups:
                            candidate_groups[ancestor_selector] = []
                        candidate_groups[ancestor_selector].append(img)
    
    # Detect list structures (ul > li, div[class*=grid] > div)
    _detect_list_structures(soup, candidate_groups, base_url)


def _find_common_ancestors(images: List) -> List:
    """Find common ancestor elements for a group of images."""
    if not images:
        return []
    
    # Start with the first image's ancestors
    first_img = images[0]
    ancestors = []
    current = first_img.parent
    while current:
        ancestors.append(current)
        current = current.parent
    
    # Filter to only ancestors that contain all images
    common_ancestors = []
    for ancestor in ancestors:
        if all(ancestor.find(img) for img in images):
            common_ancestors.append(ancestor)
    
    return common_ancestors


def _generate_ancestor_selector(ancestor) -> Optional[str]:
    """Generate a stable CSS selector for an ancestor element."""
    if not ancestor:
        return None
    
    # Try to create a meaningful selector
    tag = ancestor.name
    classes = ancestor.get('class', [])
    element_id = ancestor.get('id', '')
    
    if element_id:
        return f"#{element_id}"
    elif classes:
        # Use the most specific class
        main_class = classes[0]
        return f".{main_class}"
    else:
        return tag


def _detect_list_structures(soup: BeautifulSoup, candidate_groups: dict, base_url: str) -> None:
    """Detect list-based structures like ul > li, div[class*=grid] > div."""
    # Look for ul/ol with multiple li elements containing images
    for list_tag in ['ul', 'ol']:
        lists = soup.find_all(list_tag)
        for list_elem in lists:
            list_items = list_elem.find_all('li')
            if len(list_items) >= 3:  # Only consider lists with 3+ items
                # Check if items contain images
                images_in_list = []
                for li in list_items:
                    images = li.find_all(['img', 'source', 'video'])
                    images_in_list.extend(images)
                
                if len(images_in_list) >= 3:
                    # Generate selector for the list
                    list_selector = _generate_list_selector(list_elem)
                    if list_selector:
                        for img in images_in_list:
                            if list_selector not in candidate_groups:
                                candidate_groups[list_selector] = []
                            candidate_groups[list_selector].append(img)
    
    # Look for div-based grid structures
    grid_containers = soup.find_all('div', class_=lambda x: x and any(
        keyword in ' '.join(x).lower() for keyword in ['grid', 'masonry', 'gallery', 'list']
    ))
    
    for container in grid_containers:
        # Find direct children that might be grid items
        grid_items = [child for child in container.children 
                     if hasattr(child, 'find') and child.find(['img', 'source', 'video'])]
        
        if len(grid_items) >= 3:
            container_selector = _generate_ancestor_selector(container)
            if container_selector:
                for item in grid_items:
                    images = item.find_all(['img', 'source', 'video'])
                    for img in images:
                        if container_selector not in candidate_groups:
                            candidate_groups[container_selector] = []
                        candidate_groups[container_selector].append(img)


def _generate_list_selector(list_elem) -> Optional[str]:
    """Generate a selector for a list element."""
    if not list_elem:
        return None
    
    classes = list_elem.get('class', [])
    element_id = list_elem.get('id', '')
    
    if element_id:
        return f"#{element_id}"
    elif classes:
        return f".{'.'.join(classes)}"
    else:
        return list_elem.name


def _selector_pass(soup: BeautifulSoup, base_url: str, limits: Limits) -> List[CandidateSelector]:
    """Selector pass: find repeated containers and build candidates."""
    # Find repeated container patterns - Enhanced with 30+ patterns
    container_patterns = [
        # Video/media patterns
        '.video', '.video-item', '.video-thumb', '.video-card', '.video-block',
        '.media', '.media-item', '.media-thumb', '.media-card', '.media-wrapper',
        # Gallery patterns
        '.gallery', '.gallery-item', '.gallery-thumb', '.gallery-card', '.gallery-block',
        '.grid-item', '.masonry-item', '.photo-item', '.photo-card',
        # Thumbnail patterns
        '.thumb', '.thumbnail', '.thumb-block', '.thumb-wrapper', '.thumb-cat',
        '.thumb-inside', '.thumb-container', '.thumb-holder',
        # Generic container patterns
        '.item', '.card', '.post', '.entry', '.content-item', '.list-item', '.feed-item',
        '.tile', '.cell', '.box', '.block', '.container-item',
        # Framework-specific patterns
        '[class*="thumb"]', '[class*="grid"]', '[class*="card"]', '[class*="item"]',
        '[class*="video"]', '[class*="media"]', '[class*="gallery"]',
        # List structures
        'li', 'li.item', 'li.card', 'li.thumb',
        # Legacy patterns (preserve existing)
        '.list-global__item', '.post-item', '.video-thumb', '.media-thumb', '.content-thumb'
    ]
    
    candidate_groups = {}
    
    # Find containers and their images
    for pattern in container_patterns:
        containers = soup.select(pattern)
        if len(containers) >= 3:  # Only consider patterns with 3+ instances
            for container in containers:
                # Find images within this container
                images = container.find_all(['img', 'source', 'video'])
                for img in images:
                    # Resolve URL
                    url = resolve_image_url(img, base_url)
                    if not url:
                        continue
                    
                    # Generate stable selector
                    selector = stable_selector(img)
                    if not selector:
                        continue
                    
                    # Group by selector
                    if selector not in candidate_groups:
                        candidate_groups[selector] = []
                    candidate_groups[selector].append(img)
    
    # Enhanced structural analysis: detect repeated parent elements
    _add_structural_analysis(soup, candidate_groups, base_url)
    
    # Build candidates from groups
    candidates = []
    for selector, nodes in candidate_groups.items():
        if len(nodes) >= 2:  # Only consider selectors with 2+ matches
            # Gather evidence and score
            evidence_counts = gather_evidence(nodes)
            score = score_candidate(
                repeats=evidence_counts['repeats'],
                has_duration=evidence_counts['has_duration'],
                has_video_href=evidence_counts['has_video_href'],
                class_hits=evidence_counts['class_hits'],
                srcset_count=evidence_counts['srcset_count'],
                structural_bonus=evidence_counts['structural_bonus']
            )
            
            # Extract sample URLs
            sample_urls = []
            seen_urls = set()
            for node in nodes:
                url = resolve_image_url(node, base_url)
                if url and url not in seen_urls and len(sample_urls) < 12:
                    sample_urls.append(url)
                    seen_urls.add(url)
            
            # Create candidate
            candidate = CandidateSelector(
                selector=selector,
                description=f"{len(nodes)} images in {selector}",
                evidence={},
                sample_urls=sample_urls,
                repetition_count=len(nodes),
                score=score
            )
            candidates.append(candidate)
    
    # Sort by score and limit
    candidates.sort(key=lambda x: x.score, reverse=True)
    return candidates[:limits.max_candidates]


async def _extra_sources_pass(soup: BeautifulSoup, base_url: str, client: httpx.AsyncClient, limits: Limits) -> List[CandidateSelector]:
    """Extra sources pass: extract and validate extra sources."""
    # Get extra source selectors
    extra_selectors = extract_extra_sources(soup, base_url)
    
    candidates = []
    seen_selectors = set()
    
    for selector in extra_selectors:
        if selector in seen_selectors:
            continue
        seen_selectors.add(selector)
        
        # Extract URLs based on selector type
        urls = []
        
        if selector.startswith("meta[property='og:image']"):
            meta = soup.find('meta', property='og:image')
            if meta and meta.get('content'):
                urls.append(meta.get('content'))
        
        elif selector.startswith("link[rel='image_src']"):
            link = soup.find('link', rel='image_src')
            if link and link.get('href'):
                urls.append(link.get('href'))
        
        elif selector.startswith("video::attr(poster)"):
            videos = soup.find_all('video')
            for video in videos:
                if video.get('poster'):
                    urls.append(video.get('poster'))
        
        elif "srcset" in selector:
            if "img::attr(srcset)" in selector:
                imgs = soup.find_all('img', srcset=True)
                for img in imgs:
                    srcset = img.get('srcset')
                    if srcset:
                        # Extract URLs from srcset
                        for candidate in srcset.split(','):
                            candidate = candidate.strip()
                            if candidate:
                                url = candidate.split()[0]
                                urls.append(url)
            
            elif "source::attr(srcset)" in selector:
                sources = soup.find_all('source', srcset=True)
                for source in sources:
                    srcset = source.get('srcset')
                    if srcset:
                        # Extract URLs from srcset
                        for candidate in srcset.split(','):
                            candidate = candidate.strip()
                            if candidate:
                                url = candidate.split()[0]
                                urls.append(url)
        
        elif selector.startswith("script[type='application/ld+json']"):
            scripts = soup.find_all('script', type='application/ld+json')
            for script in scripts:
                try:
                    if script.string:
                        json_data = json.loads(script.string.strip())
                        json_objects = json_data if isinstance(json_data, list) else [json_data]
                        for obj in json_objects:
                            if isinstance(obj, dict):
                                for field in ['thumbnailUrl', 'image']:
                                    if field in obj:
                                        if isinstance(obj[field], str):
                                            urls.append(obj[field])
                                        elif isinstance(obj[field], list):
                                            urls.extend([url for url in obj[field] if isinstance(url, str)])
                except (json.JSONDecodeError, TypeError):
                    continue
        
        elif selector == "::style(background-image)":
            elements = soup.find_all(style=True)
            for element in elements:
                style = element.get('style', '')
                # Import the regex from selector_core
                from .crawler_modules.selector_core import _BACKGROUND_IMAGE_REGEX
                bg_match = _BACKGROUND_IMAGE_REGEX.search(style)
                if bg_match:
                    urls.append(bg_match.group(1))
        
        # Validate URLs and create candidate
        if urls:
            # Resolve relative URLs
            resolved_urls = []
            for url in urls:
                if url.startswith('//'):
                    resolved_urls.append('https:' + url)
                elif url.startswith('/'):
                    resolved_urls.append(urljoin(base_url, url))
                else:
                    resolved_urls.append(url)
            
            # Limit sample URLs
            sample_urls = resolved_urls[:12]
            
            # Create candidate with special label
            candidate = CandidateSelector(
                selector=selector,
                description=f"EXTRA_SOURCES: {selector}",
                evidence={},
                sample_urls=sample_urls,
                repetition_count=len(sample_urls),
                score=0.5  # Base score for extra sources
            )
            candidates.append(candidate)
    
    return candidates


async def _validate_candidates(candidates: List[CandidateSelector], client: httpx.AsyncClient, limits: Limits) -> List[CandidateSelector]:
    """Validate candidates by checking their sample URLs."""
    validated_candidates = []
    
    for candidate in candidates:
        valid_urls = []
        
        # Validate sample URLs
        for url in candidate.sample_urls[:limits.max_samples_per_candidate]:
            try:
                is_valid = await validate_image_request(url, client, limits.max_bytes)
                if is_valid:
                    valid_urls.append(url)
            except Exception as e:
                logger.debug(f"Validation failed for {url}: {e}")
                continue
        
        # Only keep candidates with at least one valid URL
        if valid_urls:
            candidate.sample_urls = valid_urls
            candidate.repetition_count = len(valid_urls)
            validated_candidates.append(candidate)
    
    return validated_candidates


async def mine_page(url: str, html: Optional[str], *, use_js: bool, client: httpx.AsyncClient, limits: Limits) -> MinedResult:
    """
    Single mining entrypoint that orchestrates static and optional JavaScript mining.
    
    Args:
        url: URL to mine
        html: HTML content (if None, will fetch once statically)
        use_js: Whether to use JavaScript fallback if no candidates found
        client: HTTP client for fetching and validation
        limits: Configuration limits
        
    Returns:
        MinedResult with candidates, status, and stats
    """
    stats = {
        'static_candidates': 0,
        'extra_sources': 0,
        'js_candidates': 0,
        'total_validated': 0
    }
    checked_urls = [url]  # Track the main URL
    
    # Fetch HTML if not provided
    if html is None:
        try:
            response = await client.get(url, timeout=limits.timeout_seconds)
            html = response.text
            logger.info(f"Fetched HTML from {url} ({len(html)} chars)")
        except Exception as e:
            logger.error(f"Failed to fetch HTML from {url}: {e}")
            raise MinerNetworkError(f"Failed to fetch HTML from {url}: {e}") from e
    
    # Parse HTML
    soup = BeautifulSoup(html, 'html.parser')
    base_url = url
    active_soup = soup  # Track which soup to use for extra sources
    
    # Selector pass: find repeated containers and build candidates
    logger.info("Starting selector pass")
    selector_candidates = _selector_pass(soup, base_url, limits)
    stats['static_candidates'] = len(selector_candidates)
    
    # If homepage didn't yield good results, try category pages
    if len(selector_candidates) < 2:
        logger.info("Homepage yielded few candidates, discovering category pages...")
        category_urls = await _discover_category_pages(soup, base_url, client)
        
        for category_url in category_urls:
            checked_urls.append(category_url)  # Track category URLs
            try:
                logger.info(f"Testing category page: {category_url}")
                cat_response = await client.get(category_url, timeout=limits.timeout_seconds)
                if cat_response.status_code == 200:
                    cat_soup = BeautifulSoup(cat_response.text, 'html.parser')
                    cat_candidates = _selector_pass(cat_soup, category_url, limits)
                    
                    # Add candidates from category pages
                    selector_candidates.extend(cat_candidates)
                    logger.info(f"Found {len(cat_candidates)} additional candidates from {category_url}")
                    
                    if len(selector_candidates) >= 3:  # Stop once we have enough candidates
                        break
            except Exception as e:
                logger.debug(f"Failed to fetch category page {category_url}: {e}")
                continue
    
    # If 3 or fewer candidates and JS is enabled, try JavaScript rendering
    if len(selector_candidates) <= 3 and use_js:
        logger.info("Found 3 or fewer candidates, trying JavaScript rendering")
        try:
            from app.services.http_service import fetch_html_with_js_rendering
            js_html, js_errors = await fetch_html_with_js_rendering(
                url, 
                timeout=10.0, 
                wait_for_network_idle=True
            )
            if js_html:
                js_soup = BeautifulSoup(js_html, 'html.parser')
                js_candidates = _selector_pass(js_soup, base_url, limits)
                stats['js_candidates'] = len(js_candidates)
                
                if len(js_candidates) > 0:
                    selector_candidates.extend(js_candidates)
                    active_soup = js_soup  # Use JS soup for extra sources
                    logger.info(f"JavaScript rendering found {len(js_candidates)} candidates")
                else:
                    logger.info("JavaScript rendering found no additional candidates")
            else:
                logger.warning(f"JavaScript rendering failed: {js_errors}")
        except Exception as e:
            logger.warning(f"JavaScript rendering failed: {e}")
    
    # Extra sources pass: extract and validate extra sources
    logger.info("Starting extra sources pass")
    extra_candidates = await _extra_sources_pass(active_soup, base_url, client, limits)
    stats['extra_sources'] = len(extra_candidates)
    
    # Combine and validate candidates
    all_candidates = selector_candidates + extra_candidates
    logger.info(f"Combined {len(selector_candidates)} total candidates (static + JS)")
    validated_candidates = await _validate_candidates(all_candidates, client, limits)
    stats['total_validated'] = len(validated_candidates)
    
    # Determine status
    if len(selector_candidates) > 0:
        status = "OK"
    elif len(extra_candidates) > 0:
        status = "EXTRA_ONLY"
    else:
        status = "NO_THUMBS_STATIC"
    
    # One-line summary logging
    skipped_by = {
        'low_score': stats['static_candidates'] + stats['extra_sources'] - stats['total_validated'],
        'network_errors': 0  # Could be tracked if needed
    }
    logger.info(f"candidates={len(validated_candidates)} accepted={len(validated_candidates)} skipped_by={skipped_by}")
    
    return MinedResult(candidates=validated_candidates, status=status, stats=stats, checked_urls=checked_urls)


# Service interface for integration with crawler
class SelectorMiningService:
    """
    Service interface for selector mining that can be integrated with the crawler.
    """
    
    def __init__(self, http_service: Optional[HttpService] = None):
        self.http_service = http_service or HttpService()
    
    async def mine_selectors_for_page(
        self, 
        html: str = None,
        url: str = None, 
        client: httpx.AsyncClient = None,
        limits: Optional[Limits] = None,
        use_js_rendering: bool = False
    ) -> MinedResult:
        """
        Mine selectors from a page with optional JS rendering.
        
        Args:
            html: HTML content of the page (optional)
            url: URL of the page (required if html is None)
            client: HTTP client for validation
            limits: Configuration limits
            use_js_rendering: Whether to use JS rendering if static mining fails
            
        Returns:
            MinedResult with candidates and metadata
        """
        if limits is None:
            limits = Limits()
        
        # Always start with provided HTML or fetch via standard HTTP
        # JS rendering will be handled as fallback in mine_page if needed
        return await mine_page(url, html, use_js=True, client=client, limits=limits)
    
    async def mine_selectors_for_site(
        self, 
        base_url: str, 
        client: httpx.AsyncClient,
        max_pages: int = 5,
        limits: Optional[Limits] = None
    ) -> MinedResult:
        """
        Mine selectors from a site using 3x3 depth approach for better structure diversity.
        
        Args:
            base_url: Base URL of the site
            client: HTTP client for fetching and validation
            max_pages: Maximum number of pages to crawl (will be used for 3x3 approach)
            limits: Configuration limits
            
        Returns:
            MinedResult with candidates from all pages
        """
        if limits is None:
            limits = Limits()
        
        # Fetch the base page
        try:
            response = await client.get(base_url, timeout=limits.timeout_seconds)
            if response.status_code != 200:
                raise MinerNetworkError(f"Failed to fetch base page: {response.status_code}")
            
            html = response.text
            soup = BeautifulSoup(html, 'html.parser')
            
            # Mine from base page
            result = await mine_page(base_url, html, use_js=False, client=client, limits=limits)
            all_checked_urls = result.checked_urls.copy()  # Track all URLs
            
            # Use 3x3 depth approach: find 3 category pages, then 3 content pages from each
            logger.info("Starting 3x3 depth mining approach...")
            deep_result = await self._deep_mine_3x3(soup, base_url, client, limits)
            
            # Merge results
            result.candidates.extend(deep_result.candidates)
            result.stats['static_candidates'] += deep_result.stats['static_candidates']
            result.stats['extra_sources'] += deep_result.stats['extra_sources']
            result.stats['total_validated'] += deep_result.stats['total_validated']
            all_checked_urls.extend(deep_result.checked_urls)
            
            # Re-sort and limit candidates
            result.candidates.sort(key=lambda x: x.score, reverse=True)
            result.candidates = result.candidates[:limits.max_candidates]
            
            # Update checked URLs
            result.checked_urls = all_checked_urls
            
            logger.info(f"3x3 mining completed: {len(all_checked_urls)} URLs checked, {len(result.candidates)} candidates found")
            return result
            
        except Exception as e:
            logger.error(f"Failed to mine site {base_url}: {e}")
            raise MinerNetworkError(f"Failed to mine site {base_url}: {e}") from e
    
    async def _deep_mine_3x3(
        self, 
        soup: BeautifulSoup, 
        base_url: str, 
        client: httpx.AsyncClient, 
        limits: Limits
    ) -> MinedResult:
        """
        Perform 3x3 depth mining: find 3 category pages, then 3 content pages from each.
        This provides better structure diversity than shallow breadth-first search.
        """
        all_candidates = []
        all_checked_urls = []
        stats = {
            'static_candidates': 0,
            'extra_sources': 0,
            'total_validated': 0
        }
        
        # Step 1: Find 3 category/listing pages
        category_urls = await _discover_category_pages(soup, base_url, client)
        logger.info(f"Found {len(category_urls)} category pages, selecting top 3 for deep mining")
        
        # Take first 3 category pages
        selected_categories = category_urls[:3]
        
        for i, category_url in enumerate(selected_categories):
            logger.info(f"Deep mining category {i+1}/3: {category_url}")
            all_checked_urls.append(category_url)
            
            try:
                # Fetch category page
                cat_response = await client.get(category_url, timeout=limits.timeout_seconds)
                if cat_response.status_code != 200:
                    logger.warning(f"Failed to fetch category page {category_url}: {cat_response.status_code}")
                    continue
                
                cat_soup = BeautifulSoup(cat_response.text, 'html.parser')
                
                # Mine from category page
                cat_result = await mine_page(category_url, cat_response.text, use_js=False, client=client, limits=limits)
                all_candidates.extend(cat_result.candidates)
                stats['static_candidates'] += cat_result.stats['static_candidates']
                stats['extra_sources'] += cat_result.stats['extra_sources']
                stats['total_validated'] += cat_result.stats['total_validated']
                all_checked_urls.extend(cat_result.checked_urls)
                
                # Step 2: Find 3 content pages from this category
                content_urls = await self._discover_content_pages(cat_soup, category_url, client)
                logger.info(f"Found {len(content_urls)} content pages in category, selecting top 3")
                
                # Take first 3 content pages from this category
                selected_content = content_urls[:3]
                
                for j, content_url in enumerate(selected_content):
                    logger.info(f"Deep mining content {j+1}/3 from category {i+1}: {content_url}")
                    all_checked_urls.append(content_url)
                    
                    try:
                        # Fetch content page
                        content_response = await client.get(content_url, timeout=limits.timeout_seconds)
                        if content_response.status_code != 200:
                            logger.warning(f"Failed to fetch content page {content_url}: {content_response.status_code}")
                            continue
                        
                        # Mine from content page
                        content_result = await mine_page(content_url, content_response.text, use_js=False, client=client, limits=limits)
                        all_candidates.extend(content_result.candidates)
                        stats['static_candidates'] += content_result.stats['static_candidates']
                        stats['extra_sources'] += content_result.stats['extra_sources']
                        stats['total_validated'] += content_result.stats['total_validated']
                        all_checked_urls.extend(content_result.checked_urls)
                        
                    except Exception as e:
                        logger.debug(f"Failed to mine content page {content_url}: {e}")
                        continue
                        
            except Exception as e:
                logger.debug(f"Failed to mine category page {category_url}: {e}")
                continue
        
        logger.info(f"3x3 deep mining completed: {len(all_checked_urls)} URLs checked, {len(all_candidates)} candidates found")
        
        return MinedResult(
            candidates=all_candidates,
            status="OK",
            stats=stats,
            checked_urls=all_checked_urls
        )
    
    async def _discover_content_pages(self, soup: BeautifulSoup, base_url: str, client: httpx.AsyncClient) -> List[str]:
        """
        Discover content pages (threads, posts, videos, etc.) from a category page.
        This looks for links that lead to individual content items rather than more categories.
        """
        same_host = urlparse(base_url).netloc
        content_urls = []
        
        # Look for common content link patterns
        content_selectors = [
            'a[href*="/t/"]',  # Discourse threads
            'a[href*="/thread"]',  # Thread links
            'a[href*="/post"]',  # Post links
            'a[href*="/video"]',  # Video links
            'a[href*="/watch"]',  # Watch links
            'a[href*="/view"]',  # View links
            'a[href*="/item"]',  # Item links
            'a[href*="/article"]',  # Article links
            'a[href*="/story"]',  # Story links
            '.topic-title a',  # Topic titles
            '.thread-title a',  # Thread titles
            '.post-title a',  # Post titles
            '.item-title a',  # Item titles
            '.content-title a',  # Content titles
        ]
        
        for selector in content_selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    # Convert relative URLs to absolute
                    absolute_url = urljoin(base_url, href)
                    parsed_url = urlparse(absolute_url)
                    
                    # Only include same-host URLs
                    if parsed_url.netloc == same_host:
                        # Filter out obvious category/admin pages
                        if not any(skip in parsed_url.path.lower() for skip in ['/category', '/admin', '/user', '/profile', '/settings']):
                            content_urls.append(absolute_url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in content_urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        
        logger.info(f"Discovered {len(unique_urls)} potential content pages")
        return unique_urls[:10]  # Return top 10 for selection
    
    async def save_selectors_to_recipe(
        self, 
        domain: str, 
        selectors: List[str], 
        recipe_file: str = "site_recipes.yaml"
    ) -> bool:
        """
        Save mined selectors to a recipe file.
        
        Args:
            domain: Domain name
            selectors: List of CSS selectors
            recipe_file: Path to recipe file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            import yaml
            
            # Load existing recipes
            if os.path.exists(recipe_file):
                with open(recipe_file, 'r', encoding='utf-8') as f:
                    recipes = yaml.safe_load(f) or {}
            else:
                recipes = {
                    'schema_version': 1,
                    'defaults': {
                        'selectors': [
                            {'selector': '.video-thumb img', 'description': '.video-thumb container images'},
                            {'selector': '.thumbnail img', 'description': '.thumbnail container images'},
                            {'selector': '.thumb img', 'description': '.thumb container images'},
                            {'selector': 'picture img', 'description': 'picture element images'},
                            {'selector': 'img.lazy', 'description': 'lazy-loaded images'},
                            {'selector': 'img', 'description': 'all images'}
                        ],
                        'attributes_priority': ['data-src', 'data-srcset', 'srcset', 'src'],
                        'extra_sources': [
                            "meta[property='og:image']::attr(content)",
                            "link[rel='image_src']::attr(href)",
                            "video::attr(poster)",
                            "img::attr(srcset)",
                            "source::attr(srcset)",
                            "source::attr(data-srcset)",
                            "script[type='application/ld+json']::jsonpath($.thumbnailUrl, $.image, $..thumbnailUrl, $..image, $..url)",
                            "::style(background-image)"
                        ],
                        'method': 'smart'
                    },
                    'sites': {}
                }
            
            # Ensure sites section exists
            if 'sites' not in recipes:
                recipes['sites'] = {}
            
            # Add the new recipe
            recipes['sites'][domain] = {
                'selectors': [
                    {'selector': sel, 'description': f"Extracted from {domain}"}
                    for sel in selectors
                ],
                'attributes_priority': list(ATTR_PRIORITY),
                'extra_sources': [],
                'method': 'miner'
            }
            
            # Write back to file
            with open(recipe_file, 'w', encoding='utf-8') as f:
                yaml.dump(recipes, f, default_flow_style=False, sort_keys=False)
            
            logger.info(f"Saved {len(selectors)} selectors for {domain} to {recipe_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save selectors for {domain}: {e}")
            return False
    
    async def close(self):
        """Close the service and clean up resources."""
        if self.http_service:
            await self.http_service.close()
