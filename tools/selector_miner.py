"""
Selector Miner Core - Phase 2

A deterministic selector-miner that analyzes HTML content to generate
candidate CSS selectors for image extraction with evidence-based scoring.

Now uses the shared selector_core library for common functionality.
"""

from __future__ import annotations

import re
import logging
import os
import yaml
import json
from typing import List, Dict, Set, Tuple, Optional, Union, Any
from urllib.parse import urljoin, urlparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag
from backend.app.services.http_service import create_safe_client

# Import shared core functionality
from backend.app.services.crawler_modules.selector_core import (
    MinerNetworkError, MinerSchemaError, emit_recipe_yaml_block,
    Limits, MinedResult, CandidateSelector,
    resolve_image_url, validate_image_request, extract_extra_sources,
    stable_selector, gather_evidence, score_candidate, discover_listing_links
)

# Optional Playwright import for JavaScript rendering
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

logger = logging.getLogger(__name__)




# Functions now imported from shared selector_core library




# gather_evidence and score_candidate functions now imported from shared selector_core library


async def render_js(url: str) -> str:
    """
    JavaScript rendering shim using Playwright.
    
    Args:
        url: URL to render
        
    Returns:
        Rendered HTML content
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError("Playwright not available for JavaScript rendering")
    
    try:
        async with async_playwright() as p:
            # Launch browser in headless mode
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                java_script_enabled=True,
                images=False,  # Block images for faster rendering
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            
            page = await context.new_page()
            page.set_default_timeout(3000)  # 3 seconds
            
            logger.info(f"Rendering JavaScript for URL: {url}")
            
            # Navigate to the page
            await page.goto(url, wait_until='domcontentloaded')
            
            # Wait for potential dynamic content
            await page.wait_for_timeout(3000)
            
            # Get the rendered HTML
            html_content = await page.content()
            
            await browser.close()
            
            logger.info(f"JavaScript rendering completed for {url} ({len(html_content)} chars)")
            return html_content
            
    except Exception as e:
        logger.warning(f"JavaScript rendering failed for {url}: {e}")
        raise MinerNetworkError(f"JavaScript rendering failed: {e}") from e


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
    
    # Selector pass: find repeated containers and build candidates
    logger.info("Starting selector pass")
    selector_candidates = _selector_pass(soup, base_url, limits)
    stats['static_candidates'] = len(selector_candidates)
    
    # If homepage didn't yield good results, try category pages
    if len(selector_candidates) < 2:
        logger.info("Homepage yielded few candidates, discovering category pages...")
        category_urls = await _discover_category_pages(soup, base_url, client)
        
        for category_url in category_urls:
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
    
    # If no candidates and JS is enabled, try JavaScript rendering
    if len(selector_candidates) == 0 and use_js:
        logger.info("No static candidates found, trying JavaScript rendering")
        try:
            js_html = await render_js(url)
            js_soup = BeautifulSoup(js_html, 'html.parser')
            js_candidates = _selector_pass(js_soup, base_url, limits)
            stats['js_candidates'] = len(js_candidates)
            
            if len(js_candidates) > 0:
                selector_candidates = js_candidates
                logger.info(f"JavaScript rendering found {len(js_candidates)} candidates")
            else:
                logger.info("JavaScript rendering found no additional candidates")
        except Exception as e:
            logger.warning(f"JavaScript rendering failed: {e}")
    
    # Extra sources pass: extract and validate extra sources
    logger.info("Starting extra sources pass")
    extra_candidates = await _extra_sources_pass(soup, base_url, client, limits)
    stats['extra_sources'] = len(extra_candidates)
    
    # Combine and validate candidates
    all_candidates = selector_candidates + extra_candidates
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
    
    return MinedResult(candidates=validated_candidates, status=status, stats=stats)


# discover_listing_links function now imported from shared selector_core library


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
            response = await client.get(url, timeout=HTTP_TIMEOUT)
            if response.status_code == 200:
                tested_urls.append(url)
                if len(tested_urls) >= 3:  # Stop after finding 3 working category pages
                    break
        except Exception:
            continue
    
    logger.info(f"Discovered {len(tested_urls)} working category pages: {tested_urls}")
    return tested_urls


def _selector_pass(soup: BeautifulSoup, base_url: str, limits: Limits) -> List[CandidateSelector]:
    """Selector pass: find repeated containers and build candidates."""
    # Find repeated container patterns
    container_patterns = [
        '.thumb', '.thumbnail', '.video', '.item', '.list-global__item',
        '.gallery-item', '.media-item', '.content-item', '.post-item',
        '.thumb-block', '.thumb-cat', '.thumb-inside', '.thumb-wrapper',
        '.video-thumb', '.video-item', '.media-thumb', '.content-thumb'
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
                srcset_count=evidence_counts['srcset_count']
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




@dataclass
class ImageNode:
    """Represents an image node found in HTML."""
    tag: Tag
    url: str
    selector_path: str
    attributes: Dict[str, str]
    context: Dict[str, any]


@dataclass
class ValidationResult:
    """Result of URL validation."""
    url: str
    is_valid: bool
    status_code: Optional[int] = None
    content_type: Optional[str] = None
    error: Optional[str] = None


@dataclass
class RecipeCandidate:
    """A candidate recipe for a domain."""
    domain: str
    selectors: List[Dict[str, str]]
    attributes_priority: List[str]
    extra_sources: List[str]
    method: str
    confidence: float
    sample_urls: List[str]
    validation_results: List[ValidationResult]


