"""
Selector Miner Core - Phase 2

A deterministic selector-miner that analyzes HTML content to generate
candidate CSS selectors for image extraction with evidence-based scoring.
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
from .redirect_utils import create_safe_client

# Optional Playwright import for JavaScript rendering
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

logger = logging.getLogger(__name__)

# Constants bucket - minimal set of configurable values
ATTR_PRIORITY = ("data-src", "data-srcset", "data-image", "srcset", "src")
HOST_DENY = {"doubleclick.net", "exoclick.com", "s.magsrv.com", "afcdn.net"}
MAX_REDIRECTS = 3
MAX_IMAGE_BYTES = int(os.getenv("MINER_MAX_IMAGE_BYTES", 10 * 1024 * 1024))
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}
HTTP_TIMEOUT = 10.0
CHUNK_SIZE = 8192


class MinerNetworkError(Exception):
    """Network-related errors during mining operations."""
    pass


class MinerSchemaError(Exception):
    """Schema validation errors for YAML emissions."""
    pass


def emit_recipe_yaml_block(domain: str, selectors: List[str], attr_priority: List[str], extra_sources: List[str]) -> Dict[str, Any]:
    """
    Emit a normalized YAML recipe block for a domain.
    
    Args:
        domain: Domain name for the recipe
        selectors: List of CSS selectors
        attr_priority: List of attribute names in priority order
        extra_sources: List of extra source selectors
        
    Returns:
        Dictionary ready for YAML serialization and merging
        
    Raises:
        MinerSchemaError: If schema validation fails
    """
    # Schema normalization: drop non-URL attributes
    # Only keep attributes that are commonly used for image URLs
    supported_attrs = set(ATTR_PRIORITY) | {'poster', 'content', 'href'}
    normalized_attr_priority = [attr for attr in attr_priority if attr in supported_attrs]
    
    # If no supported attributes found, use default priority
    if not normalized_attr_priority:
        normalized_attr_priority = list(ATTR_PRIORITY)
    
    # Schema normalization: filter extra_sources to supported types only
    supported_extra_sources = {
        "meta[property='og:image']::attr(content)",
        "link[rel='image_src']::attr(href)",
        "video::attr(poster)",
        "img::attr(srcset)",
        "source::attr(srcset)",
        "source::attr(data-srcset)",
        "script[type='application/ld+json']",
        "::style(background-image)"
    }
    normalized_extra_sources = [source for source in extra_sources if source in supported_extra_sources]
    
    # Validate domain
    if not domain or not isinstance(domain, str):
        raise MinerSchemaError("Domain must be a non-empty string")
    
    # Validate selectors
    if not selectors or not isinstance(selectors, list):
        raise MinerSchemaError("Selectors must be a non-empty list")
    
    if not all(isinstance(sel, str) and sel.strip() for sel in selectors):
        raise MinerSchemaError("All selectors must be non-empty strings")
    
    # Construct the YAML block
    recipe_block = {
        'domain': domain,
        'selectors': [
            {
                'selector': selector,
                'description': f"Extracted from {domain}",
                'sample_urls': [],  # CLIs can populate these separately
                'score': 0.8  # Default confidence score
            }
            for selector in selectors
        ],
        'attributes_priority': normalized_attr_priority,
        'extra_sources': normalized_extra_sources,
        'method': 'miner',
        'confidence': 0.8,
        'sample_urls': [],
        'validation_results': []
    }
    
    return recipe_block


@dataclass
class Limits:
    """Configuration limits for mining operations."""
    max_candidates: int = 10
    max_samples_per_candidate: int = 3
    max_bytes: int = MAX_IMAGE_BYTES
    timeout_seconds: int = int(HTTP_TIMEOUT)


@dataclass
class MinedResult:
    """Result of page mining operation."""
    candidates: List[CandidateSelector]
    status: str  # "OK", "NO_THUMBS_STATIC", "EXTRA_ONLY"
    stats: Dict[str, int]


@dataclass
class CandidateSelector:
    """Represents a candidate CSS selector with metadata and evidence."""
    selector: str
    description: str
    evidence: Dict[str, float]
    sample_urls: List[str]
    repetition_count: int
    score: float = 0.0




def resolve_image_url(el: Tag, base_url: str) -> Optional[str]:
    """
    Resolve image URL from HTML element following exact attribute priority order.
    
    Priority order: data-src > data-srcset > srcset > src
    
    Args:
        el: BeautifulSoup Tag element
        base_url: Base URL for resolving relative URLs
        
    Returns:
        Absolute URL or None if no valid URL found
    """
    if not el or not base_url:
        return None
    
    # 1. Check data-src first (highest priority)
    data_src = el.get('data-src')
    if data_src:
        if data_src.startswith('//'):
            return 'https:' + data_src
        elif data_src.startswith('/'):
            return urljoin(base_url, data_src)
        return data_src
    
    # 1.5. Check data-image (high priority for porn sites)
    data_image = el.get('data-image')
    if data_image:
        if data_image.startswith('//'):
            return 'https:' + data_image
        elif data_image.startswith('/'):
            return urljoin(base_url, data_image)
        return data_image
    
    # 2. Check data-srcset (second priority)
    data_srcset = el.get('data-srcset')
    if data_srcset:
        try:
            candidates = []
            for candidate in data_srcset.split(','):
                candidate = candidate.strip()
                if not candidate:
                    continue
                parts = candidate.split()
                if len(parts) < 2:
                    continue
                url = parts[0]
                descriptor = parts[1]
                # Normalize URL
                if url.startswith('//'):
                    url = 'https:' + url
                elif url.startswith('/'):
                    url = urljoin(base_url, url)
                # Parse descriptor
                width = 0
                if descriptor.endswith('x'):
                    try:
                        density = float(descriptor[:-1])
                        width = int(320 * density)
                    except ValueError:
                        continue
                elif descriptor.endswith('w'):
                    try:
                        width = int(descriptor[:-1])
                    except ValueError:
                        continue
                else:
                    width = 320
                candidates.append((url, width))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
        except Exception:
            pass
    
    # 3. Check srcset (third priority)
    srcset = el.get('srcset')
    if srcset:
        try:
            candidates = []
            for candidate in srcset.split(','):
                candidate = candidate.strip()
                if not candidate:
                    continue
                parts = candidate.split()
                if len(parts) < 2:
                    continue
                url = parts[0]
                descriptor = parts[1]
                # Normalize URL
                if url.startswith('//'):
                    url = 'https:' + url
                elif url.startswith('/'):
                    url = urljoin(base_url, url)
                # Parse descriptor
                width = 0
                if descriptor.endswith('x'):
                    try:
                        density = float(descriptor[:-1])
                        width = int(320 * density)
                    except ValueError:
                        continue
                elif descriptor.endswith('w'):
                    try:
                        width = int(descriptor[:-1])
                    except ValueError:
                        continue
                else:
                    width = 320
                candidates.append((url, width))
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
        except Exception:
            pass
    
    # 4. Fallback to src (lowest priority)
    src = el.get('src')
    if src:
        if src.startswith('//'):
            return 'https:' + src
        elif src.startswith('/'):
            return urljoin(base_url, src)
        return src
    
    return None


async def validate_image_request(url: str, client: httpx.AsyncClient, max_bytes: int, host_deny: Optional[Set[str]] = None) -> bool:
    """
    Validate image request with comprehensive guards and checks.
    
    Args:
        url: URL to validate
        client: HTTP client instance
        max_bytes: Maximum allowed content length
        host_deny: Set of denied hostnames
        
    Returns:
        True if URL is valid and safe, False otherwise
    """
    if host_deny is None:
        host_deny = set()
    
    try:
        parsed = urlparse(url)
        
        # Deny non-HTTP/HTTPS schemes
        if parsed.scheme.lower() not in {'http', 'https'}:
            return False
        
        # Deny hosts in deny list
        if parsed.netloc.lower() in {h.lower() for h in host_deny}:
            return False
        
        # Deny SVG files
        path_lower = parsed.path.lower()
        if path_lower.endswith('.svg'):
            return False
        
        # Manual redirect handling (max hops from constant)
        current_url = url
        for hop in range(MAX_REDIRECTS):
            try:
                # HEAD request first
                head_response = await client.head(current_url, timeout=HTTP_TIMEOUT, follow_redirects=False)
                
                # Handle redirects manually
                if head_response.status_code in REDIRECT_STATUS_CODES:
                    location = head_response.headers.get('location')
                    if not location:
                        return False
                    
                    # Validate redirect URL
                    redirect_parsed = urlparse(location)
                    if redirect_parsed.scheme.lower() not in {'http', 'https'}:
                        return False
                    if redirect_parsed.netloc.lower() in {h.lower() for h in host_deny}:
                        return False
                    
                    current_url = location
                    continue
                
                # Check content type
                content_type = head_response.headers.get('content-type', '').lower()
                if 'image/svg+xml' in content_type:
                    return False
                if not content_type.startswith('image/'):
                    return False
                
                # Check content length if present
                content_length = head_response.headers.get('content-length')
                if content_length and int(content_length) > max_bytes:
                    return False
                
                # If HEAD succeeds, do small GET to verify
                if head_response.status_code == 200:
                    get_response = await client.get(current_url, timeout=HTTP_TIMEOUT, follow_redirects=False)
                    if get_response.status_code in {200, 206}:  # 206 = Partial Content
                        # Check actual content length
                        actual_length = get_response.headers.get('content-length')
                        if actual_length and int(actual_length) > max_bytes:
                            return False
                        
                        # Check downloaded content length (streaming read with cap)
                        content_bytes = b''
                        for chunk in get_response.iter_bytes(chunk_size=CHUNK_SIZE):
                            content_bytes += chunk
                            if len(content_bytes) > max_bytes:
                                return False
                        
                        return True
                
                return False
                
            except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.RequestError):
                return False
        
        # Too many redirects
        return False
        
    except Exception:
        return False


# Compiled regex for background-image extraction (supports ", ', none)
_BACKGROUND_IMAGE_REGEX = re.compile(
    r'background-image\s*:\s*url\(["\']?([^"\')\s]+)["\']?\)',
    re.IGNORECASE
)

# BEM-ish class pattern: ^[a-z0-9]+(?:[-_]{1,2}[a-z0-9]+)+$
_BEM_CLASS_REGEX = re.compile(r'^[a-z0-9]+(?:[-_]{1,2}[a-z0-9]+)+$')


def extract_extra_sources(soup: BeautifulSoup, base_url: str) -> List[str]:
    """
    Extract extra image sources from HTML soup in ordered priority.
    
    Collects sources in this order:
    1. meta[property='og:image']::attr(content)
    2. link[rel='image_src']::attr(href)
    3. video::attr(poster)
    4. img::attr(srcset) / source::attr(srcset) / source::attr(data-srcset)
    5. script[type='application/ld+json'] JSON paths
    6. Inline style background-image
    
    Args:
        soup: BeautifulSoup parsed HTML
        base_url: Base URL for resolving relative URLs
        
    Returns:
        List of extra source selectors in priority order
    """
    extra_sources = []
    
    # 1. Meta og:image tags
    og_image_meta = soup.find('meta', property='og:image')
    if og_image_meta and og_image_meta.get('content'):
        extra_sources.append("meta[property='og:image']::attr(content)")
    
    # 2. Link image_src tags
    image_src_link = soup.find('link', rel='image_src')
    if image_src_link and image_src_link.get('href'):
        extra_sources.append("link[rel='image_src']::attr(href)")
    
    # 3. Video poster attributes
    video_with_poster = soup.find('video', poster=True)
    if video_with_poster:
        extra_sources.append("video::attr(poster)")
    
    # 4. Srcset attributes (img, source tags)
    # Check img tags with srcset
    img_with_srcset = soup.find('img', srcset=True)
    if img_with_srcset:
        extra_sources.append("img::attr(srcset)")
    
    # Check source tags with srcset
    source_with_srcset = soup.find('source', srcset=True)
    if source_with_srcset:
        extra_sources.append("source::attr(srcset)")
    
    # Check source tags with data-srcset
    source_with_data_srcset = soup.find('source', {'data-srcset': True})
    if source_with_data_srcset:
        extra_sources.append("source::attr(data-srcset)")
    
    # 5. JSON-LD script blocks
    jsonld_scripts = soup.find_all('script', type='application/ld+json')
    if jsonld_scripts:
        # Check if any contain image-related fields
        has_image_fields = False
        for script in jsonld_scripts:
            try:
                if script.string:
                    json_data = json.loads(script.string.strip())
                    # Check for image-related fields in JSON
                    json_objects = json_data if isinstance(json_data, list) else [json_data]
                    for obj in json_objects:
                        if isinstance(obj, dict):
                            if any(field in obj for field in ['thumbnailUrl', 'image']):
                                has_image_fields = True
                                break
                    if has_image_fields:
                        break
            except (json.JSONDecodeError, TypeError):
                continue
        
        if has_image_fields:
            extra_sources.append("script[type='application/ld+json']::jsonpath($.thumbnailUrl, $.image, $..thumbnailUrl, $..image, $..url)")
    
    # 6. Inline style background-image
    elements_with_bg = soup.find_all(style=True)
    has_background_image = False
    for element in elements_with_bg:
        style = element.get('style', '')
        if _BACKGROUND_IMAGE_REGEX.search(style):
            has_background_image = True
            break
    
    if has_background_image:
        extra_sources.append("::style(background-image)")
    
    return extra_sources


def stable_selector(node: Tag, max_depth: int = 4) -> str:
    """
    Generate a minimal, stable CSS selector for a node with BEM class preservation.
    
    Preserves classes matching BEM-ish regex: ^[a-z0-9]+(?:[-_]{1,2}[a-z0-9]+)+$
    Avoids IDs, :nth-child, and random tokens.
    Prefers forms like .list-global__thumb img, picture img, .thumb img.
    
    Args:
        node: BeautifulSoup Tag element
        max_depth: Maximum depth to traverse up the DOM tree
        
    Returns:
        Stable CSS selector string
    """
    if not node or not node.name:
        return ""
    
    path_parts = []
    current = node
    
    # Build path from element up to root (max_depth levels)
    level = 0
    while current and current.name and level < max_depth:
        # Skip document and html elements to avoid overly long selectors
        if current.name in ['[document]', 'html']:
            current = current.parent
            continue
        
        selector_part = current.name
        
        # Check for BEM-ish classes (prefer over IDs)
        if current.get('class'):
            classes = current.get('class')
            # Find BEM-ish classes
            bem_classes = [cls for cls in classes if _BEM_CLASS_REGEX.match(cls)]
            
            if bem_classes:
                # Use the first BEM class found
                selector_part += f'.{bem_classes[0]}'
            else:
                # Fallback to non-random classes if no BEM classes found
                # Inline random token detection
                def is_random_token(token):
                    if len(token) < 3:
                        return False
                    random_patterns = [
                        r'^[a-f0-9]{8,}$',  # Hex strings
                        r'^[0-9]+$',        # Pure numbers
                        r'[A-Z]{3,}',       # Multiple caps (often random)
                        r'_[a-f0-9]{6,}_',  # Underscore hex patterns
                        r'^[a-z]{1,2}[0-9]{4,}$',  # Short letters + long numbers
                        r'^[0-9]{4,}[a-z]{1,2}$',  # Long numbers + short letters
                    ]
                    return any(re.search(pattern, token) for pattern in random_patterns)
                
                non_random_classes = [
                    cls for cls in classes 
                    if not is_random_token(cls) and len(cls) > 2
                ]
                if non_random_classes:
                    selector_part += f'.{non_random_classes[0]}'
        
        path_parts.insert(0, selector_part)
        current = current.parent
        level += 1
    
    return ' '.join(path_parts)




def gather_evidence(nodes: List[Tag]) -> Dict[str, int]:
    """
    Gather evidence from nodes in one pass for scoring.
    
    Args:
        nodes: List of BeautifulSoup Tag elements to analyze
        
    Returns:
        Dictionary with evidence counts: repeats, has_duration, has_video_href, class_hits, srcset_count
    """
    evidence = {
        'repeats': len(nodes),
        'has_duration': 0,
        'has_video_href': 0,
        'class_hits': 0,
        'srcset_count': 0
    }
    
    # Duration regex pattern
    duration_pattern = re.compile(r'\b\d{1,2}:\d{2}\b')
    
    # Video-related URL patterns
    video_url_patterns = [
        re.compile(r'/(video|watch|v|id)/', re.IGNORECASE),
        re.compile(r'embed/', re.IGNORECASE),
        re.compile(r'player/', re.IGNORECASE)
    ]
    
    # Positive class name hints for image content
    positive_class_hints = {
        'thumb', 'thumbnail', 'preview', 'poster', 'image', 'img', 
        'pic', 'picture', 'cover', 'artwork', 'screenshot', 'still',
        'frame', 'snapshot', 'gallery', 'media', 'visual', 'display'
    }
    
    # Negative class name hints (likely not target images)
    negative_class_hints = {
        'icon', 'logo', 'avatar', 'profile', 'social', 'share', 'button',
        'nav', 'menu', 'header', 'footer', 'sidebar', 'banner', 'ad',
        'sponsor', 'sponsored', 'advertisement', 'promo', 'promotion'
    }
    
    for tag in nodes:
        # Check for duration text in nearby elements (siblings, parents)
        has_duration = False
        for element in [tag] + list(tag.parents)[:3]:
            # Check element's own text content
            if element.string and duration_pattern.search(element.string):
                has_duration = True
                break
            
            # Check all text content in element
            if element.get_text() and duration_pattern.search(element.get_text()):
                has_duration = True
                break
            
            # Check siblings for duration text
            if element.parent:
                for sibling in element.parent.children:
                    if hasattr(sibling, 'get_text'):
                        if sibling.get_text() and duration_pattern.search(sibling.get_text()):
                            has_duration = True
                            break
                    elif hasattr(sibling, 'string') and sibling.string:
                        if duration_pattern.search(str(sibling)):
                            has_duration = True
                            break
                if has_duration:
                    break
        
        if has_duration:
            evidence['has_duration'] += 1
        
        # Check for video-related URL in ancestor links
        has_video_href = False
        for ancestor in tag.parents:
            if ancestor.name == 'a' and ancestor.get('href'):
                href = ancestor.get('href')
                for pattern in video_url_patterns:
                    if pattern.search(href):
                        has_video_href = True
                        break
                if has_video_href:
                    break
            
            # Also check for .go anchors (common video site pattern)
            if ancestor.name == 'a' and ancestor.get('class'):
                classes = ancestor.get('class')
                if any('go' in cls.lower() for cls in classes):
                    has_video_href = True
                    break
        
        if has_video_href:
            evidence['has_video_href'] += 1
        
        # Count class hints (positive - negative)
        all_classes = []
        current = tag
        level = 0
        while current and level < 3:  # Check up to 3 levels
            if current.get('class'):
                all_classes.extend(current.get('class'))
            current = current.parent
            level += 1
        
        positive_count = 0
        negative_count = 0
        for class_name in all_classes:
            if any(hint in class_name.lower() for hint in positive_class_hints):
                positive_count += 1
            if any(hint in class_name.lower() for hint in negative_class_hints):
                negative_count += 1
        
        evidence['class_hits'] += max(0, positive_count - negative_count)
        
        # Check srcset richness
        if tag.get('srcset'):
            srcset_entries = len(tag.get('srcset', '').split(','))
            if srcset_entries > 1:
                evidence['srcset_count'] += 1
    
    return evidence


def score_candidate(repeats: int, has_duration: int, has_video_href: int, class_hits: int, srcset_count: int) -> float:
    """
    Score a candidate selector based on evidence.
    
    Weights:
    - repeats: +2 if ≥12 else +1 if ≥6
    - duration: +2 if ≥3
    - video_href: +2 if ≥3
    - class_hits: +1 if ≥3
    - srcset: +1 if ≥5
    
    Args:
        repeats: Number of times selector matches
        has_duration: Number of nodes with duration text nearby
        has_video_href: Number of nodes with video-related ancestor links
        class_hits: Net positive class hints (positive - negative)
        srcset_count: Number of nodes with rich srcset
        
    Returns:
        Score between 0.0 and 1.0
    """
    score = 0.0
    
    # Repetition scoring
    if repeats >= 12:
        score += 2.0
    elif repeats >= 6:
        score += 1.0
    else:
        score += repeats / 12.0  # Gradual scaling for lower counts
    
    # Duration scoring
    if has_duration >= 3:
        score += 2.0
    elif has_duration > 0:
        score += has_duration / 3.0  # Gradual scaling
    
    # Video href scoring
    if has_video_href >= 3:
        score += 2.0
    elif has_video_href > 0:
        score += has_video_href / 3.0  # Gradual scaling
    
    # Class hits scoring
    if class_hits >= 3:
        score += 1.0
    elif class_hits > 0:
        score += class_hits / 3.0  # Gradual scaling
    
    # Srcset scoring
    if srcset_count >= 5:
        score += 1.0
    elif srcset_count > 0:
        score += srcset_count / 5.0  # Gradual scaling
    
    # Normalize to [0, 1] range (max possible score is 8.0)
    return max(0.0, min(1.0, score / 8.0))


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


def discover_listing_links(soup: BeautifulSoup, base_url: str, same_host: str) -> List[str]:
    """Extract likely listing pages based on anchor element scoring.
    
    Args:
        soup: BeautifulSoup parsed HTML
        base_url: Base URL for resolving relative links
        same_host: Hostname to filter same-domain links
        
    Returns:
        List of top 5 listing page URLs, sorted by score (desc) then stable-sorted
    """
    import re
    from urllib.parse import urljoin, urlparse
    
    # Scoring patterns
    positive_text_patterns = re.compile(
        r'\b(trending|popular|hot|new|latest|top|videos|all videos|browse|most viewed|most recent|explore|watch)\b',
        re.IGNORECASE
    )
    
    positive_href_patterns = re.compile(
        r'/(videos|video|new|latest|popular|trending|top)/|page=1',
        re.IGNORECASE
    )
    
    parent_class_patterns = re.compile(
        r'\b(nav|menu|tabs|filter|sort|category|pill)\b',
        re.IGNORECASE
    )
    
    negative_patterns = re.compile(
        r'\b(login|signup|join|premium|faq|blog|ad|ads|sponsor|out/|redirect|go\.php)\b',
        re.IGNORECASE
    )
    
    scored_links = []
    
    # Iterate through all anchor elements
    for link in soup.find_all('a', href=True):
        href = link.get('href', '').strip()
        if not href:
            continue
            
        # Skip javascript: and fragment-only links
        if href.startswith('javascript:') or href.startswith('#'):
            continue
            
        # Resolve to absolute URL
        try:
            absolute_url = urljoin(base_url, href)
            parsed = urlparse(absolute_url)
        except Exception:
            continue
            
        # Only keep same-domain, http/https URLs
        if parsed.netloc != same_host or parsed.scheme not in {'http', 'https'}:
            continue
            
        # Skip if URL has fragment
        if parsed.fragment:
            continue
            
        # Calculate score
        score = 0
        
        # Get link text and aria-label
        text = link.get_text(strip=True)
        aria_label = link.get('aria-label', '')
        combined_text = f"{text} {aria_label}".strip()
        
        # +3 for positive text patterns
        if positive_text_patterns.search(combined_text):
            score += 3
            
        # +2 for positive href patterns
        if positive_href_patterns.search(href):
            score += 2
            
        # +1 for parent with navigation classes
        parent = link.parent
        while parent and parent.name != 'body':
            parent_classes = ' '.join(parent.get('class', []))
            if parent_class_patterns.search(parent_classes):
                score += 1
                break
            parent = parent.parent
            
        # -3 for negative patterns
        if negative_patterns.search(combined_text) or negative_patterns.search(href):
            score -= 3
            
        # Only keep links with positive scores
        if score > 0:
            scored_links.append((score, absolute_url, text))
    
    # Remove duplicates (by URL)
    seen_urls = set()
    unique_links = []
    for score, url, text in scored_links:
        if url not in seen_urls:
            seen_urls.add(url)
            unique_links.append((score, url, text))
    
    # Sort by score (desc) then by URL (stable sort)
    unique_links.sort(key=lambda x: (-x[0], x[1]))
    
    # Return top 5 URLs
    return [url for score, url, text in unique_links[:5]]


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


