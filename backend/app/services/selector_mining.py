"""
Selector Mining Service

Extracts core selector mining logic from tools/selector_miner.py
and provides it as a service for integration with the crawler.

This service mines CSS selectors for image extraction from HTML content,
with evidence-based scoring and validation.
"""

import re
import logging
import os
import json
from typing import List, Dict, Set, Tuple, Optional, Union, Any
from urllib.parse import urljoin, urlparse
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from bs4 import BeautifulSoup, Tag

from .http_service import HttpService, fetch_html_with_redirects

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
    candidates: List['CandidateSelector']
    status: str  # "OK", "NO_THUMBS_STATIC", "EXTRA_ONLY"
    stats: Dict[str, int]
    checked_urls: List[str] = field(default_factory=list)  # URLs that were checked during mining


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


def _detect_structural_bonus(nodes: List[Tag]) -> int:
    """
    Detect structural patterns that indicate grid/gallery layouts.
    
    Args:
        nodes: List of BeautifulSoup Tag elements to analyze
        
    Returns:
        Structural bonus score (0-5)
    """
    if len(nodes) < 3:
        return 0
    
    bonus = 0
    
    # Check for common parent containers
    parent_groups = {}
    for node in nodes:
        parent = node.parent
        if parent:
            parent_key = f"{parent.name}.{'.'.join(parent.get('class', []))}"
            if parent_key not in parent_groups:
                parent_groups[parent_key] = 0
            parent_groups[parent_key] += 1
    
    # Bonus for having many nodes with same parent structure
    max_parent_count = max(parent_groups.values()) if parent_groups else 0
    if max_parent_count >= len(nodes) * 0.8:  # 80% or more have same parent
        bonus += 2
    
    # Check for list structures (ul > li, ol > li)
    list_structures = 0
    for node in nodes:
        if node.parent and node.parent.name in ['li']:
            if node.parent.parent and node.parent.parent.name in ['ul', 'ol']:
                list_structures += 1
    
    if list_structures >= len(nodes) * 0.7:  # 70% or more in list structures
        bonus += 2
    
    # Check for grid-like class patterns
    grid_patterns = ['grid', 'masonry', 'gallery', 'list', 'row', 'column']
    grid_indicators = 0
    for node in nodes:
        # Check node's own classes
        classes = node.get('class', [])
        if any(pattern in ' '.join(classes).lower() for pattern in grid_patterns):
            grid_indicators += 1
            continue
        
        # Check parent classes
        if node.parent:
            parent_classes = node.parent.get('class', [])
            if any(pattern in ' '.join(parent_classes).lower() for pattern in grid_patterns):
                grid_indicators += 1
    
    if grid_indicators >= len(nodes) * 0.6:  # 60% or more have grid indicators
        bonus += 1
    
    return min(5, bonus)  # Cap at 5


def gather_evidence(nodes: List[Tag]) -> Dict[str, int]:
    """
    Gather evidence from nodes in one pass for scoring.
    
    Args:
        nodes: List of BeautifulSoup Tag elements to analyze
        
    Returns:
        Dictionary with evidence counts: repeats, has_duration, has_video_href, class_hits, srcset_count, structural_bonus
    """
    evidence = {
        'repeats': len(nodes),
        'has_duration': 0,
        'has_video_href': 0,
        'class_hits': 0,
        'srcset_count': 0,
        'structural_bonus': 0
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
    
    # Structural bonus detection
    evidence['structural_bonus'] = _detect_structural_bonus(nodes)
    
    return evidence


def score_candidate(repeats: int, has_duration: int, has_video_href: int, class_hits: int, srcset_count: int, structural_bonus: int = 0) -> float:
    """
    Score a candidate selector based on evidence.
    
    Weights:
    - repeats: +2 if ≥12 else +1 if ≥6
    - duration: +2 if ≥3
    - video_href: +2 if ≥3
    - class_hits: +1 if ≥3
    - srcset: +1 if ≥5
    - structural_bonus: +1.5 if ≥3 (for grid/gallery patterns)
    
    Args:
        repeats: Number of times selector matches
        has_duration: Number of nodes with duration text nearby
        has_video_href: Number of nodes with video-related ancestor links
        class_hits: Net positive class hints (positive - negative)
        srcset_count: Number of nodes with rich srcset
        structural_bonus: Bonus for structural patterns (grid/gallery detection)
        
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
    
    # Structural bonus scoring (for enhanced grid detection)
    if structural_bonus >= 3:
        score += 1.5  # Bonus for structural patterns
    elif structural_bonus > 0:
        score += structural_bonus / 3.0  # Gradual scaling
    
    # Normalize to [0, 1] range (max possible score is 9.5 with structural bonus)
    return max(0.0, min(1.0, score / 9.5))


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
        html: str, 
        url: str, 
        client: httpx.AsyncClient,
        limits: Optional[Limits] = None
    ) -> MinedResult:
        """
        Mine selectors from a single page.
        
        Args:
            html: HTML content of the page
            url: URL of the page
            client: HTTP client for validation
            limits: Configuration limits
            
        Returns:
            MinedResult with candidates and metadata
        """
        if limits is None:
            limits = Limits()
        
        return await mine_page(url, html, use_js=False, client=client, limits=limits)
    
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
