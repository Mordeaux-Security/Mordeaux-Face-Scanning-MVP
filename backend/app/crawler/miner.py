"""
Selector Mining Module

Consolidates selector_miner functionality for automatic CSS selector discovery
and site recipe generation. Provides intelligent selector mining with validation.
"""

import asyncio
import logging
import os
import random
import re
import time
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup

from .config import CrawlerConfig
from .http_service import HTTPService, get_http_service
from .js_rendering_service import JSRenderingService, get_js_rendering_service

logger = logging.getLogger(__name__)

# Discovery hop constants
DISCOVERY_TEXT = r"(trending|popular|hot|new|latest|top|videos|browse|most viewed|most recent|explore|watch)"
DISCOVERY_HREF = r"/(videos?|new|latest|popular|trending|top)(/|$)|[?&]page=1\b"
ADVERSE_TEXT = r"(login|signup|join|premium|blog|faq|ads?|sponsor|out/|redirect|go\.php)"
MIN_SELECTORS_FOR_DISCOVERY = 2  # Minimum selectors needed to skip discovery
MIN_SELECTORS_FOR_PAGINATION = 1  # Minimum selectors needed to skip pagination

# Ad/redirector denial constants
AD_CONTAINER_PATTERNS = [
    r"ad\b",
    r"ads\b", 
    r"sponsor\b",
    r"promoted\b",
    r"exo-native-widget\b"
]

AD_HOST_DENYLIST = {
    "doubleclick.net",
    "exoclick.com", 
    "s.magsrv.com",
    "afcdn.net"
}

# Stable selector constants
BEM_PATTERN = re.compile(r"^[a-z0-9]+(?:[-_]{1,2}[a-z0-9]+)+$")
STABLE_SELECTOR_MAX_DEPTH = 4
STABLE_SELECTOR_ALLOWED_TAGS = {"img", "picture", "a", "div", "figure"}

# Attribute priority constants
ATTR_PRIORITY = ("data-src", "data-srcset", "srcset", "src")
SRCSET_WIDTH_PATTERN = re.compile(r"(\S+)\s+(\d+)w")

# Multi-kind classification constants
ALBUM_KEYWORDS = ["album", "gallery", "photos", "pics", "collection", "portfolio"]
GALLERY_KEYWORDS = ["gallery", "photos", "images", "pictures"]
KIND_VIDEO_GRID = "video_grid"
KIND_ALBUM_GRID = "album_grid"
KIND_GALLERY_IMAGES = "gallery_images"

# YAML emission constants
CANONICAL_ATTRIBUTES = ["data-src", "data-srcset", "srcset", "src"]
YAML_KEY_ORDER = ["selectors", "attributes_priority", "extra_sources", "method", "confidence"]

# Extra sources fallback patterns
EXTRA_SOURCES_PATTERNS = [
    "meta[property='og:image']::attr(content)",
    "link[rel='image_src']::attr(href)",
    "video::attr(poster)",
    "script[type='application/ld+json']",
    "[style*='background-image']",
    # Lazy loading patterns
    "img[data-src]",
    "img[data-lazy-src]",
    "img[data-original]",
    "img[loading='lazy']",
    "img[data-srcset]",
    "img[data-lazy]",
    # Common image containers
    "div.image img",
    "div.photo img",
    "div.pic img",
    "figure img",
    "picture img"
]


@dataclass
class CandidateSelector:
    """Represents a candidate CSS selector with metadata."""
    selector: str
    description: str
    evidence: Dict[str, float]
    sample_urls: List[str]
    repetition_count: int
    score: float = 0.0
    kind: str = "video_grid"


@dataclass
class MiningResult:
    """Result of selector mining operation."""
    candidates: List[CandidateSelector]
    status: str  # "OK", "NO_THUMBS_STATIC", "EXTRA_ONLY"
    stats: Dict[str, int]
    mining_time: float
    success: bool
    error: Optional[str] = None


class SelectorMiner:
    """
    Mines CSS selectors for image extraction from web pages.
    
    Analyzes HTML structure to discover patterns and generate site-specific
    recipes for image extraction with validation and scoring.
    """
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.http_service: Optional[HTTPService] = None
        self.js_service: Optional[JSRenderingService] = None
        
        # Mining configuration
        self.max_candidates = 15  # Increased for better selector discovery
        self.max_samples_per_candidate = 5  # More samples for validation
        self.max_bytes = config.max_image_bytes
        self.timeout_seconds = config.timeout_seconds
        
        # Recipe file path
        self.recipes_file = "site_recipes.yaml"
        
        # JavaScript rendering tracking
        self.js_render_count = 0  # Track JS renders per run
        self.js_render_max = config.js_render_max_per_run
        
        # Caching and retry configuration
        self.url_cache: Dict[str, str] = {}  # URL -> HTML content cache
        self.cache_max_size = 50  # Maximum number of URLs to cache
        self.retry_max_attempts = 2  # Max retries for transient errors
        self.retry_jitter_min = 150  # Minimum jitter in milliseconds
        self.retry_jitter_max = 400  # Maximum jitter in milliseconds
        
        # URL validation
        self.host_deny = {
            "doubleclick.net", "exoclick.com", "s.magsrv.com", "afcdn.net",
            "google-analytics.com", "googletagmanager.com", "facebook.com", "twitter.com"
        }
        
        # Gallery detection patterns
        self.gallery_patterns = [
            r'/gallery/', r'/album/', r'/photos/', r'/images/',
            r'/collection/', r'/portfolio/', r'/showcase/', r'/media/',
            r'/pictures/', r'/pics/', r'album=', r'gallery=', r'photos=', r'images='
        ]
        
        # Forum detection patterns
        self.forum_patterns = [
            r'/forum/', r'/forums/', r'/thread/', r'/threads/',
            r'/topic/', r'/topics/', r'/post/', r'/posts/',
            r'/discussion/', r'/discussions/', r'/board/', r'/boards/',
            r'/community/', r'/message/', r'/messages/', r'forum=', r'thread='
        ]
        
        # Quality indicators
        self.quality_indicators = {
            'high': ['high-res', 'highres', 'hd', 'full', 'original', 'large', 'xl', 'xxl'],
            'medium': ['medium', 'med', 'standard', 'normal', 'regular'],
            'low': ['thumb', 'thumbnail', 'small', 'mini', 'preview', 'low-res', 'lowres']
        }
        
        # Mining statistics
        self.stats = {
            'sites_mined': 0,
            'selectors_found': 0,
            'validated_selectors': 0,
            'mining_time': 0.0,
            'errors': 0
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_http_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
    
    async def _initialize_http_client(self):
        """Initialize HTTP service and JS rendering service for mining operations."""
        if self.http_service is None:
            self.http_service = await get_http_service(self.config)
        if self.js_service is None:
            self.js_service = await get_js_rendering_service(self.config)
    
    async def _cleanup(self):
        """Cleanup HTTP service."""
        # Note: HTTP service cleanup is handled globally
        pass
    
    async def mine_site(self, url: str) -> MiningResult:
        """
        Mine selectors for a site.
        
        Args:
            url: URL to mine selectors for
            
        Returns:
            MiningResult with discovered selectors and metadata
        """
        start_time = time.time()
        
        try:
            # Fetch page content with caching and retry
            page_content = await self._fetch_page_with_retry(url)
            if not page_content:
                return MiningResult(
                    candidates=[],
                    status="NO_THUMBS_STATIC",
                    stats={},
                    mining_time=time.time() - start_time,
                    success=False,
                    error="Failed to fetch page content"
                )
            
            # Parse HTML
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Extract hostname for same-domain discovery
            parsed_url = urlparse(url)
            same_host = parsed_url.netloc
            
            # Mine selectors with discovery hop fallback
            candidates = await self._mine_with_discovery(url, soup, same_host)
            
            # Validate candidates
            validated_candidates = await self._validate_candidates(candidates)
            
            # Determine status
            if len(validated_candidates) > 0:
                status = "OK"
            else:
                status = "NO_THUMBS_STATIC"
            
            mining_time = time.time() - start_time
            
            # Update statistics
            self.stats['sites_mined'] += 1
            self.stats['selectors_found'] += len(candidates)
            self.stats['validated_selectors'] += len(validated_candidates)
            self.stats['mining_time'] += mining_time
            
            logger.info(f"Mined {len(validated_candidates)} selectors for {url} in {mining_time:.2f}s")
            
            return MiningResult(
                candidates=validated_candidates,
                status=status,
                stats={
                    'total_candidates': len(candidates),
                    'validated_candidates': len(validated_candidates),
                    'discovery_used': len(candidates) > 0 and len(candidates) < MIN_SELECTORS_FOR_DISCOVERY
                },
                mining_time=mining_time,
                success=True
            )  
        except Exception as e:
            logger.error(f"Error mining selectors for {url}: {e}")
            self.stats['errors'] += 1
            return MiningResult(
                candidates=[],
                status="NO_THUMBS_STATIC",
                stats={},
                mining_time=time.time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content using HTTP service."""
        if not self.http_service:
            await self._initialize_http_client()
        
        # Use HTTP service to fetch page content (disable caching for fresh content)
        content, status_code = await self.http_service.get(url, as_text=True, use_cache=False)
        
        if content is None:
            logger.warning(f"Failed to fetch {url}: {status_code}")
            return None
        
        return content
    
    def _get_cached_content(self, url: str) -> Optional[str]:
        """
        Get cached content for a URL.
        
        Args:
            url: URL to check in cache
            
        Returns:
            Cached HTML content or None if not cached
        """
        return self.url_cache.get(url)
    
    def _cache_content(self, url: str, content: str) -> None:
        """
        Cache content for a URL with size management.
        
        Args:
            url: URL to cache
            content: HTML content to cache
        """
        # If cache is full, remove oldest entries (simple FIFO)
        if len(self.url_cache) >= self.cache_max_size:
            # Remove first (oldest) entry
            oldest_url = next(iter(self.url_cache))
            del self.url_cache[oldest_url]
        
        self.url_cache[url] = content
        logger.debug(f"Cached content for {url} (cache size: {len(self.url_cache)})")
    
    def _is_transient_error(self, error: Exception) -> bool:
        """
        Check if an error is transient and should be retried.
        
        Args:
            error: Exception to check
            
        Returns:
            True if error is transient and should be retried
        """
        # Transient errors that should be retried
        transient_errors = (
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
        )
        
        # Check if it's a transient error
        if isinstance(error, transient_errors):
            return True
        
        # Check for specific HTTP status codes that might be transient
        if isinstance(error, httpx.HTTPStatusError):
            # 5xx server errors are typically transient
            if 500 <= error.response.status_code < 600:
                return True
            # 429 Too Many Requests might be transient
            if error.response.status_code == 429:
                return True
        
        return False
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """
        Calculate retry delay with jitter.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff base delay
        base_delay = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s, etc.
        
        # Add jitter
        jitter = random.uniform(self.retry_jitter_min, self.retry_jitter_max) / 1000.0  # Convert to seconds
        
        return base_delay + jitter
    
    async def _fetch_page_with_retry(self, url: str) -> Optional[str]:
        """
        Fetch page with caching and light retry for transient errors.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed
        """
        # Check cache first
        cached_content = self._get_cached_content(url)
        if cached_content is not None:
            logger.debug(f"Using cached content for {url}")
            return cached_content
        
        # Fetch with retry logic
        last_error = None
        
        for attempt in range(self.retry_max_attempts + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    # Calculate delay for retry
                    delay = self._calculate_retry_delay(attempt - 1)
                    logger.info(f"Retrying {url} (attempt {attempt + 1}/{self.retry_max_attempts + 1}) after {delay:.2f}s delay")
                    await asyncio.sleep(delay)
                
                # Attempt to fetch the page
                content = await self._fetch_page(url)
                
                if content is not None:
                    # Cache successful content
                    self._cache_content(url, content)
                    return content
                else:
                    # If _fetch_page returns None, it's likely a non-transient error
                    logger.warning(f"Failed to fetch {url} (non-transient error)")
                    return None
                    
            except Exception as e:
                last_error = e
                
                # Check if this is a transient error that should be retried
                if self._is_transient_error(e):
                    if attempt < self.retry_max_attempts:
                        logger.warning(f"Transient error fetching {url} (attempt {attempt + 1}): {e}")
                        continue  # Retry
                    else:
                        logger.error(f"Max retries exceeded for {url}: {e}")
                        return None
                else:
                    # Non-transient error, don't retry
                    logger.error(f"Non-transient error fetching {url}: {e}")
                    return None
        
        # If we get here, all retries failed
        logger.error(f"Failed to fetch {url} after {self.retry_max_attempts + 1} attempts: {last_error}")
        return None
    
    def _detect_forum_site(self, url: str, soup: BeautifulSoup) -> Tuple[bool, float]:
        """Detect if a site is a forum."""
        confidence = 0.0
        
        # Check URL patterns
        url_lower = url.lower()
        for pattern in self.forum_patterns:
            if re.search(pattern, url_lower):
                confidence += 0.3
                break
        
        # Check HTML structure indicators
        html_text = soup.get_text().lower()
        forum_indicators = [
            'forum', 'thread', 'topic', 'post', 'discussion', 'board', 'community',
            'message', 'reply', 'comment', 'author', 'member', 'user', 'avatar'
        ]
        
        for indicator in forum_indicators:
            if indicator in html_text:
                confidence += 0.05
        
        # Check for forum-specific elements
        forum_selectors = [
            '.forum', '.thread', '.topic', '.post', '.discussion', '.board',
            '.message', '.reply', '.comment', '.thread-list', '.topic-list'
        ]
        
        for selector in forum_selectors:
            if soup.select(selector):
                confidence += 0.1
        
        is_forum = confidence >= 0.3
        return is_forum, min(confidence, 1.0)
    
    def _detect_gallery_site(self, url: str, soup: BeautifulSoup) -> bool:
        """Detect if a site is a gallery/album site."""
        # Check URL patterns
        url_lower = url.lower()
        for pattern in self.gallery_patterns:
            if re.search(pattern, url_lower):
                return True
        
        # Check HTML structure
        gallery_selectors = [
            '.gallery', '.album', '.photos', '.images', '.collection',
            '.portfolio', '.showcase', '.media-grid', '.photo-grid'
        ]
        
        for selector in gallery_selectors:
            if soup.select(selector):
                return True
        
        return False
    
    async def _mine_general_selectors(self, soup: BeautifulSoup, base_url: str) -> List[CandidateSelector]:
        """Mine selectors for general sites."""
        candidates = []
        
        # Common container pattern keywords - look for partial matches
        container_keywords = [
            'thumb', 'thumbnail', 'video', 'item', 'gallery', 'media', 'content', 
            'post', 'block', 'grid', 'card', 'tile', 'photo', 'image', 'picture', 
            'pic', 'img', 'slide', 'slider', 'carousel', 'banner', 'ad', 'advertisement',
            'preview', 'cover', 'featured', 'main', 'hero', 'header', 'avatar', 
            'profile', 'user', 'member', 'author', 'album', 'collection', 'portfolio',
            'showcase', 'feet', 'face', 'head', 'body', 'leg', 'foot'
        ]
        
        # Find all elements with classes that contain our keywords
        all_elements = soup.find_all(attrs={'class': True})
        keyword_matches = {}
        
        for element in all_elements:
            # Skip elements in ad containers
            if self._is_in_ad_container(element):
                continue
                
            classes = element.get('class', [])
            for cls in classes:
                cls_lower = cls.lower()
                for keyword in container_keywords:
                    if keyword in cls_lower:
                        if keyword not in keyword_matches:
                            keyword_matches[keyword] = []
                        keyword_matches[keyword].append(element)
        
        # Process matches for each keyword
        for keyword, elements in keyword_matches.items():
            if len(elements) >= 3:  # Minimum threshold
                # Group elements by their full class name for better selectors
                class_groups = {}
                for element in elements:
                    classes = ' '.join(element.get('class', []))
                    if classes not in class_groups:
                        class_groups[classes] = []
                    class_groups[classes].append(element)
                
                # Create selectors for each class group
                for classes, group_elements in class_groups.items():
                    if len(group_elements) >= 2:  # At least 2 elements with same classes
                        # Find images within these elements
                        all_images = []
                        for element in group_elements:
                            images = element.find_all(['img', 'source', 'video'])
                            # Filter out images in ad containers
                            for img in images:
                                if not self._is_in_ad_container(img):
                                    all_images.append(img)
                        
                        if all_images:
                            # Try to generate a stable selector from the first element
                            stable_selector = self._generate_stable_selector(group_elements[0])
                            if stable_selector:
                                selector = stable_selector
                            else:
                                # Fallback to class-based selector
                                selector = f".{'.'.join(classes.split())}"
                            
                            # Classify the selector kind
                            selector_kind = self._classify_selector_kind(group_elements[0], base_url, soup)
                            
                            candidates.append(CandidateSelector(
                                selector=selector,
                                description=f"Keyword '{keyword}': {len(group_elements)} containers with {selector}",
                                evidence={'repeats': len(group_elements), 'keyword': keyword},
                                sample_urls=self._extract_sample_urls(all_images, base_url),
                                repetition_count=len(group_elements),
                                score=self._score_selector(selector, group_elements, soup),
                                kind=selector_kind
                            ))
        
        return candidates
    
    async def _mine_forum_selectors(self, soup: BeautifulSoup, base_url: str) -> List[CandidateSelector]:
        """Mine selectors for forum sites."""
        candidates = []
        
        # Forum-specific patterns
        forum_patterns = [
            '.post img', '.message img', '.thread img', '.topic img',
            '.reply img', '.comment img', '.attachment img', '.user-avatar img'
        ]
        
        for pattern in forum_patterns:
            elements = soup.select(pattern)
            # Filter out elements in ad containers
            filtered_elements = [elem for elem in elements if not self._is_in_ad_container(elem)]
            if len(filtered_elements) >= 2:
                # Try to generate a stable selector from the first element
                stable_selector = self._generate_stable_selector(filtered_elements[0])
                if stable_selector:
                    selector = stable_selector
                else:
                    # Fallback to pattern-based selector
                    selector = pattern
                
                # Classify the selector kind
                selector_kind = self._classify_selector_kind(filtered_elements[0], base_url, soup)
                
                candidates.append(CandidateSelector(
                    selector=selector,
                    description=f"Forum: {len(filtered_elements)} images in {pattern}",
                    evidence={'repeats': len(filtered_elements), 'forum_specific': True},
                    sample_urls=self._extract_sample_urls(filtered_elements, base_url),
                    repetition_count=len(filtered_elements),
                    score=self._score_selector(selector, filtered_elements, soup),
                    kind=selector_kind
                ))
        
        return candidates
    
    async def _mine_gallery_selectors(self, soup: BeautifulSoup, base_url: str) -> List[CandidateSelector]:
        """Mine selectors for gallery/album sites."""
        candidates = []
        
        # Gallery-specific patterns
        gallery_patterns = [
            '.gallery img', '.album img', '.photos img', '.images img',
            '.collection img', '.portfolio img', '.showcase img',
            '.media-grid img', '.photo-grid img', '.image-grid img',
            '.gallery-item img', '.photo-item img', '.image-item img',
            '.album-item img', '.collection-item img', '.portfolio-item img'
        ]
        
        for pattern in gallery_patterns:
            elements = soup.select(pattern)
            # Filter out elements in ad containers
            filtered_elements = [elem for elem in elements if not self._is_in_ad_container(elem)]
            if len(filtered_elements) >= 3:
                # Try to generate a stable selector from the first element
                stable_selector = self._generate_stable_selector(filtered_elements[0])
                if stable_selector:
                    selector = stable_selector
                else:
                    # Fallback to pattern-based selector
                    selector = pattern
                
                # Classify the selector kind
                selector_kind = self._classify_selector_kind(filtered_elements[0], base_url, soup)
                
                candidates.append(CandidateSelector(
                    selector=selector,
                    description=f"Gallery: {len(filtered_elements)} images in {pattern}",
                    evidence={'repeats': len(filtered_elements), 'gallery_specific': True},
                    sample_urls=self._extract_sample_urls(filtered_elements, base_url),
                    repetition_count=len(filtered_elements),
                    score=self._score_selector(selector, filtered_elements, soup),
                    kind=selector_kind
                ))
        
        return candidates
    
    def _generate_selector(self, element) -> Optional[str]:
        """Generate CSS selector for an element."""
        try:
            if not element or not element.name:
                return None
            
            # Start with tag name
            selector = element.name
            
            # Add class if present and not random
            if element.get('class'):
                classes = element.get('class')
                # Find non-random classes
                non_random_classes = []
                for cls in classes:
                    if len(cls) > 2 and not self._is_random_class(cls):
                        non_random_classes.append(cls)
                
                if non_random_classes:
                    selector += f'.{non_random_classes[0]}'
            
            return selector
            
        except Exception as e:
            logger.debug(f"Error generating selector: {e}")
            return None
    
    def _is_random_class(self, cls: str) -> bool:
        """Check if a class name appears to be random/generated."""
        if len(cls) < 3:
            return True
        
        random_patterns = [
            r'^[a-f0-9]{8,}$',  # Hex strings
            r'^[0-9]+$',        # Pure numbers
            r'[A-Z]{3,}',       # Multiple caps
            r'_[a-f0-9]{6,}_',  # Underscore hex patterns
            r'^[a-z]{1,2}[0-9]{4,}$',  # Short letters + long numbers
            r'^[0-9]{4,}[a-z]{1,2}$',  # Long numbers + short letters
        ]
        
        return any(re.search(pattern, cls) for pattern in random_patterns)
    
    def _extract_sample_urls(self, elements: List, base_url: str) -> List[str]:
        """Extract sample URLs from elements."""
        urls = []
        seen_urls = set()
        
        for element in elements:
            url = self._resolve_image_url(element, base_url)
            if url and url not in seen_urls and len(urls) < 12:
                urls.append(url)
                seen_urls.add(url)
        
        return urls
    
    def _resolve_image_url(self, element, base_url: str) -> Optional[str]:
        """Resolve image URL from element using attribute priority and srcset parsing."""
        return self._pick_best_url(element, base_url)
    
    def _score_selector(self, selector: str, elements: List, soup: BeautifulSoup) -> float:
        """Advanced scoring algorithm for selectors based on multiple criteria."""
        try:
            score = 0.0
            
            # 1. Element count scoring (logarithmic scale)
            element_count = len(elements)
            if element_count >= 3:
                score += min(1.0, 0.3 + (element_count - 3) * 0.05)
            else:
                score += element_count * 0.1
            
            # 2. Selector specificity scoring
            specificity_score = self._calculate_specificity(selector)
            score += specificity_score * 0.2
            
            # 3. Pattern recognition scoring
            pattern_score = self._score_patterns(selector)
            score += pattern_score * 0.25
            
            # 4. Content analysis scoring
            content_score = self._analyze_content_quality(elements, soup)
            score += content_score * 0.15
            
            # 5. URL validation scoring
            url_score = self._validate_extracted_urls(elements)
            score += url_score * 0.1
            
            # 6. Consistency scoring
            consistency_score = self._check_selector_consistency(selector, elements)
            score += consistency_score * 0.1
            
            # 7. Penalty for overly complex selectors
            if selector.count('.') > 4 or selector.count('#') > 2:
                score -= 0.2
            
            # 8. Bonus for semantic selectors
            semantic_bonus = self._calculate_semantic_bonus(selector)
            score += semantic_bonus
            
            return max(0.0, min(1.0, score))
            
        except Exception as e:
            logger.error(f"Error scoring selector {selector}: {e}")
            return 0.0
    
    def _calculate_specificity(self, selector: str) -> float:
        """Calculate CSS selector specificity score."""
        try:
            # Count different types of selectors
            id_count = selector.count('#')
            class_count = selector.count('.')
            tag_count = len([s for s in selector.split() if s and not s.startswith(('.', '#', '['))])
            attr_count = selector.count('[')
            
            # Calculate specificity (higher is more specific)
            specificity = (id_count * 100) + (class_count * 10) + tag_count + (attr_count * 5)
            
            # Normalize to 0-1 scale (prefer moderate specificity)
            if specificity == 0:
                return 0.0
            elif specificity <= 20:
                return 0.8  # Good specificity
            elif specificity <= 50:
                return 0.6  # Moderate specificity
            else:
                return 0.3  # Too specific
            
        except Exception:
            return 0.5
    
    def _score_patterns(self, selector: str) -> float:
        """Score selector based on recognized patterns."""
        score = 0.0
        selector_lower = selector.lower()
        
        # High-value patterns
        high_value_patterns = {
            'thumb': 0.4, 'thumbnail': 0.4, 'gallery': 0.3, 'album': 0.3,
            'photo': 0.3, 'image': 0.2, 'media': 0.2, 'content': 0.2,
            'item': 0.2, 'card': 0.2, 'tile': 0.2, 'grid': 0.2
        }
        
        for pattern, value in high_value_patterns.items():
            if pattern in selector_lower:
                score += value
        
        # Medium-value patterns
        medium_value_patterns = {
            'post': 0.1, 'entry': 0.1, 'article': 0.1, 'section': 0.1,
            'container': 0.1, 'wrapper': 0.1, 'box': 0.1
        }
        
        for pattern, value in medium_value_patterns.items():
            if pattern in selector_lower:
                score += value
        
        # Penalty patterns
        penalty_patterns = {
            'ad': -0.3, 'advertisement': -0.3, 'banner': -0.2, 'sidebar': -0.2,
            'footer': -0.2, 'header': -0.1, 'nav': -0.1, 'menu': -0.1
        }
        
        for pattern, penalty in penalty_patterns.items():
            if pattern in selector_lower:
                score += penalty
        
        return max(0.0, min(1.0, score))
    
    def _analyze_content_quality(self, elements: List, soup: BeautifulSoup) -> float:
        """Analyze the quality of content in elements."""
        try:
            if not elements:
                return 0.0
            
            score = 0.0
            total_images = 0
            valid_images = 0
            
            for element in elements:
                # Count images within elements
                images = element.find_all(['img', 'source', 'video'])
                total_images += len(images)
                
                for img in images:
                    # Check for valid image attributes
                    if any(attr in img.attrs for attr in ['src', 'data-src', 'data-srcset']):
                        valid_images += 1
                        
                        # Check image dimensions
                        width = img.get('width') or img.get('data-width')
                        height = img.get('height') or img.get('data-height')
                        
                        if width and height:
                            try:
                                w, h = int(width), int(height)
                                if w >= 100 and h >= 100:  # Minimum size
                                    score += 0.1
                            except (ValueError, TypeError):
                                pass
            
            # Ratio of valid images
            if total_images > 0:
                valid_ratio = valid_images / total_images
                score += valid_ratio * 0.5
            
            return min(1.0, score)
            
        except Exception as e:
            logger.error(f"Error analyzing content quality: {e}")
            return 0.0
    
    def _validate_extracted_urls(self, elements: List) -> float:
        """Validate URLs extracted from elements."""
        try:
            if not elements:
                return 0.0
            
            total_urls = 0
            valid_urls = 0
            
            for element in elements:
                images = element.find_all(['img', 'source', 'video'])
                for img in images:
                    for attr in ['src', 'data-src', 'data-srcset']:
                        url = img.get(attr)
                        if url:
                            total_urls += 1
                            if self._is_valid_image_url(url):
                                valid_urls += 1
            
            if total_urls == 0:
                return 0.0
            
            return valid_urls / total_urls
            
        except Exception as e:
            logger.error(f"Error validating URLs: {e}")
            return 0.0
    
    def _is_valid_image_url(self, url: str) -> bool:
        """Check if URL is a valid image URL."""
        try:
            from urllib.parse import urlparse
            
            parsed = urlparse(url)
            
            # Must have valid scheme
            if parsed.scheme not in ['http', 'https']:
                return False
            
            # Must have hostname
            if not parsed.hostname:
                return False
            
            # Check for image extensions
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg'}
            path_lower = parsed.path.lower()
            
            if any(path_lower.endswith(ext) for ext in image_extensions):
                return True
            
            # Check for image indicators in path
            image_indicators = ['image', 'img', 'photo', 'pic', 'thumb', 'gallery']
            if any(indicator in path_lower for indicator in image_indicators):
                return True
            
            return False
            
        except Exception:
            return False
    
    def _check_selector_consistency(self, selector: str, elements: List) -> float:
        """Check consistency of selector results."""
        try:
            if len(elements) < 2:
                return 0.5  # Neutral for single elements
            
            # Check if elements have similar structure
            structures = []
            for element in elements:
                structure = self._get_element_structure(element)
                structures.append(structure)
            
            # Calculate consistency
            if len(set(structures)) == 1:
                return 1.0  # Perfect consistency
            elif len(set(structures)) <= len(structures) * 0.7:
                return 0.7  # Good consistency
            else:
                return 0.3  # Poor consistency
            
        except Exception as e:
            logger.error(f"Error checking selector consistency: {e}")
            return 0.5
    
    def _get_element_structure(self, element) -> str:
        """Get a simplified structure representation of an element."""
        try:
            # Get tag name and key attributes
            tag = element.name or 'unknown'
            classes = ' '.join(sorted(element.get('class', [])))
            id_attr = element.get('id', '')
            
            return f"{tag}:{classes}:{id_attr}"
        except Exception:
            return "unknown"
    
    def _calculate_semantic_bonus(self, selector: str) -> float:
        """Calculate bonus for semantically meaningful selectors."""
        try:
            bonus = 0.0
            selector_lower = selector.lower()
            
            # Semantic patterns that indicate good selectors
            semantic_patterns = {
                'gallery-item': 0.2, 'photo-item': 0.2, 'image-item': 0.2,
                'thumb-item': 0.2, 'media-item': 0.15, 'content-item': 0.15,
                'post-image': 0.15, 'article-image': 0.15, 'entry-image': 0.15
            }
            
            for pattern, value in semantic_patterns.items():
                if pattern in selector_lower:
                    bonus += value
            
            return min(0.3, bonus)  # Cap at 0.3
            
        except Exception:
            return 0.0
    
    async def _validate_candidates(self, candidates: List[CandidateSelector]) -> List[CandidateSelector]:
        """Validate candidates by checking their sample URLs."""
        validated_candidates = []
        
        for candidate in candidates:
            valid_urls = []
            
            # Validate sample URLs
            for url in candidate.sample_urls[:self.max_samples_per_candidate]:
                if await self._validate_image_url(url):
                    valid_urls.append(url)
            
            # Only keep candidates with at least one valid URL
            if valid_urls:
                candidate.sample_urls = valid_urls
                candidate.repetition_count = len(valid_urls)
                validated_candidates.append(candidate)
        
        # Sort by score and limit
        validated_candidates.sort(key=lambda x: x.score, reverse=True)
        return validated_candidates[:self.max_candidates]
    
    async def _validate_image_url(self, url: str) -> bool:
        """Validate image URL."""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme.lower() not in {'http', 'https'}:
                return False
            
            # Check host
            if parsed.netloc.lower() in {h.lower() for h in self.host_deny}:
                return False
            
            # Check for ad hosts
            if self._is_ad_host(url):
                return False
            
            # Check for SVG files
            if parsed.path.lower().endswith('.svg'):
                return False
            
            # Quick HEAD request to validate
            if self.http_client:
                try:
                    response = await self.http_client.head(url, timeout=5.0)
                    if response.status_code == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        return content_type.startswith('image/')
                except:
                    pass
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating URL {url}: {e}")
            return False
    
    def _discover_listing_links(self, soup: BeautifulSoup, base_url: str, same_host: str) -> List[str]:
        """
        Discover internal links likely to be listing pages.
        
        Args:
            soup: BeautifulSoup object of the current page
            base_url: Base URL of the current page
            same_host: Hostname to match for same-domain links
            
        Returns:
            List of up to 5 URLs sorted by discovery score
        """
        from urllib.parse import urljoin, urlparse
        
        candidates = []
        
        for a in soup.select("a[href]"):
            # Get link text (prefer aria-label, fallback to text content)
            text = (a.get("aria-label") or a.get_text(" ")).strip().lower()
            href = a["href"]
            
            # Convert to absolute URL
            url = urljoin(base_url, href)
            parsed = urlparse(url)
            
            # Skip if not same host or invalid scheme
            if parsed.scheme not in ("http", "https") or parsed.netloc != same_host:
                continue
            
            # Calculate discovery score
            score = 0
            
            # Positive indicators
            if re.search(DISCOVERY_TEXT, text, flags=re.I):
                score += 3
            if re.search(DISCOVERY_HREF, href, flags=re.I):
                score += 2
            
            # Negative indicators
            if re.search(ADVERSE_TEXT, text, flags=re.I):
                score -= 3
            
            # Only include positive scoring links
            if score > 0:
                candidates.append((score, url))
        
        # Stable sort: highest score first, then by URL for consistency
        # Remove duplicates by URL while preserving highest score
        unique_candidates = {}
        for score, url in candidates:
            if url not in unique_candidates or score > unique_candidates[url]:
                unique_candidates[url] = score
        
        # Sort by score (descending), then by URL (ascending) for stability
        sorted_candidates = sorted(unique_candidates.items(), key=lambda kv: (-kv[1], kv[0]))
        
        # Return top 5 URLs
        return [url for url, _ in sorted_candidates[:5]]
    
    def _get_next_page_url(self, url: str) -> Optional[str]:
        """
        Generate next page URL using simple pagination heuristics.
        
        Args:
            url: Current page URL
            
        Returns:
            Next page URL or None if no pagination pattern found
        """
        # Simple heuristic: page=N to page=N+1
        match = re.search(r"([?&])page=(\d+)", url)
        if match:
            page_num = int(match.group(2)) + 1
            next_url = re.sub(r"([?&])page=\d+", fr"\1page={page_num}", url, count=1)
            return next_url
        return None
    
    def _find_next_page_link(self, soup: BeautifulSoup, base_url: str, same_host: str) -> Optional[str]:
        """
        Find next page link by looking for common pagination patterns.
        
        Args:
            soup: BeautifulSoup object of the current page
            base_url: Base URL of the current page
            same_host: Hostname to match for same-domain links
            
        Returns:
            Next page URL or None if no next link found
        """
        from urllib.parse import urljoin, urlparse
        
        # Common next page link patterns
        next_patterns = [
            r"next",
            r"more",
            r"continue",
            r"→",
            r"»",
            r"page\s*2",
            r"page\s*next"
        ]
        
        for a in soup.select("a[href]"):
            text = (a.get("aria-label") or a.get_text(" ")).strip().lower()
            href = a["href"]
            
            # Check if link text matches next page patterns
            for pattern in next_patterns:
                if re.search(pattern, text, flags=re.I):
                    # Convert to absolute URL
                    url = urljoin(base_url, href)
                    parsed = urlparse(url)
                    
                    # Only return same-host links
                    if parsed.scheme in ("http", "https") and parsed.netloc == same_host:
                        return url
        
        return None
    
    def _is_in_ad_container(self, element) -> bool:
        """
        Check if an element is inside an ad container.
        
        Args:
            element: BeautifulSoup element to check
            
        Returns:
            True if element is inside an ad container, False otherwise
        """
        # Check current element's classes and IDs
        current_classes = element.get('class', [])
        current_id = element.get('id', '')
        
        # Check current element
        for pattern in AD_CONTAINER_PATTERNS:
            # Check classes
            for class_name in current_classes:
                if re.search(pattern, class_name, flags=re.I):
                    return True
            
            # Check ID
            if current_id and re.search(pattern, current_id, flags=re.I):
                return True
        
        # Check data attributes that might indicate ads
        for attr_name, attr_value in element.attrs.items():
            if attr_name.startswith('data-') and isinstance(attr_value, str):
                for pattern in AD_CONTAINER_PATTERNS:
                    if re.search(pattern, attr_value, flags=re.I):
                        return True
        
        # Check parent elements
        parent = element.parent
        while parent and parent.name:  # Stop at document root
            parent_classes = parent.get('class', [])
            parent_id = parent.get('id', '')
            
            for pattern in AD_CONTAINER_PATTERNS:
                # Check parent classes
                for class_name in parent_classes:
                    if re.search(pattern, class_name, flags=re.I):
                        return True
                
                # Check parent ID
                if parent_id and re.search(pattern, parent_id, flags=re.I):
                    return True
            
            # Check parent data attributes
            for attr_name, attr_value in parent.attrs.items():
                if attr_name.startswith('data-') and isinstance(attr_value, str):
                    for pattern in AD_CONTAINER_PATTERNS:
                        if re.search(pattern, attr_value, flags=re.I):
                            return True
            
            parent = parent.parent
        
        return False
    
    def _is_ad_host(self, url: str) -> bool:
        """
        Check if a URL is from a known ad host.
        
        Args:
            url: URL to check
            
        Returns:
            True if URL is from an ad host, False otherwise
        """
        try:
            parsed = urlparse(url)
            hostname = parsed.netloc.lower()
            
            # Check exact matches
            if hostname in AD_HOST_DENYLIST:
                return True
            
            # Check subdomain matches
            for ad_host in AD_HOST_DENYLIST:
                if hostname.endswith('.' + ad_host):
                    return True
            
            return False
        except Exception:
            return False
    
    def _generate_stable_selector(self, node) -> Optional[str]:
        """
        Generate a stable CSS selector using BEM methodology and depth limits.
        
        Args:
            node: BeautifulSoup element to generate selector for
            
        Returns:
            Stable CSS selector or None if no valid selector can be generated
        """
        parts = []
        n = node
        
        # Walk up the DOM tree, collecting BEM classes and allowed tags
        while n and len(parts) < STABLE_SELECTOR_MAX_DEPTH:
            # Look for BEM-like classes first
            classes = n.get("class", [])
            bem_classes = [c for c in classes if BEM_PATTERN.match(c)]
            
            if bem_classes:
                # Use the first BEM class found
                parts.append("." + bem_classes[0])
            elif n.name in STABLE_SELECTOR_ALLOWED_TAGS:
                # Use the tag name if it's an allowed tag
                parts.append(n.name)
            
            n = n.parent
        
        # Reverse to get the correct order (from root to target)
        parts = list(reversed(parts))
        
        # Ensure the selector ends with img
        if not parts or parts[-1] != "img":
            parts.append("img")
        
        # Take only the last 4 parts to limit depth
        final_parts = parts[-STABLE_SELECTOR_MAX_DEPTH:]
        
        # Join with spaces to create the final selector
        selector = " ".join(final_parts)
        
        # Validate the selector is not empty and has reasonable length
        if len(selector.strip()) > 0 and len(selector) < 200:
            return selector
        
        return None
    
    def _parse_srcset(self, srcset_value: str) -> Optional[str]:
        """
        Parse srcset attribute and return the URL with the largest width.
        
        Args:
            srcset_value: The srcset attribute value
            
        Returns:
            URL with the largest width or None if parsing fails
        """
        if not srcset_value:
            return None
        
        try:
            # Parse srcset entries (URL width pairs)
            entries = []
            for match in SRCSET_WIDTH_PATTERN.finditer(srcset_value):
                url = match.group(1)
                width = int(match.group(2))
                entries.append((url, width))
            
            if not entries:
                # Fallback: if no width descriptors, return the first URL
                first_url = srcset_value.split(',')[0].strip().split()[0]
                return first_url if first_url else None
            
            # Sort by width (descending) and return the largest
            entries.sort(key=lambda x: x[1], reverse=True)
            return entries[0][0]
            
        except (ValueError, IndexError):
            # Fallback: return the first URL if parsing fails
            first_url = srcset_value.split(',')[0].strip().split()[0]
            return first_url if first_url else None
    
    def _pick_best_url(self, element, base_url: str) -> Optional[str]:
        """
        Pick the best URL from an element using attribute priority and srcset parsing.
        
        Args:
            element: BeautifulSoup element to extract URL from
            base_url: Base URL for resolving relative URLs
            
        Returns:
            Best URL or None if no valid URL found
        """
        from urllib.parse import urljoin
        
        # Check attributes in priority order
        for attr in ATTR_PRIORITY:
            value = element.get(attr)
            if not value:
                continue
            
            # Handle srcset attributes specially
            if "srcset" in attr:
                url = self._parse_srcset(value)
            else:
                url = value
            
            if url:
                try:
                    return urljoin(base_url, url)
                except Exception:
                    continue
        
        return None
    
    def _extract_extra_sources(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract image URLs from extra sources when CSS selectors are insufficient.
        
        Args:
            soup: BeautifulSoup object of the page
            base_url: Base URL for resolving relative URLs
            
        Returns:
            List of image URLs found from extra sources
        """
        extra_urls = []
        
        try:
            # 1. Open Graph meta tags
            og_images = soup.find_all('meta', property='og:image')
            for meta in og_images:
                content = meta.get('content')
                if content:
                    url = self._resolve_url(content, base_url)
                    if url and self._is_valid_image_url(url):
                        extra_urls.append(url)
            
            # 2. Link tags with image_src rel
            image_links = soup.find_all('link', rel='image_src')
            for link in image_links:
                href = link.get('href')
                if href:
                    url = self._resolve_url(href, base_url)
                    if url and self._is_valid_image_url(url):
                        extra_urls.append(url)
            
            # 3. Video poster attributes
            videos = soup.find_all('video')
            for video in videos:
                poster = video.get('poster')
                if poster:
                    url = self._resolve_url(poster, base_url)
                    if url and self._is_valid_image_url(url):
                        extra_urls.append(url)
            
            # 4. JSON-LD structured data
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    import json
                    data = json.loads(script.string or '{}')
                    urls = self._extract_urls_from_json_ld(data)
                    for url in urls:
                        resolved_url = self._resolve_url(url, base_url)
                        if resolved_url and self._is_valid_image_url(resolved_url):
                            extra_urls.append(resolved_url)
                except (json.JSONDecodeError, AttributeError):
                    continue
            
            # 5. Background images from style attributes
            elements_with_bg = soup.find_all(attrs={'style': True})
            for element in elements_with_bg:
                style = element.get('style', '')
                urls = self._extract_background_image_urls(style)
                for url in urls:
                    resolved_url = self._resolve_url(url, base_url)
                    if resolved_url and self._is_valid_image_url(resolved_url):
                        extra_urls.append(resolved_url)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in extra_urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            return unique_urls
            
        except Exception as e:
            logger.warning(f"Error extracting extra sources: {e}")
            return []
    
    def _extract_urls_from_json_ld(self, data: Dict[str, Any]) -> List[str]:
        """
        Extract image URLs from JSON-LD structured data using multiple JSONPath fallbacks.
        
        Args:
            data: JSON-LD data structure
            
        Returns:
            List of image URLs found in the JSON-LD data
        """
        urls = []
        
        def extract_from_value(value):
            if isinstance(value, str):
                if self._looks_like_image_url(value):
                    urls.append(value)
            elif isinstance(value, dict):
                for key, val in value.items():
                    if key in ['image', 'thumbnailUrl', 'contentUrl', 'url']:
                        extract_from_value(val)
                    else:
                        extract_from_value(val)
            elif isinstance(value, list):
                for item in value:
                    extract_from_value(item)
        
        # Common JSON-LD patterns for images
        patterns = [
            '$.image',
            '$.thumbnailUrl', 
            '$.contentUrl',
            '$.associatedMedia[*].contentUrl',
            '$..image',
            '$..thumbnailUrl',
            '$..contentUrl',
            '$.mainEntity.image',
            '$.itemListElement[*].item.image'
        ]
        
        # Extract from the data structure
        extract_from_value(data)
        
        return urls
    
    def _extract_background_image_urls(self, style: str) -> List[str]:
        """
        Extract image URLs from CSS background-image properties.
        
        Args:
            style: CSS style string
            
        Returns:
            List of image URLs found in background-image properties
        """
        urls = []
        
        # Pattern to match background-image: url(...)
        import re
        pattern = r'background-image\s*:\s*url\s*\(\s*["\']?([^"\')\s]+)["\']?\s*\)'
        matches = re.findall(pattern, style, re.IGNORECASE)
        
        for match in matches:
            if self._looks_like_image_url(match):
                urls.append(match)
        
        return urls
    
    def _looks_like_image_url(self, url: str) -> bool:
        """
        Check if a URL looks like an image URL.
        
        Args:
            url: URL to check
            
        Returns:
            True if the URL appears to be an image
        """
        if not url or not isinstance(url, str):
            return False
        
        # Common image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp', '.svg', '.tiff', '.ico'}
        
        # Check file extension
        url_lower = url.lower()
        if any(url_lower.endswith(ext) for ext in image_extensions):
            return True
        
        # Check for common image URL patterns
        image_patterns = [
            '/image/', '/img/', '/photo/', '/picture/', '/thumb/', '/thumbnail/',
            'image=', 'img=', 'photo=', 'picture=', 'thumb=', 'thumbnail='
        ]
        
        return any(pattern in url_lower for pattern in image_patterns)
    
    def _resolve_url(self, url: str, base_url: str) -> Optional[str]:
        """
        Resolve a relative URL against a base URL.
        
        Args:
            url: URL to resolve (may be relative)
            base_url: Base URL for resolution
            
        Returns:
            Resolved absolute URL or None if invalid
        """
        try:
            from urllib.parse import urljoin
            return urljoin(base_url, url)
        except Exception:
            return None
    
    async def _fetch_with_js_rendering_capped(self, url: str) -> Optional[str]:
        """
        Fetch page with JavaScript rendering, respecting caps and timeout.
        
        Args:
            url: URL to fetch
            
        Returns:
            HTML content or None if failed or cap exceeded
        """
        # Check if JS rendering is allowed
        if not self.config.js_render_allowed:
            logger.debug(f"JS rendering not allowed for {url}")
            return None
        
        # Check if we've exceeded the JS render cap
        if self.js_render_count >= self.js_render_max:
            logger.info(f"JS render cap exceeded ({self.js_render_count}/{self.js_render_max}), skipping JS rendering for {url}")
            return None
        
        # Increment JS render counter
        self.js_render_count += 1
        logger.info(f"Attempting JS rendering for {url} ({self.js_render_count}/{self.js_render_max})")
        
        try:
            # Use JS rendering service
            if not self.js_service:
                await self._initialize_http_client()
            
            rendered_content, status_code = await self.js_service.render_page(url)
            
            if rendered_content is None:
                logger.warning(f"JS rendering failed for {url}: {status_code}")
                return None
            
            logger.info(f"JS rendering successful for {url}")
            return rendered_content.html
                
        except Exception as e:
            logger.error(f"Error with capped JS rendering for {url}: {e}")
            return None
    
    
    def _classify_selector_kind(self, element, url: str, soup: BeautifulSoup) -> str:
        """
        Classify the selector kind based on content structure and context.
        
        Args:
            element: BeautifulSoup element to classify
            url: Current page URL
            soup: BeautifulSoup object of the current page
            
        Returns:
            Selector kind: 'album_grid', 'gallery_images', or 'video_grid'
        """
        # Check if we're on an album page with many images
        all_images = soup.find_all(['img', 'picture'])
        if len(all_images) >= 20:  # Threshold for gallery page
            # Check if URL or page content suggests gallery
            url_lower = url.lower()
            page_text = soup.get_text().lower()
            
            for keyword in GALLERY_KEYWORDS:
                if keyword in url_lower or keyword in page_text:
                    return KIND_GALLERY_IMAGES
        
        # Check ancestor classes for album keywords
        current = element
        while current and current.name:
            classes = current.get('class', [])
            for class_name in classes:
                class_lower = class_name.lower()
                for keyword in ALBUM_KEYWORDS:
                    if keyword in class_lower:
                        return KIND_ALBUM_GRID
            
            current = current.parent
        
        # Check element's own classes
        element_classes = element.get('class', [])
        for class_name in element_classes:
            class_lower = class_name.lower()
            for keyword in ALBUM_KEYWORDS:
                if keyword in class_lower:
                    return KIND_ALBUM_GRID
        
        # Check URL for album indicators
        url_lower = url.lower()
        for keyword in ALBUM_KEYWORDS:
            if keyword in url_lower:
                return KIND_ALBUM_GRID
        
        # Check page content for album indicators
        page_text = soup.get_text().lower()
        for keyword in ALBUM_KEYWORDS:
            if keyword in page_text:
                return KIND_ALBUM_GRID
        
        # Default to video_grid
        return KIND_VIDEO_GRID
    
    async def _mine_with_discovery(self, url: str, soup: BeautifulSoup, same_host: str) -> List[CandidateSelector]:
        """
        Mine selectors with discovery hop fallback.
        
        Args:
            url: Original URL
            soup: BeautifulSoup object of the original page
            same_host: Hostname for same-domain discovery
            
        Returns:
            List of candidate selectors
        """
        # First, try mining the original page
        is_forum, forum_confidence = self._detect_forum_site(url, soup)
        is_gallery = self._detect_gallery_site(url, soup)
        
        if is_forum:
            candidates = await self._mine_forum_selectors(soup, url)
        elif is_gallery:
            candidates = await self._mine_gallery_selectors(soup, url)
        else:
            candidates = await self._mine_general_selectors(soup, url)
        
        # If we found enough selectors, return them
        if len(candidates) >= MIN_SELECTORS_FOR_DISCOVERY:
            logger.info(f"Found {len(candidates)} selectors on original page, skipping discovery")
            return candidates
        
        # If selectors are thin, try extra sources fallback
        if len(candidates) < MIN_SELECTORS_FOR_DISCOVERY:
            logger.info(f"Found only {len(candidates)} selectors, trying extra sources fallback")
            extra_urls = self._extract_extra_sources(soup, url)
            if extra_urls:
                # Create a fallback selector for extra sources
                extra_selector = CandidateSelector(
                    selector="meta[property='og:image']::attr(content), link[rel='image_src']::attr(href), video::attr(poster), script[type='application/ld+json'], [style*='background-image']",
                    description=f"Extra sources fallback: {len(extra_urls)} URLs from meta tags, links, videos, JSON-LD, and background images",
                    evidence={'extra_sources': True, 'url_count': len(extra_urls)},
                    sample_urls=extra_urls[:5],  # Limit sample URLs
                    repetition_count=len(extra_urls),
                    score=0.7,  # Lower score for fallback sources
                    kind=KIND_VIDEO_GRID  # Default to video_grid for extra sources
                )
                candidates.append(extra_selector)
                logger.info(f"Added extra sources fallback with {len(extra_urls)} URLs")
        
        # If we now have enough candidates, return them
        if len(candidates) >= MIN_SELECTORS_FOR_DISCOVERY:
            logger.info(f"Found {len(candidates)} selectors after extra sources, skipping discovery")
            return candidates
        
        # If selectors are still thin and JS rendering is allowed, try JS rendering
        if len(candidates) < MIN_SELECTORS_FOR_DISCOVERY and self.config.js_render_allowed:
            logger.info(f"Found only {len(candidates)} selectors, trying JS rendering on original URL")
            js_content = await self._fetch_with_js_rendering_capped(url)
            if js_content:
                js_soup = BeautifulSoup(js_content, 'html.parser')
                js_candidates = await self._mine_general_selectors(js_soup, url)
                if js_candidates:
                    candidates.extend(js_candidates)
                    logger.info(f"JS rendering yielded {len(js_candidates)} additional selectors")
        
        # If we now have enough candidates, return them
        if len(candidates) >= MIN_SELECTORS_FOR_DISCOVERY:
            logger.info(f"Found {len(candidates)} selectors after JS rendering, skipping discovery")
            return candidates
        
        # Otherwise, try discovery hop
        logger.info(f"Found only {len(candidates)} selectors, attempting discovery hop")
        
        discovery_links = self._discover_listing_links(soup, url, same_host)
        if not discovery_links:
            logger.info("No discovery links found")
            return candidates
        
        logger.info(f"Found {len(discovery_links)} discovery links: {discovery_links[:3]}...")
        
        # Try each discovery link
        for discovery_url in discovery_links:
            try:
                logger.info(f"Trying discovery link: {discovery_url}")
                
                # Fetch the discovery page with caching and retry
                discovery_content = await self._fetch_page_with_retry(discovery_url)
                if not discovery_content:
                    continue
                
                discovery_soup = BeautifulSoup(discovery_content, 'html.parser')
                
                # Mine selectors from discovery page
                if is_forum:
                    discovery_candidates = await self._mine_forum_selectors(discovery_soup, discovery_url)
                elif is_gallery:
                    discovery_candidates = await self._mine_gallery_selectors(discovery_soup, discovery_url)
                else:
                    discovery_candidates = await self._mine_general_selectors(discovery_soup, discovery_url)
                
                # If we found good selectors, use them
                if len(discovery_candidates) > len(candidates):
                    logger.info(f"Discovery page yielded {len(discovery_candidates)} selectors (vs {len(candidates)} from original)")
                    candidates = discovery_candidates
                    
                    # If we have enough now, stop trying more discovery links
                    if len(candidates) >= MIN_SELECTORS_FOR_DISCOVERY:
                        break
                
                # If still not enough selectors, try pagination hop (one page only)
                if len(discovery_candidates) < MIN_SELECTORS_FOR_PAGINATION:
                    logger.info(f"Discovery page has only {len(discovery_candidates)} selectors, trying pagination hop")
                    
                    # Try pagination hop
                    pagination_candidates = await self._try_pagination_hop(
                        discovery_url, discovery_soup, same_host, is_forum, is_gallery
                    )
                    
                    if len(pagination_candidates) > len(candidates):
                        logger.info(f"Pagination hop yielded {len(pagination_candidates)} selectors (vs {len(candidates)} from discovery)")
                        candidates = pagination_candidates
                        
                        # If we have enough now, stop trying more discovery links
                        if len(candidates) >= MIN_SELECTORS_FOR_DISCOVERY:
                            break
                
                # If still not enough selectors and JS rendering is allowed, try JS on discovery URL
                if len(candidates) < MIN_SELECTORS_FOR_DISCOVERY and self.config.js_render_allowed:
                    logger.info(f"Trying JS rendering on discovery URL: {discovery_url}")
                    js_content = await self._fetch_with_js_rendering_capped(discovery_url)
                    if js_content:
                        js_soup = BeautifulSoup(js_content, 'html.parser')
                        if is_forum:
                            js_candidates = await self._mine_forum_selectors(js_soup, discovery_url)
                        elif is_gallery:
                            js_candidates = await self._mine_gallery_selectors(js_soup, discovery_url)
                        else:
                            js_candidates = await self._mine_general_selectors(js_soup, discovery_url)
                        
                        if js_candidates and len(js_candidates) > len(candidates):
                            logger.info(f"JS rendering on discovery URL yielded {len(js_candidates)} selectors")
                            candidates = js_candidates
                            
                            # If we have enough now, stop trying more discovery links
                            if len(candidates) >= MIN_SELECTORS_FOR_DISCOVERY:
                                break
                
            except Exception as e:
                logger.warning(f"Error mining discovery link {discovery_url}: {e}")
                continue
        
        return candidates
    
    async def _try_pagination_hop(self, current_url: str, current_soup: BeautifulSoup, 
                                 same_host: str, is_forum: bool, is_gallery: bool) -> List[CandidateSelector]:
        """
        Try pagination hop to find more selectors.
        
        Args:
            current_url: Current page URL
            current_soup: BeautifulSoup object of the current page
            same_host: Hostname for same-domain links
            is_forum: Whether this is a forum site
            is_gallery: Whether this is a gallery site
            
        Returns:
            List of candidate selectors from pagination hop
        """
        # Try URL-based pagination first (page=N to page=N+1)
        next_url = self._get_next_page_url(current_url)
        
        if next_url:
            logger.info(f"Trying URL-based pagination: {next_url}")
            try:
                next_content = await self._fetch_page_with_retry(next_url)
                if next_content:
                    next_soup = BeautifulSoup(next_content, 'html.parser')
                    
                    # Mine selectors from next page
                    if is_forum:
                        next_candidates = await self._mine_forum_selectors(next_soup, next_url)
                    elif is_gallery:
                        next_candidates = await self._mine_gallery_selectors(next_soup, next_url)
                    else:
                        next_candidates = await self._mine_general_selectors(next_soup, next_url)
                    
                    if next_candidates:
                        logger.info(f"URL-based pagination yielded {len(next_candidates)} selectors")
                        return next_candidates
            except Exception as e:
                logger.warning(f"Error with URL-based pagination {next_url}: {e}")
        
        # Try link-based pagination (find "Next" link)
        next_link_url = self._find_next_page_link(current_soup, current_url, same_host)
        
        if next_link_url:
            logger.info(f"Trying link-based pagination: {next_link_url}")
            try:
                next_content = await self._fetch_page_with_retry(next_link_url)
                if next_content:
                    next_soup = BeautifulSoup(next_content, 'html.parser')
                    
                    # Mine selectors from next page
                    if is_forum:
                        next_candidates = await self._mine_forum_selectors(next_soup, next_link_url)
                    elif is_gallery:
                        next_candidates = await self._mine_gallery_selectors(next_soup, next_link_url)
                    else:
                        next_candidates = await self._mine_general_selectors(next_soup, next_link_url)
                    
                    if next_candidates:
                        logger.info(f"Link-based pagination yielded {len(next_candidates)} selectors")
                        return next_candidates
            except Exception as e:
                logger.warning(f"Error with link-based pagination {next_link_url}: {e}")
        
        logger.info("No pagination hop available or successful")
        return []
    
    def _normalize_attributes_priority(self, attributes: List[str]) -> List[str]:
        """
        Normalize attributes to canonical order.
        
        Args:
            attributes: List of attribute names
            
        Returns:
            Normalized list with only canonical attributes in correct order
        """
        # Filter to only canonical attributes and maintain order
        normalized = []
        for attr in CANONICAL_ATTRIBUTES:
            if attr in attributes:
                normalized.append(attr)
        return normalized
    
    def _create_canonical_site_recipe(self, candidates: List[CandidateSelector], 
                                    base_url: str) -> Dict[str, Any]:
        """
        Create a canonical site recipe from candidates.
        
        Args:
            candidates: List of validated candidate selectors
            base_url: Base URL for the site
            
        Returns:
            Canonical site recipe dictionary
        """
        # Sort candidates by score (descending) for consistent ordering
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        # Create selectors list with canonical format
        selectors = []
        for candidate in sorted_candidates:
            selector_dict = {
                "kind": candidate.kind,
                "css": candidate.selector,
                "description": candidate.description,
                "score": round(candidate.score, 3)  # Round to 3 decimal places for consistency
            }
            selectors.append(selector_dict)
        
        # Create canonical recipe with fixed key order
        recipe = {}
        for key in YAML_KEY_ORDER:
            if key == "selectors":
                recipe[key] = selectors
            elif key == "attributes_priority":
                recipe[key] = CANONICAL_ATTRIBUTES
            elif key == "extra_sources":
                recipe[key] = EXTRA_SOURCES_PATTERNS
            elif key == "method":
                recipe[key] = "mined"
            elif key == "confidence":
                recipe[key] = 1.0
        
        return recipe
    
    def _load_recipes_sync(self) -> Dict[str, Any]:
        """Load recipes synchronously."""
        try:
            if os.path.exists(self.recipes_file):
                with open(self.recipes_file, 'r', encoding='utf-8') as f:
                    import yaml
                    return yaml.safe_load(f) or {}
            return {
                'schema_version': 2,
                'sites': {}
            }
        except Exception as e:
            logger.error(f"Error loading recipes sync: {e}")
            return {
                'schema_version': 2,
                'sites': {}
            }
    
    def _save_recipe_canonical(self, domain: str, candidates: List[CandidateSelector], 
                             base_url: str) -> bool:
        """
        Save recipe in canonical, idempotent format.
        
        Args:
            domain: Domain name for the recipe
            candidates: List of validated candidate selectors
            base_url: Base URL for the site
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            # Load existing recipes
            recipes = self._load_recipes_sync()
            
            # Create canonical recipe
            canonical_recipe = self._create_canonical_site_recipe(candidates, base_url)
            
            # Replace (not append) the domain's block
            if 'sites' not in recipes:
                recipes['sites'] = {}
            recipes['sites'][domain] = canonical_recipe
            
            # Save with canonical formatting
            self._save_recipes_canonical(recipes)
            
            logger.info(f"Saved canonical recipe for {domain} with {len(candidates)} selectors")
            return True
            
        except Exception as e:
            logger.error(f"Error saving canonical recipe for {domain}: {e}")
            return False
    
    def _save_recipes_canonical(self, recipes: Dict[str, Any]) -> None:
        """
        Save recipes with canonical YAML formatting.
        
        Args:
            recipes: Recipes dictionary to save
        """
        import yaml
        from collections import OrderedDict
        
        # Create ordered dictionary to maintain key order
        ordered_recipes = OrderedDict()
        
        # Add schema version first
        ordered_recipes['schema_version'] = recipes.get('schema_version', 2)
        
        # Add defaults with canonical order
        if 'defaults' in recipes:
            ordered_defaults = OrderedDict()
            defaults = recipes['defaults']
            
            # Add defaults keys in canonical order
            for key in YAML_KEY_ORDER:
                if key in defaults:
                    ordered_defaults[key] = defaults[key]
            
            ordered_recipes['defaults'] = ordered_defaults
        
        # Add sites with sorted domain names for consistency
        if 'sites' in recipes:
            ordered_sites = OrderedDict()
            for domain in sorted(recipes['sites'].keys()):
                site_recipe = recipes['sites'][domain]
                ordered_site = OrderedDict()
                
                # Add site keys in canonical order
                for key in YAML_KEY_ORDER:
                    if key in site_recipe:
                        ordered_site[key] = site_recipe[key]
                
                ordered_sites[domain] = ordered_site
            
            ordered_recipes['sites'] = ordered_sites
        
        # Write with canonical formatting (convert OrderedDict to regular dict for clean YAML)
        clean_recipes = self._convert_ordered_dict_to_dict(ordered_recipes)
        
        with open(self.recipes_file, 'w', encoding='utf-8') as f:
            yaml.dump(clean_recipes, f, 
                     default_flow_style=False,
                     sort_keys=False,  # Maintain our custom order
                     allow_unicode=True,
                     width=120,
                     indent=2)
    
    def _convert_ordered_dict_to_dict(self, obj):
        """Convert OrderedDict objects to regular dicts for clean YAML output."""
        if isinstance(obj, dict):
            return {key: self._convert_ordered_dict_to_dict(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_ordered_dict_to_dict(item) for item in obj]
        else:
            return obj
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get mining statistics."""
        return self.stats.copy()
    
    def reset_statistics(self) -> None:
        """Reset mining statistics."""
        self.stats = {
            'sites_mined': 0,
            'selectors_found': 0,
            'validated_selectors': 0,
            'mining_time': 0.0,
            'errors': 0
        }
