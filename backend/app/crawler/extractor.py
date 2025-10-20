"""
Image URL Extraction Module

Handles HTML parsing and image URL extraction using site-specific recipes.
Provides streaming extraction capabilities and comprehensive URL validation.
"""

import asyncio
import logging
import re
from typing import List, Dict, Any, Optional, AsyncIterator, Set, Tuple
from urllib.parse import urljoin, urlparse, parse_qs
from dataclasses import dataclass

import httpx
from bs4 import BeautifulSoup

from .config import CrawlerConfig

logger = logging.getLogger(__name__)


@dataclass
class ExtractedImage:
    """Represents an extracted image with metadata."""
    url: str
    selector: str
    attributes: Dict[str, str]
    context: Dict[str, Any]
    quality_score: float = 0.0


@dataclass
class ExtractionResult:
    """Result of image extraction operation."""
    images: List[ExtractedImage]
    total_found: int
    unique_urls: int
    extraction_time: float
    success: bool
    error: Optional[str] = None


class ImageExtractor:
    """
    Extracts image URLs from web pages using site-specific recipes.
    
    Provides streaming extraction capabilities and comprehensive URL validation
    to ensure only valid, safe image URLs are returned.
    """
    
    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.http_client: Optional[httpx.AsyncClient] = None
        self._url_cache: Set[str] = set()
        
        # URL validation patterns
        self._malicious_schemes = self.config.malicious_schemes
        self._suspicious_extensions = self.config.suspicious_extensions
        self._bait_query_keys = self.config.bait_query_keys
        self._blocked_hosts = self.config.blocked_hosts
        self._blocked_tlds = self.config.blocked_tlds
        
        # Image quality indicators
        self._quality_indicators = {
            'high': ['high-res', 'highres', 'hd', 'full', 'original', 'large', 'xl', 'xxl'],
            'medium': ['medium', 'med', 'standard', 'normal', 'regular'],
            'low': ['thumb', 'thumbnail', 'small', 'mini', 'preview', 'low-res', 'lowres']
        }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self._initialize_http_client()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self._cleanup()
    
    async def _initialize_http_client(self):
        """Initialize HTTP client with proper configuration."""
        if self.http_client is None:
            limits = httpx.Limits(
                max_connections=self.config.max_connections,
                max_keepalive_connections=self.config.max_keepalive_connections,
                keepalive_expiry=self.config.keepalive_expiry
            )
            
            self.http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=self.config.timeout_seconds,
                    pool=self.config.timeout_seconds
                ),
                limits=limits,
                follow_redirects=True,
                max_redirects=self.config.max_redirects,
                headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
            )
    
    async def _cleanup(self):
        """Cleanup HTTP client and resources."""
        if self.http_client:
            await self.http_client.aclose()
            self.http_client = None
        self._url_cache.clear()
    
    async def extract_images(self, url: str, recipe: Dict[str, Any]) -> ExtractionResult:
        """
        Extract image URLs from a web page using site-specific recipe.
        
        Args:
            url: URL to extract images from
            recipe: Site-specific extraction recipe
            
        Returns:
            ExtractionResult with extracted images and metadata
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Fetch page content
            page_content = await self._fetch_page(url)
            if not page_content:
                return ExtractionResult(
                    images=[],
                    total_found=0,
                    unique_urls=0,
                    extraction_time=asyncio.get_event_loop().time() - start_time,
                    success=False,
                    error="Failed to fetch page content"
                )
            
            # Parse HTML
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Extract images using recipe selectors
            images = []
            for selector_config in recipe.get('selectors', []):
                selector_images = await self._extract_with_selector(
                    soup, selector_config, url, recipe
                )
                images.extend(selector_images)
            
            # Extract from extra sources
            extra_images = await self._extract_extra_sources(soup, url, recipe)
            images.extend(extra_images)
            
            # Remove duplicates and validate URLs
            unique_images = self._deduplicate_and_validate(images)
            
            # Limit results
            limited_images = unique_images[:self.config.max_images]
            
            extraction_time = asyncio.get_event_loop().time() - start_time
            
            logger.info(f"Extracted {len(limited_images)} images from {url} in {extraction_time:.2f}s")
            
            return ExtractionResult(
                images=limited_images,
                total_found=len(images),
                unique_urls=len(unique_images),
                extraction_time=extraction_time,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error extracting images from {url}: {e}")
            return ExtractionResult(
                images=[],
                total_found=0,
                unique_urls=0,
                extraction_time=asyncio.get_event_loop().time() - start_time,
                success=False,
                error=str(e)
            )
    
    async def extract_images_streaming(self, url: str, recipe: Dict[str, Any]) -> AsyncIterator[ExtractedImage]:
        """
        Stream image URLs as they're discovered.
        
        Args:
            url: URL to extract images from
            recipe: Site-specific extraction recipe
            
        Yields:
            ExtractedImage objects as they're found
        """
        try:
            # Fetch page content
            page_content = await self._fetch_page(url)
            if not page_content:
                return
            
            # Parse HTML
            soup = BeautifulSoup(page_content, 'html.parser')
            
            # Stream images using recipe selectors
            for selector_config in recipe.get('selectors', []):
                async for image in self._extract_with_selector_streaming(
                    soup, selector_config, url, recipe
                ):
                    if self._validate_url(image.url):
                        yield image
            
            # Stream from extra sources
            async for image in self._extract_extra_sources_streaming(soup, url, recipe):
                if self._validate_url(image.url):
                    yield image
                    
        except Exception as e:
            logger.error(f"Error in streaming extraction from {url}: {e}")
    
    async def _fetch_page(self, url: str) -> Optional[str]:
        """Fetch page content with advanced retry logic, security validation, and JavaScript rendering fallback."""
        if not self.http_client:
            await self._initialize_http_client()
        
        # Validate URL security first
        is_safe, reason = self._validate_url_security(url)
        if not is_safe:
            logger.warning(f"Unsafe URL blocked: {url} (reason: {reason})")
            return None
        
        # Advanced retry logic with exponential backoff
        for attempt in range(self.config.max_retries):
            try:
                # Track redirects
                redirect_count = 0
                current_url = url
                visited_urls = set()
                
                while redirect_count < self.config.max_redirects:
                    if current_url in visited_urls:
                        logger.warning(f"Redirect loop detected for {url}")
                        break
                    visited_urls.add(current_url)
                    
                    response = await self.http_client.get(current_url)
                    
                    # Handle redirects
                    if response.status_code in {301, 302, 303, 307, 308}:
                        redirect_count += 1
                        location = response.headers.get('location')
                        if location:
                            current_url = self._resolve_redirect_url(current_url, location)
                            continue
                    
                    response.raise_for_status()
                    content = response.text
                    
                    # Check if page might need JavaScript rendering
                    if self._needs_js_rendering(content):
                        logger.info(f"Page appears to need JavaScript rendering, trying JS fallback for {current_url}")
                        js_content = await self._fetch_with_js_rendering(current_url)
                        if js_content:
                            return js_content
                    
                    return content
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Page not found: {url}")
                    return None
                elif e.response.status_code >= 500:
                    # Exponential backoff for server errors
                    backoff_delay = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Server error {e.response.status_code} for {url}, attempt {attempt + 1}, retrying in {backoff_delay}s")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(backoff_delay)
                        continue
                else:
                    logger.warning(f"HTTP error {e.response.status_code} for {url}")
                    return None
                    
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                # Exponential backoff for network errors
                backoff_delay = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Network error for {url}, attempt {attempt + 1}, retrying in {backoff_delay}s: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(backoff_delay)
                    continue
                    
            except Exception as e:
                logger.error(f"Unexpected error fetching {url}: {e}")
                return None
        
        # If regular fetch failed, try JavaScript rendering as fallback
        logger.info(f"Regular fetch failed, trying JavaScript rendering for {url}")
        js_content = await self._fetch_with_js_rendering(url)
        if js_content:
            return js_content
        
        logger.error(f"Failed to fetch {url} after {self.config.max_retries} attempts and JS fallback")
        return None
    
    def _needs_js_rendering(self, content: str) -> bool:
        """Check if page content suggests JavaScript rendering is needed."""
        if not content:
            return False
        
        content_lower = content.lower()
        
        # Check for JavaScript indicators
        js_indicators = [
            'react', 'vue', 'angular', 'spa', 'single-page',
            'lazy-load', 'infinite-scroll', 'dynamic-content',
            'document.ready', 'window.onload', 'addEventListener'
        ]
        
        # Count script tags
        script_count = content_lower.count('<script')
        
        # Check for minimal content with lots of scripts
        if script_count >= self.config.js_detection_script_threshold:
            return True
        
        # Check for JavaScript framework indicators
        for indicator in js_indicators:
            if indicator in content_lower:
                return True
        
        # Check for very little visible content
        if len(content.strip()) < 1000 and script_count > 3:
            return True
        
        return False
    
    async def _fetch_with_js_rendering(self, url: str) -> Optional[str]:
        """Fetch page content using advanced JavaScript rendering with resource management."""
        if not self.config.js_rendering_enabled:
            return None
        
        try:
            from playwright.async_api import async_playwright
            import psutil
            
            # Check system resources before launching browser
            memory_percent = psutil.virtual_memory().percent
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if memory_percent > 85:
                logger.warning(f"High memory usage ({memory_percent}%), skipping JS rendering for {url}")
                return None
            
            if cpu_percent > 90:
                logger.warning(f"High CPU usage ({cpu_percent}%), skipping JS rendering for {url}")
                return None
            
            async with async_playwright() as p:
                # Launch browser with resource limits
                browser = await p.chromium.launch(
                    headless=self.config.js_rendering_headless,
                    args=[
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-web-security',
                        '--disable-features=VizDisplayCompositor',
                        '--memory-pressure-off',
                        '--max_old_space_size=512'  # Limit memory usage
                    ]
                )
                
                try:
                    # Create context with resource limits
                    context = await browser.new_context(
                        viewport={
                            'width': self.config.js_rendering_viewport_width,
                            'height': self.config.js_rendering_viewport_height
                        },
                        user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                    )
                    
                    page = await context.new_page()
                    
                    # Set timeouts and limits
                    page.set_default_timeout(self.config.js_rendering_timeout * 1000)
                    page.set_default_navigation_timeout(self.config.js_rendering_timeout * 1000)
                    
                    # Block unnecessary resources to save bandwidth and memory
                    await page.route("**/*", self._route_handler)
                    
                    # Navigate with advanced options
                    response = await page.goto(
                        url, 
                        wait_until='networkidle',
                        timeout=self.config.js_rendering_timeout * 1000
                    )
                    
                    if not response or response.status >= 400:
                        logger.warning(f"JS rendering got bad response for {url}: {response.status if response else 'No response'}")
                        return None
                    
                    # Wait for dynamic content
                    await asyncio.sleep(self.config.js_rendering_wait_time)
                    
                    # Scroll to trigger lazy loading
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await asyncio.sleep(1)
                    await page.evaluate("window.scrollTo(0, 0)")
                    await asyncio.sleep(1)
                    
                    # Get final content
                    content = await page.content()
                    
                    # Get performance metrics
                    metrics = await page.evaluate("""
                        () => {
                            const perf = performance.getEntriesByType('navigation')[0];
                            return {
                                loadTime: perf.loadEventEnd - perf.loadEventStart,
                                domContentLoaded: perf.domContentLoadedEventEnd - perf.domContentLoadedEventStart,
                                imagesLoaded: document.images.length,
                                scriptsExecuted: document.scripts.length
                            };
                        }
                    """)
                    
                    logger.info(f"JS rendering successful for {url}: {metrics['imagesLoaded']} images, {metrics['scriptsExecuted']} scripts")
                    return content
                    
                finally:
                    await context.close()
                    await browser.close()
                
        except Exception as e:
            logger.warning(f"JavaScript rendering failed for {url}: {e}")
            return None
    
    async def _route_handler(self, route):
        """Route handler to block unnecessary resources."""
        resource_type = route.request.resource_type
        url = route.request.url
        
        # Block unnecessary resources
        blocked_types = {'font', 'media', 'websocket', 'manifest'}
        blocked_domains = {
            'google-analytics.com', 'googletagmanager.com', 'facebook.com',
            'twitter.com', 'doubleclick.net', 'adsystem.amazon.com'
        }
        
        if resource_type in blocked_types:
            await route.abort()
            return
        
        # Block ads and tracking
        if any(domain in url for domain in blocked_domains):
            await route.abort()
            return
        
        # Block large files
        if any(ext in url.lower() for ext in ['.pdf', '.zip', '.exe', '.dmg']):
            await route.abort()
            return
        
        await route.continue_()
    
    def _validate_url_security(self, url: str) -> Tuple[bool, str]:
        """Validate URL for security threats."""
        try:
            parsed = urlparse(url)
            
            # Check for malicious schemes
            malicious_schemes = {'javascript', 'data', 'file', 'ftp'}
            if parsed.scheme.lower() in malicious_schemes:
                return False, "MALICIOUS_SCHEME"
            
            # Only allow http/https schemes
            if parsed.scheme.lower() not in {'http', 'https'}:
                return False, "UNSAFE_SCHEME"
            
            # Check for blocked hosts
            blocked_hosts = {
                "doubleclick.net", "exoclick.com", "s.magsrv.com", "afcdn.net",
                "google-analytics.com", "googletagmanager.com", "facebook.com", "twitter.com"
            }
            if parsed.hostname in blocked_hosts:
                return False, "BLOCKED_HOST"
            
            # Check for suspicious patterns
            suspicious_patterns = [
                r'javascript:', r'data:', r'vbscript:', r'onload=', r'onerror=',
                r'<script', r'<iframe', r'<object', r'<embed'
            ]
            for pattern in suspicious_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False, "SUSPICIOUS_PATTERN"
            
            return True, "SAFE"
            
        except Exception as e:
            logger.warning(f"URL validation error for {url}: {e}")
            return False, "VALIDATION_ERROR"
    
    def _resolve_redirect_url(self, base_url: str, redirect_url: str) -> str:
        """Resolve relative redirect URLs to absolute URLs."""
        try:
            if redirect_url.startswith(('http://', 'https://')):
                return redirect_url
            else:
                return urljoin(base_url, redirect_url)
        except Exception as e:
            logger.warning(f"Error resolving redirect URL {redirect_url} from {base_url}: {e}")
            return redirect_url
    
    async def _extract_with_selector(
        self, 
        soup: BeautifulSoup, 
        selector_config: Dict[str, Any], 
        base_url: str, 
        recipe: Dict[str, Any]
    ) -> List[ExtractedImage]:
        """Extract images using a specific selector."""
        selector = selector_config.get('css', '')
        if not selector:
            return []
        
        try:
            elements = soup.select(selector)
            images = []
            
            for element in elements:
                image = self._extract_from_element(element, selector, base_url, recipe)
                if image:
                    images.append(image)
            
            return images
            
        except Exception as e:
            logger.warning(f"Error with selector '{selector}': {e}")
            return []
    
    async def _extract_with_selector_streaming(
        self, 
        soup: BeautifulSoup, 
        selector_config: Dict[str, Any], 
        base_url: str, 
        recipe: Dict[str, Any]
    ) -> AsyncIterator[ExtractedImage]:
        """Stream images using a specific selector."""
        selector = selector_config.get('css', '')
        if not selector:
            return
        
        try:
            elements = soup.select(selector)
            
            for element in elements:
                image = self._extract_from_element(element, selector, base_url, recipe)
                if image:
                    yield image
                    
        except Exception as e:
            logger.warning(f"Error with selector '{selector}': {e}")
    
    def _extract_from_element(
        self, 
        element, 
        selector: str, 
        base_url: str, 
        recipe: Dict[str, Any]
    ) -> Optional[ExtractedImage]:
        """Extract image information from a single HTML element."""
        try:
            # Get attributes priority from recipe
            attrs_priority = recipe.get('attributes_priority', ['src', 'data-src', 'srcset'])
            
            # Try to resolve image URL
            image_url = self._resolve_image_url(element, base_url, attrs_priority)
            if not image_url:
                return None
            
            # Extract all attributes
            attributes = dict(element.attrs) if hasattr(element, 'attrs') else {}
            
            # Check if image meets minimum size requirements
            if not self._meets_minimum_size(attributes):
                logger.debug(f"Image filtered out due to size: {image_url}")
                return None
            
            # Check for ad/redirect container leakage
            if self._is_in_ad_container(element):
                logger.debug(f"Image filtered out due to ad container: {image_url}")
                return None
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(image_url, attributes, element)
            
            # Get context information
            context = self._get_element_context(element)
            
            return ExtractedImage(
                url=image_url,
                selector=selector,
                attributes=attributes,
                context=context,
                quality_score=quality_score
            )
            
        except Exception as e:
            logger.debug(f"Error extracting from element: {e}")
            return None
    
    def _resolve_image_url(self, element, base_url: str, attrs_priority: List[str]) -> Optional[str]:
        """Resolve image URL from HTML element using attribute priority."""
        for attr in attrs_priority:
            value = element.get(attr)
            if value:
                if attr == 'srcset':
                    # Handle srcset - take highest resolution
                    url = self._parse_srcset(value)
                else:
                    url = value
                
                if url:
                    return urljoin(base_url, url)
        
        return None
    
    def _parse_srcset(self, srcset: str) -> Optional[str]:
        """Parse srcset and return highest resolution URL."""
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
                
                # Parse descriptor to get width/density
                width = 0
                if descriptor.endswith('x'):
                    try:
                        density = float(descriptor[:-1])
                        width = int(320 * density)  # Assume base width of 320
                    except ValueError:
                        continue
                elif descriptor.endswith('w'):
                    try:
                        width = int(descriptor[:-1])
                    except ValueError:
                        continue
                else:
                    width = 320  # Default width
                
                candidates.append((url, width))
            
            if candidates:
                # Sort by width (descending) and return highest
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
                
        except Exception as e:
            logger.debug(f"Error parsing srcset: {e}")
        
        return None
    
    def _calculate_quality_score(self, url: str, attributes: Dict[str, str], element) -> float:
        """Calculate quality score for an image based on URL and attributes."""
        score = 0.0
        
        # Check URL for quality indicators
        url_lower = url.lower()
        for quality, indicators in self._quality_indicators.items():
            for indicator in indicators:
                if indicator in url_lower:
                    if quality == 'high':
                        score += 0.3
                    elif quality == 'medium':
                        score += 0.2
                    else:  # low
                        score += 0.1
                    break
        
        # Check attributes for quality indicators
        for attr_name, attr_value in attributes.items():
            if attr_value:
                attr_lower = attr_value.lower()
                for quality, indicators in self._quality_indicators.items():
                    for indicator in indicators:
                        if indicator in attr_lower:
                            if quality == 'high':
                                score += 0.2
                            elif quality == 'medium':
                                score += 0.1
                            else:  # low
                                score += 0.05
                            break
        
        # Check element context
        if hasattr(element, 'parent') and element.parent:
            parent_classes = ' '.join(element.parent.get('class', []))
            if any(keyword in parent_classes.lower() for keyword in ['gallery', 'album', 'photo']):
                score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _meets_minimum_size(self, attributes: Dict[str, str]) -> bool:
        """Check if image meets minimum size requirements."""
        try:
            min_width, min_height = self.config.min_image_size
            
            # Try to get dimensions from attributes
            width = None
            height = None
            
            # Check common dimension attributes
            for attr in ['width', 'data-width', 'w']:
                if attr in attributes:
                    try:
                        width = int(attributes[attr])
                        break
                    except (ValueError, TypeError):
                        continue
            
            for attr in ['height', 'data-height', 'h']:
                if attr in attributes:
                    try:
                        height = int(attributes[attr])
                        break
                    except (ValueError, TypeError):
                        continue
            
            # If we have both dimensions, check them
            if width is not None and height is not None:
                if width >= min_width and height >= min_height:
                    return True
                else:
                    logger.debug(f"Image too small: {width}x{height} < {min_width}x{min_height}")
                    return False
            
            # If we only have one dimension, be conservative and allow it
            # (we'll do a more thorough check during actual image processing)
            if width is not None or height is not None:
                logger.debug(f"Partial size info available, allowing image: width={width}, height={height}")
                return True
            
            # If no size info available, allow it (will be checked during processing)
            logger.debug("No size info available, allowing image for processing")
            return True
            
        except Exception as e:
            logger.debug(f"Error checking minimum size: {e}")
            return True  # Allow on error to avoid false negatives
    
    def _get_element_context(self, element) -> Dict[str, Any]:
        """Get context information about an element."""
        context = {}
        
        try:
            # Get parent information
            if hasattr(element, 'parent') and element.parent:
                context['parent_tag'] = element.parent.name
                context['parent_classes'] = element.parent.get('class', [])
                context['parent_id'] = element.parent.get('id', '')
            
            # Get sibling information
            if hasattr(element, 'find_next_sibling'):
                next_sibling = element.find_next_sibling()
                if next_sibling:
                    context['next_sibling_tag'] = next_sibling.name
            
            # Get text content
            if hasattr(element, 'get_text'):
                text = element.get_text(strip=True)
                if text:
                    context['text_content'] = text[:100]  # Limit length
            
        except Exception as e:
            logger.debug(f"Error getting element context: {e}")
        
        return context
    
    def _is_in_ad_container(self, element) -> bool:
        """Check if element is within an ad/redirect container that should be skipped."""
        try:
            # Ad-related class patterns to skip
            ad_patterns = [
                'ad', 'ads', 'advertisement', 'advert', 'sponsor', 'sponsored', 
                'promoted', 'promo', 'exo-native-widget', 'native-ad', 'native_ad',
                'banner', 'popup', 'modal', 'overlay', 'sidebar-ad', 'header-ad',
                'footer-ad', 'in-content-ad', 'between-content-ad', 'sticky-ad',
                'floating-ad', 'video-ad', 'display-ad', 'text-ad', 'link-ad'
            ]
            
            # Denylist of known ad/tracking hosts
            ad_hosts = {
                'doubleclick.net', 'googlesyndication.com', 'googleadservices.com',
                'amazon-adsystem.com', 'adsystem.amazon.com', 'facebook.com',
                'twitter.com', 'linkedin.com', 'pinterest.com', 'snapchat.com',
                'tiktok.com', 'youtube.com', 'vimeo.com', 'dailymotion.com',
                'outbrain.com', 'taboola.com', 'criteo.com', 'adsrvr.org',
                'adsystem.amazon.com', 'amazon-adsystem.com', 'googletagmanager.com',
                'google-analytics.com', 'hotjar.com', 'mixpanel.com', 'segment.com'
            }
            
            # Check current element and all ancestors
            current = element
            while current:
                # Check element classes and IDs
                if hasattr(current, 'get'):
                    classes = current.get('class', [])
                    element_id = current.get('id', '')
                    
                    # Check classes for ad patterns
                    for cls in classes:
                        cls_lower = cls.lower()
                        for pattern in ad_patterns:
                            if pattern in cls_lower:
                                logger.debug(f"Found ad pattern '{pattern}' in class '{cls}'")
                                return True
                    
                    # Check ID for ad patterns
                    if element_id:
                        id_lower = element_id.lower()
                        for pattern in ad_patterns:
                            if pattern in id_lower:
                                logger.debug(f"Found ad pattern '{pattern}' in ID '{element_id}'")
                                return True
                    
                    # Check data attributes for ad indicators
                    for attr_name, attr_value in current.attrs.items():
                        if attr_name.startswith('data-') and isinstance(attr_value, str):
                            attr_lower = attr_value.lower()
                            for pattern in ad_patterns:
                                if pattern in attr_lower:
                                    logger.debug(f"Found ad pattern '{pattern}' in attribute '{attr_name}'")
                                    return True
                
                # Check if element contains links to ad hosts
                if hasattr(current, 'find_all'):
                    links = current.find_all('a', href=True)
                    for link in links:
                        href = link.get('href', '')
                        if href:
                            try:
                                from urllib.parse import urlparse
                                parsed = urlparse(href)
                                if parsed.hostname and parsed.hostname.lower() in ad_hosts:
                                    logger.debug(f"Found ad host '{parsed.hostname}' in link")
                                    return True
                            except Exception:
                                continue
                
                # Move to parent element
                current = getattr(current, 'parent', None)
            
            return False
            
        except Exception as e:
            logger.debug(f"Error checking ad container: {e}")
            return False  # Allow on error to avoid false positives
    
    async def _extract_extra_sources(
        self, 
        soup: BeautifulSoup, 
        base_url: str, 
        recipe: Dict[str, Any]
    ) -> List[ExtractedImage]:
        """Extract images from extra sources (meta tags, etc.)."""
        extra_sources = recipe.get('extra_sources', [])
        images = []
        
        for source in extra_sources:
            try:
                source_images = self._extract_from_extra_source(soup, source, base_url)
                images.extend(source_images)
            except Exception as e:
                logger.debug(f"Error extracting from extra source '{source}': {e}")
        
        return images
    
    async def _extract_extra_sources_streaming(
        self, 
        soup: BeautifulSoup, 
        base_url: str, 
        recipe: Dict[str, Any]
    ) -> AsyncIterator[ExtractedImage]:
        """Stream images from extra sources."""
        extra_sources = recipe.get('extra_sources', [])
        
        for source in extra_sources:
            try:
                source_images = self._extract_from_extra_source(soup, source, base_url)
                for image in source_images:
                    yield image
            except Exception as e:
                logger.debug(f"Error extracting from extra source '{source}': {e}")
    
    def _extract_from_extra_source(
        self, 
        soup: BeautifulSoup, 
        source: str, 
        base_url: str
    ) -> List[ExtractedImage]:
        """Extract images from a specific extra source."""
        images = []
        
        try:
            if source.startswith("meta[property='og:image']"):
                meta = soup.find('meta', property='og:image')
                if meta and meta.get('content'):
                    url = urljoin(base_url, meta.get('content'))
                    images.append(ExtractedImage(
                        url=url,
                        selector=source,
                        attributes={'content': meta.get('content')},
                        context={'type': 'og_image'},
                        quality_score=0.8  # High quality for OG images
                    ))
            
            elif source.startswith("link[rel='image_src']"):
                link = soup.find('link', rel='image_src')
                if link and link.get('href'):
                    url = urljoin(base_url, link.get('href'))
                    images.append(ExtractedImage(
                        url=url,
                        selector=source,
                        attributes={'href': link.get('href')},
                        context={'type': 'image_src'},
                        quality_score=0.7
                    ))
            
            elif source.startswith("video::attr(poster)"):
                videos = soup.find_all('video')
                for video in videos:
                    if video.get('poster'):
                        url = urljoin(base_url, video.get('poster'))
                        images.append(ExtractedImage(
                            url=url,
                            selector=source,
                            attributes={'poster': video.get('poster')},
                            context={'type': 'video_poster'},
                            quality_score=0.6
                        ))
            
            # Add more extra source types as needed
            
        except Exception as e:
            logger.debug(f"Error extracting from extra source '{source}': {e}")
        
        return images
    
    def _deduplicate_and_validate(self, images: List[ExtractedImage]) -> List[ExtractedImage]:
        """Remove duplicates and validate URLs."""
        seen_urls = set()
        valid_images = []
        
        for image in images:
            if image.url not in seen_urls and self._validate_url(image.url):
                seen_urls.add(image.url)
                valid_images.append(image)
        
        # Sort by quality score (descending)
        valid_images.sort(key=lambda x: x.quality_score, reverse=True)
        
        return valid_images
    
    def _validate_url(self, url: str) -> bool:
        """Validate URL for security and safety."""
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme.lower() in self._malicious_schemes:
                return False
            
            # Check for suspicious extensions
            path_lower = parsed.path.lower()
            if any(path_lower.endswith(ext) for ext in self._suspicious_extensions):
                return False
            
            # Check for bait query parameters
            query_params = parse_qs(parsed.query)
            if any(key.lower() in self._bait_query_keys for key in query_params.keys()):
                return False
            
            # Check blocked hosts
            if parsed.netloc.lower() in {host.lower() for host in self._blocked_hosts}:
                return False
            
            # Check blocked TLDs
            if any(parsed.netloc.lower().endswith(tld) for tld in self._blocked_tlds):
                return False
            
            # Check content type (if available)
            if hasattr(self, '_content_type_cache'):
                content_type = self._content_type_cache.get(url)
                if content_type and content_type in self.config.blocked_content_types:
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error validating URL {url}: {e}")
            return False
