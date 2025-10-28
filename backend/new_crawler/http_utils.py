"""
HTTP Utils for New Crawler System

Provides HTTP client with retry logic, redirect handling, and JavaScript rendering fallback.
Reuses patterns from existing http_service.py but with cleaner implementation.
"""

import asyncio
import logging
import random
import time
from typing import Optional, Tuple, Dict, Any, List
from urllib.parse import urljoin, urlparse
from contextlib import asynccontextmanager

import httpx
from bs4 import BeautifulSoup

# Optional Playwright import for JavaScript rendering
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

from .config import get_config

logger = logging.getLogger(__name__)

# Singleton pattern per process
_http_utils_instance = None


class HTTPUtils:
    """HTTP utilities with retry logic and JavaScript rendering fallback."""
    
    def __init__(self):
        self.config = get_config()
        self._client: Optional[httpx.AsyncClient] = None
        self._playwright = None
        self._browser = None
        self._browser_pool = None
        self._semaphore = asyncio.Semaphore(self.config.nc_crawler_concurrency)
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with realistic headers and redirect control."""
        if self._client is None:
            # Improved connection pooling for higher concurrency
            limits = httpx.Limits(
                max_connections=300,  # Up from crawler_concurrency to handle extractor load
                max_keepalive_connections=200,  # Up from 20 for better connection reuse
                keepalive_expiry=60.0  # Up from 30s to keep connections alive longer
            )
            timeout = httpx.Timeout(
                connect=10.0,
                read=self.config.nc_http_timeout,
                write=10.0,
                pool=5.0
            )
            
            # Don't set default headers at client level - we'll set them per request
            # This allows for User-Agent rotation and better header control
            
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                follow_redirects=False,  # Manual redirect control
                cookies=httpx.Cookies(),  # Persistent cookies per site
                http2=True  # Enable HTTP/2 for better multiplexing
            )
        return self._client
    
    def _get_headers(self) -> Dict[str, str]:
        """Get realistic browser headers."""
        ua = random.choice(self.user_agents)
        headers = {
            'User-Agent': ua,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        try:
            logger.info(f"HTTP headers (UA) selected: {ua}")
        except Exception:
            pass
        return headers
    
    def _needs_js_rendering(self, html: str) -> bool:
        """Check if HTML needs JavaScript rendering."""
        if not html or len(html) < 500:
            return True
        
        html_lower = html.lower()
        blocking_indicators = [
            'cloudflare',
            'captcha',
            'access denied',
            'please enable javascript',
            'javascript is disabled',
            'checking your browser',
            'ddos protection'
        ]
        
        return any(indicator in html_lower for indicator in blocking_indicators)
    
    async def fetch_html(self, url: str, use_js_fallback: bool = True, force_compare_first_visit: bool = False) -> Tuple[Optional[str], str]:
        """Fetch HTML content with optional JavaScript rendering fallback.
        If force_compare_first_visit is True, fetch both HTTP and JS and choose the better one by heuristic.
        """
        async with self._semaphore:  # Limit concurrency
            try:
                # Dual-fetch compare on first visit if enabled
                if force_compare_first_visit and PLAYWRIGHT_AVAILABLE and self.config.nc_js_first_visit_compare:
                    start = time.time()
                    # Run HTTP and JS concurrently with soft timeouts
                    async def _http_task():
                        return await self._fetch_with_redirects(url)
                    async def _js_task():
                        return await self._fetch_with_js(url)
                    http_task = asyncio.create_task(_http_task())
                    js_task = asyncio.create_task(_js_task())
                    done, pending = await asyncio.wait(
                        {http_task, js_task},
                        timeout=self.config.nc_js_render_timeout + 5,
                        return_when=asyncio.ALL_COMPLETED
                    )
                    http_html, http_err = (None, "TIMEOUT")
                    js_html, js_err = (None, "TIMEOUT")
                    if http_task in done:
                        http_html, http_err = http_task.result()
                    if js_task in done:
                        js_html, js_err = js_task.result()
                    # Cancel any stragglers
                    for p in pending:
                        p.cancel()
                    http_count = await self._estimate_img_candidates(http_html, url) if http_html else 0
                    js_count = await self._estimate_img_candidates(js_html, url) if js_html else 0
                    logger.info(f"First-visit compare for {url}: HTTP={http_count}, JS={js_count}, elapsed={(time.time()-start)*1000:.1f}ms")
                    if js_count >= max(5, 2 * http_count):
                        return js_html, "SUCCESS"
                    return (http_html or js_html), ("SUCCESS" if (http_html or js_html) else (http_err or js_err or "FAILED"))

                # Try standard HTTP first
                html, error = await self._fetch_with_redirects(url)
                
                if html and not self._needs_js_rendering(html):
                    logger.debug(f"Standard HTTP successful for {url}")
                    return html, "SUCCESS"
                
                # Try JavaScript rendering if needed and enabled
                if use_js_fallback and PLAYWRIGHT_AVAILABLE:
                    logger.info(f"Standard HTTP failed or needs JS rendering for {url}")
                    js_html, js_error = await self._fetch_with_js(url)
                    
                    if js_html:
                        logger.info(f"JavaScript rendering successful for {url}")
                        return js_html, "SUCCESS"
                    else:
                        logger.warning(f"JavaScript rendering also failed for {url}: {js_error}")
                
                # Return whatever we got from standard HTTP
                return html, error or "FAILED"
                
            except Exception as e:
                logger.error(f"Error fetching HTML from {url}: {e}")
                return None, f"ERROR: {str(e)}"

    async def _estimate_img_candidates(self, html: Optional[str], base_url: str) -> int:
        """Rudimentary estimator: count <img> tags and common patterns to pick better HTML source."""
        try:
            if not html:
                return 0
            soup = BeautifulSoup(html, 'html.parser')
            return len(soup.find_all('img'))
        except Exception:
            return 0
    
    async def _fetch_with_redirects(self, url: str) -> Tuple[Optional[str], str]:
        """Fetch URL with manual redirect handling and same-origin policy."""
        client = await self._get_client()
        current_url = url
        redirect_count = 0
        max_redirects = self.config.nc_max_redirects
        original_domain = urlparse(url).netloc
        
        # Get headers for this request (allows User-Agent rotation)
        headers = self._get_headers() if self.config.nc_realistic_headers else {}
        
        while redirect_count <= max_redirects:
            try:
                response = await client.get(current_url, headers=headers)
                
                # Handle redirects
                if response.status_code in {301, 302, 303, 307, 308}:
                    redirect_url = response.headers.get('location')
                    if not redirect_url:
                        return None, "REDIRECT_NO_LOCATION"
                    
                    # Resolve relative redirects
                    redirect_url = urljoin(current_url, redirect_url)
                    redirect_domain = urlparse(redirect_url).netloc
                    
                    # Check same-origin policy
                    if self.config.nc_same_origin_redirects_only and redirect_domain != original_domain:
                        logger.warning(f"Cross-domain redirect blocked: {current_url} -> {redirect_url}")
                        return None, "CROSS_DOMAIN_REDIRECT_BLOCKED"
                    
                    # Check blocklist
                    if redirect_domain in self.config.nc_blocklist_redirect_hosts:
                        logger.warning(f"Redirect to blocked host: {redirect_url}")
                        return None, "REDIRECT_TO_BLOCKED_HOST"
                    
                    if redirect_count >= max_redirects:
                        logger.warning(f"Redirect limit reached for {url}")
                        return None, "REDIRECT_LIMIT"
                    
                    current_url = redirect_url
                    redirect_count += 1
                    logger.debug(f"Following redirect {redirect_count}/{max_redirects}: {redirect_url}")
                    continue
                
                # Check for successful response
                if response.status_code == 200:
                    content_type = response.headers.get('content-type', '').lower()
                    if 'text/html' in content_type:
                        return response.text, "SUCCESS"
                    else:
                        return None, f"NOT_HTML: {content_type}"
                else:
                    return None, f"HTTP_{response.status_code}"
                    
            except httpx.TimeoutException:
                return None, "TIMEOUT"
            except httpx.ConnectError:
                return None, "CONNECTION_ERROR"
            except Exception as e:
                return None, f"ERROR: {str(e)}"
        
        return None, "REDIRECT_LIMIT"
    
    async def _fetch_with_js(self, url: str) -> Tuple[Optional[str], str]:
        """Fetch HTML with JavaScript rendering."""
        try:
            if not PLAYWRIGHT_AVAILABLE:
                return None, "PLAYWRIGHT_NOT_AVAILABLE"
            
            # Get browser pool
            if not self._browser_pool:
                self._browser_pool = BrowserPool()
                await self._browser_pool.__aenter__()
            
            html, network_images = await self._browser_pool.render_page(url, timeout=self.config.nc_js_render_timeout)
            
            # Log network-captured images for debugging
            if network_images:
                logger.debug(f"Captured {len(network_images)} image URLs from network requests for {url}")
            
            return html, "SUCCESS"
            
        except Exception as e:
            logger.error(f"JavaScript rendering failed for {url}: {e}")
            return None, f"JS_ERROR: {str(e)}"
    
    async def head_check(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """Perform HEAD request to check if URL is accessible."""
        try:
            client = await self._get_client()
            headers = self._get_headers() if self.config.nc_realistic_headers else {}
            response = await client.head(url, headers=headers)
            
            info = {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': response.headers.get('content-length'),
                'last_modified': response.headers.get('last-modified'),
                'etag': response.headers.get('etag')
            }
            
            # Check if it's an image
            content_type = info['content_type'].lower()
            is_image = any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp'])
            
            # Check content length
            content_length = info['content_length']
            if content_length:
                try:
                    size_mb = int(content_length) / (1024 * 1024)
                    if size_mb > 10:  # 10MB limit
                        logger.warning(f"Image too large: {url} ({size_mb:.1f}MB)")
                        return False, info
                except (ValueError, TypeError):
                    pass
            
            return response.status_code == 200 and is_image, info
            
        except Exception as e:
            logger.error(f"HEAD check failed for {url}: {e}")
            return False, {'error': str(e)}
    
    async def download_to_temp(self, url: str, temp_dir: str = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Download image to temporary file."""
        import tempfile
        import os
        
        try:
            client = await self._get_client()
            headers = self._get_headers() if self.config.nc_realistic_headers else {}
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                return None, {'error': f'HTTP {response.status_code}'}
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if not any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp']):
                return None, {'error': f'Not an image: {content_type}'}
            
            # Check file size
            content_length = len(response.content)
            if content_length > 10 * 1024 * 1024:  # 10MB limit
                return None, {'error': f'File too large: {content_length / (1024*1024):.1f}MB'}
            
            # Create temporary file
            if temp_dir:
                os.makedirs(temp_dir, exist_ok=True)
            
            with tempfile.NamedTemporaryFile(
                dir=temp_dir,
                suffix='.jpg',
                delete=False
            ) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            info = {
                'content_type': content_type,
                'content_length': content_length,
                'temp_path': temp_path
            }
            
            logger.debug(f"Downloaded {url} to {temp_path}")
            return temp_path, info
            
        except Exception as e:
            logger.error(f"Failed to download {url}: {e}")
            return None, {'error': str(e)}
    
    async def close(self):
        """Close HTTP client and browser pool."""
        try:
            if self._client:
                await self._client.aclose()
                self._client = None
            
            if self._browser_pool:
                await self._browser_pool.__aexit__(None, None, None)
                self._browser_pool = None
            
            logger.info("HTTP client and browser pool closed")
        except Exception as e:
            logger.error(f"Error closing HTTP client: {e}")


class BrowserPool:
    """Manages Playwright browser instances for JavaScript rendering."""
    
    def __init__(self, max_browsers: int = 2):
        self.max_browsers = max_browsers
        self._browsers: List[Browser] = []
        self._playwright = None
        self._lock = asyncio.Lock()
        self._js_semaphore: Optional[asyncio.Semaphore] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not PLAYWRIGHT_AVAILABLE:
            raise RuntimeError("Playwright not available for JavaScript rendering")
        
        async with self._lock:
            if not self._playwright:
                self._playwright = await async_playwright().start()
                
                # Create browser pool
                for _ in range(self.max_browsers):
                    browser = await self._playwright.chromium.launch(
                        headless=True,
                        args=['--no-sandbox', '--disable-dev-shm-usage']
                    )
                    self._browsers.append(browser)
                # Initialize JS concurrency semaphore
                from .config import get_config
                cfg = get_config()
                self._js_semaphore = asyncio.Semaphore(max(1, int(getattr(cfg, 'nc_js_concurrency', 4))))
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        async with self._lock:
            # Close all browsers
            for browser in self._browsers:
                try:
                    await browser.close()
                except Exception as e:
                    logger.warning(f"Error closing browser: {e}")
            
            self._browsers.clear()
            
            # Stop playwright
            if self._playwright:
                try:
                    await self._playwright.stop()
                except Exception as e:
                    logger.warning(f"Error stopping playwright: {e}")
                finally:
                    self._playwright = None
    
    async def render_page(self, url: str, timeout: float = 15.0) -> Tuple[str, List[str]]:
        """Render a page with JavaScript using pooled browser with configurable wait strategies and network capture."""
        if not self._browsers:
            raise RuntimeError("Browser pool not initialized")
        
        # Import config at method level to avoid circular imports
        from .config import get_config
        config = get_config()
        
        # Get available browser (simple round-robin)
        browser = self._browsers[0]
        
        try:
            # Concurrency gate for JS rendering
            sem = self._js_semaphore or asyncio.Semaphore(4)
            await sem.acquire()
            # Align Playwright UA with httpx-selected UA for the process
            pw_ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            try:
                pw_ua = self.user_agents[0]
            except Exception:
                pass
            logger.info(f"Playwright UA selected: {pw_ua}")
            context = await browser.new_context(
                java_script_enabled=True,
                user_agent=pw_ua
            )
            
            page = await context.new_page()
            page.set_default_timeout(timeout * 1000)
            
            logger.info(f"Rendering JavaScript for URL: {url}")
            
            # Set up network capture for image URLs
            image_requests = set()
            if config.nc_capture_network_images:
                async def handle_response(response):
                    try:
                        # Capture direct image requests
                        if any(response.url.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                            image_requests.add(response.url)
                        
                        # Capture JSON responses that might contain image URLs
                        if 'application/json' in response.headers.get('content-type', ''):
                            try:
                                json_data = await response.json()
                                # Look for common image URL fields
                                for key in ['image', 'thumbnail', 'src', 'srcset', 'url', 'photo']:
                                    if key in json_data:
                                        value = json_data[key]
                                        if isinstance(value, str) and any(value.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                                            image_requests.add(value)
                                        elif isinstance(value, list):
                                            for item in value:
                                                if isinstance(item, str) and any(item.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                                                    image_requests.add(item)
                            except Exception:
                                pass  # Ignore JSON parsing errors
                    except Exception:
                        pass  # Ignore response handling errors
                
                page.on('response', handle_response)
            
            # Navigate to the page with configurable wait strategy
            wait_strategy = config.nc_js_wait_strategy
            
            if wait_strategy == "networkidle":
                # Wait for network to be idle (no requests for 500ms)
                await page.goto(url, wait_until='networkidle')
                logger.debug(f"Used networkidle wait strategy for {url}")
            elif wait_strategy == "fixed":
                # Use fixed wait time
                await page.goto(url, wait_until='domcontentloaded')
                await page.wait_for_timeout(config.nc_js_wait_timeout * 1000)
                logger.debug(f"Used fixed wait strategy ({config.nc_js_wait_timeout}s) for {url}")
            else:  # "both" strategy
                # Use both networkidle and fixed wait for maximum compatibility
                await page.goto(url, wait_until='domcontentloaded')
                
                # Wait for network idle with timeout
                try:
                    await page.wait_for_load_state('networkidle', timeout=config.nc_js_networkidle_timeout * 1000)
                    logger.debug(f"Network idle achieved for {url}")
                except Exception as e:
                    logger.debug(f"Network idle timeout for {url}: {e}")
                
                # Additional fixed wait for slow-loading content
                await page.wait_for_timeout(config.nc_js_wait_timeout * 1000)
                logger.debug(f"Used both wait strategy for {url}")
            
            # Wait for image selectors if configured
            if config.nc_js_wait_selectors:
                try:
                    await page.wait_for_selector(config.nc_js_wait_selectors, timeout=int(config.nc_js_wait_timeout * 1000))
                    logger.debug(f"Image selectors found for {url}")
                except Exception as e:
                    logger.debug(f"Image selector wait timeout for {url}: {e}")
            
            # Get the rendered HTML
            html_content = await page.content()
            
            await context.close()
            
            logger.info(f"JavaScript rendering completed for {url} ({len(html_content)} chars, {len(image_requests)} network images)")
            return html_content, list(image_requests)
            
        except Exception as e:
            logger.warning(f"JavaScript rendering failed for {url}: {e}")
            raise
        finally:
            try:
                sem.release()
            except Exception:
                pass




def get_http_utils() -> HTTPUtils:
    """Get singleton HTTP utils instance."""
    global _http_utils_instance
    if _http_utils_instance is None:
        _http_utils_instance = HTTPUtils()
    return _http_utils_instance


async def close_http_utils():
    """Close singleton HTTP utils."""
    global _http_utils_instance
    if _http_utils_instance:
        await _http_utils_instance.close()
        _http_utils_instance = None
