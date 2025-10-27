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
        """Get or create HTTP client."""
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
            self._client = httpx.AsyncClient(
                limits=limits,
                timeout=timeout,
                follow_redirects=True,
                max_redirects=self.config.nc_max_redirects,
                http2=True  # Enable HTTP/2 for better multiplexing
            )
        return self._client
    
    def _get_headers(self) -> Dict[str, str]:
        """Get realistic browser headers."""
        return {
            'User-Agent': random.choice(self.user_agents),
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
    
    async def fetch_html(self, url: str, use_js_fallback: bool = True) -> Tuple[Optional[str], str]:
        """Fetch HTML content with optional JavaScript rendering fallback."""
        async with self._semaphore:  # Limit concurrency
            try:
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
    
    async def _fetch_with_redirects(self, url: str) -> Tuple[Optional[str], str]:
        """Fetch URL with manual redirect handling."""
        client = await self._get_client()
        current_url = url
        redirect_count = 0
        max_redirects = self.config.nc_max_redirects
        
        while redirect_count <= max_redirects:
            try:
                response = await client.get(current_url)
                
                # Handle redirects
                if response.status_code in {301, 302, 303, 307, 308}:
                    redirect_url = response.headers.get('location')
                    if not redirect_url:
                        return None, "REDIRECT_NO_LOCATION"
                    
                    # Resolve relative redirects
                    redirect_url = urljoin(current_url, redirect_url)
                    
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
            
            html = await self._browser_pool.render_page(url, timeout=self.config.nc_js_render_timeout)
            return html, "SUCCESS"
            
        except Exception as e:
            logger.error(f"JavaScript rendering failed for {url}: {e}")
            return None, f"JS_ERROR: {str(e)}"
    
    async def head_check(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """Perform HEAD request to check if URL is accessible."""
        try:
            client = await self._get_client()
            response = await client.head(url)
            
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
            response = await client.get(url)
            
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
    
    async def render_page(self, url: str, timeout: float = 15.0) -> str:
        """Render a page with JavaScript using pooled browser."""
        if not self._browsers:
            raise RuntimeError("Browser pool not initialized")
        
        # Get available browser (simple round-robin)
        browser = self._browsers[0]
        
        try:
            context = await browser.new_context(
                java_script_enabled=True,
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            
            page = await context.new_page()
            page.set_default_timeout(timeout * 1000)
            
            logger.info(f"Rendering JavaScript for URL: {url}")
            
            # Navigate to the page
            await page.goto(url, wait_until='domcontentloaded')
            
            # Wait for potential dynamic content
            await page.wait_for_timeout(3000)
            
            # Get the rendered HTML
            html_content = await page.content()
            
            await context.close()
            
            logger.info(f"JavaScript rendering completed for {url} ({len(html_content)} chars)")
            return html_content
            
        except Exception as e:
            logger.warning(f"JavaScript rendering failed for {url}: {e}")
            raise




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
