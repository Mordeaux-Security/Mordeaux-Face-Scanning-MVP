"""
Unified HTTP Service

Provides shared HTTP client creation, connection pooling, redirect handling,
and JavaScript rendering for both crawler and selector miner.

Consolidates logic from:
- backend/app/services/redirect_utils.py
- tools/redirect_utils.py (duplicate)
- tools/selector_miner.py (Playwright rendering)
"""

import asyncio
import logging
import os
from typing import Optional, Tuple, Callable, Any, Dict
from urllib.parse import urljoin, urlparse
import httpx
from contextlib import asynccontextmanager

# Optional Playwright import for JavaScript rendering
try:
    from playwright.async_api import async_playwright, Browser, BrowserContext
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    async_playwright = None

logger = logging.getLogger(__name__)

# Security constants (from redirect_utils.py)
MALICIOUS_SCHEMES = {'javascript', 'data', 'file', 'ftp'}
BLOCKED_HOSTS = set()  # Can be populated with blocked hosts
BLOCKED_TLDS = set()   # Can be populated with blocked TLDs

# HTTP Configuration
DEFAULT_HTTP_LIMITS = httpx.Limits(
    max_keepalive_connections=200,
    max_connections=500
)

DEFAULT_TIMEOUT = httpx.Timeout(
    connect=10.0,
    read=30.0,
    write=10.0,
    pool=5.0
)

DEFAULT_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Redirect configuration
MAX_REDIRECTS = 3
REDIRECT_STATUS_CODES = {301, 302, 303, 307, 308}


def validate_url_security(url: str) -> Tuple[bool, str]:
    """
    Validate URL for security threats.
    
    Args:
        url: URL to validate
        
    Returns:
        Tuple of (is_safe, reason_code)
    """
    try:
        parsed = urlparse(url)
        
        # Check for malicious schemes
        if parsed.scheme.lower() in MALICIOUS_SCHEMES:
            return False, "MALICIOUS_SCHEME"
        
        # Only allow http/https schemes
        if parsed.scheme.lower() not in {'http', 'https'}:
            return False, "UNSAFE_SCHEME"
        
        # Check blocked hosts
        if parsed.netloc.lower() in BLOCKED_HOSTS:
            return False, "BLOCKED_HOST"
        
        # Check blocked TLDs
        for tld in BLOCKED_TLDS:
            if parsed.netloc.lower().endswith(tld):
                return False, "BLOCKED_TLD"
        
        return True, "SAFE"
        
    except Exception as e:
        logger.warning(f"URL validation error for {url}: {e}")
        return False, "VALIDATION_ERROR"


async def fetch_with_redirects(
    url: str, 
    client: httpx.AsyncClient, 
    max_hops: int = MAX_REDIRECTS,
    method: str = "GET"
) -> Tuple[Optional[httpx.Response], str]:
    """
    Fetch URL with manual redirect handling and security validation.
    
    Args:
        url: URL to fetch
        client: httpx.AsyncClient instance (must have follow_redirects=False)
        max_hops: Maximum number of redirect hops (default: 3)
        method: HTTP method to use ("GET", "HEAD", etc.)
        
    Returns:
        Tuple of (response, reason_code)
        - response: httpx.Response if successful, None if failed
        - reason_code: Reason for failure or "SUCCESS" if successful
    """
    current_url = url
    redirect_count = 0
    
    while redirect_count <= max_hops:
        # Validate URL security
        is_safe, reason = validate_url_security(current_url)
        if not is_safe:
            logger.warning(f"URL rejected: {reason} - {current_url}")
            return None, f"REDIRECT_BLOCKED_{reason}"
        
        try:
            # Make request
            if method.upper() == "GET":
                response = await client.get(current_url)
            elif method.upper() == "HEAD":
                response = await client.head(current_url)
            else:
                response = await client.request(method, current_url)
            
            # Handle redirects
            if response.status_code in REDIRECT_STATUS_CODES:
                redirect_url = response.headers.get('location')
                if not redirect_url:
                    return None, "REDIRECT_BLOCKED_NO_LOCATION"
                
                # Resolve relative redirects
                redirect_url = urljoin(current_url, redirect_url)
                
                # Check redirect limit
                if redirect_count >= max_hops:
                    logger.warning(f"Redirect limit reached ({max_hops} hops): {current_url}")
                    return None, "REDIRECT_CAP"
                
                # Validate redirect security
                is_safe, reason = validate_url_security(redirect_url)
                if not is_safe:
                    logger.warning(f"Redirect blocked: {reason} - {redirect_url}")
                    return None, f"REDIRECT_BLOCKED_{reason}"
                
                current_url = redirect_url
                redirect_count += 1
                logger.debug(f"Following redirect {redirect_count}/{max_hops}: {redirect_url}")
                continue
            
            # Non-redirect response
            return response, "SUCCESS"
            
        except httpx.HTTPError as e:
            error_msg = f"HTTP error at {current_url}: {str(e)}"
            logger.error(error_msg)
            return None, f"HTTP_ERROR_{e.response.status_code if hasattr(e, 'response') else 'UNKNOWN'}"
        except Exception as e:
            error_msg = f"Request error at {current_url}: {str(e)}"
            logger.error(error_msg)
            return None, "REQUEST_ERROR"
    
    # This should never be reached due to the redirect_count >= max_hops check above
    return None, "REDIRECT_CAP"


async def fetch_html_with_redirects(
    url: str, 
    client: httpx.AsyncClient, 
    max_hops: int = MAX_REDIRECTS
) -> Tuple[Optional[str], str]:
    """
    Fetch HTML content with manual redirect handling.
    
    Args:
        url: URL to fetch
        client: httpx.AsyncClient instance (must have follow_redirects=False)
        max_hops: Maximum number of redirect hops (default: 3)
        
    Returns:
        Tuple of (html_content, reason_code)
    """
    response, reason = await fetch_with_redirects(url, client, max_hops, "GET")
    
    if response is None:
        return None, reason
    
    try:
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if 'text/html' not in content_type:
            return None, "NOT_HTML_CONTENT"
        
        response.raise_for_status()
        return response.text, "SUCCESS"
        
    except httpx.HTTPError as e:
        return None, f"HTTP_ERROR_{e.response.status_code}"
    except Exception as e:
        return None, f"CONTENT_ERROR_{str(e)}"


async def head_with_redirects(
    url: str, 
    client: httpx.AsyncClient, 
    max_hops: int = MAX_REDIRECTS
) -> Tuple[Optional[httpx.Response], str]:
    """
    Perform HEAD request with manual redirect handling.
    
    Args:
        url: URL to fetch
        client: httpx.AsyncClient instance (must have follow_redirects=False)
        max_hops: Maximum number of redirect hops (default: 3)
        
    Returns:
        Tuple of (response, reason_code)
    """
    return await fetch_with_redirects(url, client, max_hops, "HEAD")


class BrowserPool:
    """
    Manages Playwright browser instances for JavaScript rendering.
    Provides connection pooling and proper cleanup.
    """
    
    def __init__(self, max_browsers: int = 2):
        self.max_browsers = max_browsers
        self._browsers: list[Browser] = []
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
    
    async def render_page(self, url: str, timeout: int = 30000) -> str:
        """
        Render a page with JavaScript using pooled browser.
        
        Args:
            url: URL to render
            timeout: Timeout in milliseconds
            
        Returns:
            Rendered HTML content
        """
        if not self._browsers:
            raise RuntimeError("Browser pool not initialized")
        
        # Get available browser (simple round-robin)
        browser = self._browsers[0]  # Could implement proper load balancing
        
        try:
            context = await browser.new_context(
                java_script_enabled=True,
                user_agent=DEFAULT_HEADERS['User-Agent']
            )
            
            page = await context.new_page()
            page.set_default_timeout(timeout)
            
            logger.info(f"Rendering JavaScript for URL: {url}")
            
            # Navigate to the page
            await page.goto(url, wait_until='domcontentloaded')
            
            # Wait for potential dynamic content (removed fixed 3-second wait for faster processing)
            
            # Get the rendered HTML
            html_content = await page.content()
            
            await context.close()
            
            logger.info(f"JavaScript rendering completed for {url} ({len(html_content)} chars)")
            return html_content
            
        except Exception as e:
            logger.warning(f"JavaScript rendering failed for {url}: {e}")
            raise


def create_http_client(
    limits: Optional[httpx.Limits] = None,
    timeout: Optional[httpx.Timeout] = None,
    headers: Optional[Dict[str, str]] = None,
    **kwargs
) -> httpx.AsyncClient:
    """
    Create an httpx.AsyncClient with safe defaults for redirect handling.
    
    Args:
        limits: Connection limits (defaults to DEFAULT_HTTP_LIMITS)
        timeout: Request timeouts (defaults to DEFAULT_TIMEOUT)
        headers: Request headers (defaults to DEFAULT_HEADERS)
        **kwargs: Additional arguments for httpx.AsyncClient
        
    Returns:
        httpx.AsyncClient with follow_redirects=False
    """
    # Ensure follow_redirects is False
    kwargs['follow_redirects'] = False
    
    # Set safe defaults if not provided
    if limits is None:
        limits = DEFAULT_HTTP_LIMITS
    if timeout is None:
        timeout = DEFAULT_TIMEOUT
    if headers is None:
        headers = DEFAULT_HEADERS.copy()
    
    # Merge headers
    if 'headers' in kwargs:
        headers.update(kwargs['headers'])
        del kwargs['headers']
    
    # Set other defaults
    if 'verify' not in kwargs:
        kwargs['verify'] = True
    
    if 'http2' not in kwargs:
        kwargs['http2'] = True
    
    return httpx.AsyncClient(
        limits=limits,
        timeout=timeout,
        headers=headers,
        **kwargs
    )


class HttpService:
    """
    Unified HTTP service providing client creation, HTML fetching,
    and JavaScript rendering capabilities.
    """
    
    def __init__(self, browser_pool_size: int = 2):
        self.browser_pool_size = browser_pool_size
        self._browser_pool: Optional[BrowserPool] = None
    
    async def create_client(self, **kwargs) -> httpx.AsyncClient:
        """
        Create a new HTTP client with configured settings.
        
        Args:
            **kwargs: Arguments passed to create_http_client()
            
        Returns:
            Configured httpx.AsyncClient
        """
        return create_http_client(**kwargs)
    
    async def fetch_html(
        self, 
        url: str, 
        client: httpx.AsyncClient,
        use_js: bool = False,
        max_redirects: int = MAX_REDIRECTS
    ) -> Tuple[Optional[str], str]:
        """
        Fetch HTML content from URL.
        
        Args:
            url: URL to fetch
            client: HTTP client to use
            use_js: Whether to use JavaScript rendering
            max_redirects: Maximum redirect hops
            
        Returns:
            Tuple of (html_content, reason_code)
        """
        if use_js and PLAYWRIGHT_AVAILABLE:
            return await self._fetch_with_js(url, client)
        else:
            return await fetch_html_with_redirects(url, client, max_redirects)
    
    async def _fetch_with_js(self, url: str, client: httpx.AsyncClient) -> Tuple[Optional[str], str]:
        """
        Fetch HTML with JavaScript rendering.
        
        Args:
            url: URL to fetch
            client: HTTP client (not used for JS rendering)
            
        Returns:
            Tuple of (html_content, reason_code)
        """
        try:
            # Initialize browser pool if needed
            if not self._browser_pool:
                self._browser_pool = BrowserPool(self.browser_pool_size)
                await self._browser_pool.__aenter__()
            
            # Render with JavaScript
            html_content = await self._browser_pool.render_page(url)
            return html_content, "SUCCESS"
            
        except Exception as e:
            logger.warning(f"JavaScript rendering failed for {url}: {e}")
            return None, f"JS_RENDER_ERROR_{str(e)}"
    
    async def fetch_with_validation(
        self,
        url: str,
        client: httpx.AsyncClient,
        validate_fn: Callable[[httpx.Response], bool],
        **kwargs
    ) -> Tuple[Optional[httpx.Response], str]:
        """
        Fetch URL with custom validation function.
        
        Args:
            url: URL to fetch
            client: HTTP client to use
            validate_fn: Function to validate response
            **kwargs: Additional arguments for fetch_with_redirects
            
        Returns:
            Tuple of (response, reason_code)
        """
        response, reason = await fetch_with_redirects(url, client, **kwargs)
        
        if response is None:
            return None, reason
        
        try:
            if validate_fn(response):
                return response, "SUCCESS"
            else:
                return None, "VALIDATION_FAILED"
        except Exception as e:
            return None, f"VALIDATION_ERROR_{str(e)}"
    
    async def close(self):
        """Close browser pool if initialized."""
        if self._browser_pool:
            await self._browser_pool.__aexit__(None, None, None)
            self._browser_pool = None


async def fetch_html_with_js_rendering(
    url: str,
    timeout: float = 10.0,  # Increased from 5.0
    wait_for_selector: str = None,
    wait_for_network_idle: bool = True  # NEW parameter
) -> tuple[str, list]:
    """
    Fetch HTML with JavaScript rendering using Playwright.
    
    Args:
        url: URL to fetch
        timeout: Maximum time to wait (seconds)
        wait_for_selector: Optional selector to wait for
        wait_for_network_idle: Whether to wait for network idle
        
    Returns:
        Tuple of (html, errors)
    """
    errors = []
    
    if not PLAYWRIGHT_AVAILABLE:
        errors.append("PLAYWRIGHT_NOT_AVAILABLE")
        return None, errors
    
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                java_script_enabled=True,
                # NOTE: Do NOT block images - we need them to load so we can extract their URLs!
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = await context.new_page()
            page.set_default_timeout(timeout * 1000)
            
            try:
                # Navigate with appropriate wait strategy
                if wait_for_network_idle:
                    await page.goto(url, wait_until='networkidle', timeout=timeout * 1000)
                else:
                    await page.goto(url, wait_until='domcontentloaded', timeout=timeout * 1000)
                
                # CRITICAL: Wait for dynamic content to load (3 seconds for JS execution)
                await page.wait_for_timeout(3000)
                
                # Optionally wait for specific selector
                if wait_for_selector:
                    await page.wait_for_selector(wait_for_selector, timeout=timeout * 1000)
                
                # Get rendered HTML
                html = await page.content()
                
                await browser.close()
                logger.info(f"JS rendering completed for {url} ({len(html)} chars)")
                return html, errors
                
            except Exception as e:
                if "Timeout" in str(e):
                    errors.append(f"JS_RENDER_TIMEOUT - Page load exceeded {timeout}s")
                else:
                    errors.append(f"JS_RENDER_ERROR - {str(e)}")
                await browser.close()
                return None, errors
                
    except Exception as e:
        errors.append(f"BROWSER_LAUNCH_ERROR - {str(e)}")
        return None, errors


# Convenience functions for backward compatibility
async def create_safe_client(**kwargs) -> httpx.AsyncClient:
    """
    Create an httpx.AsyncClient with safe defaults for redirect handling.
    Backward compatibility function.
    
    Args:
        **kwargs: Additional arguments for httpx.AsyncClient
        
    Returns:
        httpx.AsyncClient with follow_redirects=False
    """
    return create_http_client(**kwargs)
