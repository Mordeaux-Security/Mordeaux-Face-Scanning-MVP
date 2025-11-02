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
from collections import deque

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


class DomainConnectionPool:
    """Per-domain connection pool manager with adaptive rate limiting and circuit breakers."""
    
    def __init__(self, config):
        self.config = config
        self._pools: Dict[str, httpx.AsyncClient] = {}
        self._default_pool: Optional[httpx.AsyncClient] = None
        self._pool_lock = asyncio.Lock()
        
        # Domain metrics for adaptive behavior
        self._domain_metrics: Dict[str, Dict[str, Any]] = {}  # domain -> metrics
        self._metrics_lock = asyncio.Lock()
        
        # Circuit breaker state per domain
        self._circuit_breakers: Dict[str, Dict[str, Any]] = {}  # domain -> {'state': 'open'|'half_open'|'closed', 'failures': int, 'open_until': float}
        self._circuit_lock = asyncio.Lock()
        
        # Base connection limits
        self._base_max_connections = config.nc_extractor_concurrency + 150
        self._base_max_keepalive = min(800, int(self._base_max_connections * 0.8))
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc or 'unknown'
    
    async def _get_domain_pool(self, domain: str) -> httpx.AsyncClient:
        """Get or create connection pool for domain."""
        async with self._pool_lock:
            if domain not in self._pools:
                # Calculate adaptive connection limits based on domain performance
                max_connections = await self._calculate_connection_limit(domain)
                max_keepalive = min(800, int(max_connections * 0.8))
                
                limits = httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive,
                    keepalive_expiry=60.0
                )
                timeout = httpx.Timeout(
                    connect=10.0,
                    read=self.config.nc_http_timeout,
                    write=10.0,
                    pool=5.0
                )
                
                self._pools[domain] = httpx.AsyncClient(
                    limits=limits,
                    timeout=timeout,
                    follow_redirects=False,
                    cookies=httpx.Cookies(),
                    http2=True
                )
                logger.debug(f"Created connection pool for {domain}: {max_connections} connections")
            
            return self._pools[domain]
    
    async def _calculate_connection_limit(self, domain: str) -> int:
        """Calculate adaptive connection limit for domain based on real-time metrics."""
        async with self._metrics_lock:
            if domain not in self._domain_metrics:
                # New domain: start with base limit divided by estimated active domains
                # Assume ~10-20 active domains, so divide base by 15
                return max(10, self._base_max_connections // 15)
            
            metrics = self._domain_metrics[domain]
            avg_response = metrics.get('avg_response_time', 500)
            success_rate = metrics.get('success_rate', 1.0)
            
            # Fast domains (< 200ms) get more connections
            if avg_response < 200 and success_rate > 0.95:
                return min(self._base_max_connections, max(50, int(self._base_max_connections * 0.15)))
            # Slow domains (> 1000ms) get fewer connections
            elif avg_response > 1000 or success_rate < 0.7:
                return max(5, int(self._base_max_connections * 0.05))
            # Default: moderate allocation
            else:
                return max(10, int(self._base_max_connections * 0.10))
    
    async def get_client_for_url(self, url: str) -> httpx.AsyncClient:
        """Get appropriate HTTP client for URL based on domain."""
        domain = self._extract_domain(url)
        return await self._get_domain_pool(domain)
    
    async def record_request(self, url: str, response_time: float, success: bool):
        """Record request metrics for adaptive behavior."""
        domain = self._extract_domain(url)
        async with self._metrics_lock:
            if domain not in self._domain_metrics:
                self._domain_metrics[domain] = {
                    'response_times': deque(maxlen=100),  # Keep last 100 response times
                    'success_count': 0,
                    'failure_count': 0,
                    'total_requests': 0
                }
            
            metrics = self._domain_metrics[domain]
            metrics['response_times'].append(response_time)
            metrics['total_requests'] += 1
            
            if success:
                metrics['success_count'] += 1
            else:
                metrics['failure_count'] += 1
            
            # Calculate metrics (deque automatically maintains maxlen=100)
            if len(metrics['response_times']) > 0:
                metrics['avg_response_time'] = sum(metrics['response_times']) / len(metrics['response_times'])
            else:
                metrics['avg_response_time'] = 500.0  # Default
            metrics['success_rate'] = metrics['success_count'] / metrics['total_requests']
            
            # Check circuit breaker threshold
            failure_rate = 1.0 - metrics['success_rate']
            if metrics['total_requests'] >= 10:  # Need minimum requests before opening circuit
                await self._update_circuit_breaker(domain, failure_rate)
    
    async def _update_circuit_breaker(self, domain: str, failure_rate: float):
        """Update circuit breaker state based on failure rate."""
        async with self._circuit_lock:
            if domain not in self._circuit_breakers:
                self._circuit_breakers[domain] = {
                    'state': 'closed',
                    'failures': 0,
                    'open_until': 0.0
                }
            
            cb = self._circuit_breakers[domain]
            failure_threshold = getattr(self.config, 'nc_circuit_breaker_failure_threshold', 5)
            
            if cb['state'] == 'open':
                # Check if timeout expired
                if time.time() > cb['open_until']:
                    cb['state'] = 'half_open'
                    cb['failures'] = 0
                    logger.info(f"Circuit breaker for {domain} moved to half_open")
            elif cb['state'] == 'half_open':
                # Test with one request
                if failure_rate > 0.5:
                    cb['state'] = 'open'
                    base_timeout = getattr(self.config, 'nc_circuit_breaker_open_timeout_base', 30.0)
                    cb['open_until'] = time.time() + base_timeout
                    logger.warning(f"Circuit breaker for {domain} opened (failure_rate={failure_rate:.2f})")
                else:
                    cb['state'] = 'closed'
                    logger.info(f"Circuit breaker for {domain} closed (recovered)")
            else:  # closed
                if failure_rate > 0.5 or cb['failures'] >= failure_threshold:
                    cb['state'] = 'open'
                    base_timeout = getattr(self.config, 'nc_circuit_breaker_open_timeout_base', 30.0)
                    # Exponential backoff: 30s, 60s, 120s, 240s, max 300s
                    timeout_multiplier = min(2 ** min(cb['failures'] // failure_threshold, 3), 10)
                    cb['open_until'] = time.time() + (base_timeout * timeout_multiplier)
                    cb['failures'] = 0
                    logger.warning(f"Circuit breaker for {domain} opened (failure_rate={failure_rate:.2f})")
                else:
                    cb['failures'] = int(failure_rate * 10)  # Track failures
    
    async def is_circuit_open(self, url: str) -> bool:
        """Check if circuit breaker is open for domain."""
        domain = self._extract_domain(url)
        async with self._circuit_lock:
            if domain not in self._circuit_breakers:
                return False
            cb = self._circuit_breakers[domain]
            if cb['state'] == 'open' and time.time() < cb['open_until']:
                return True
            return False
    
    async def close_all(self):
        """Close all connection pools."""
        async with self._pool_lock:
            for domain, pool in self._pools.items():
                try:
                    await pool.aclose()
                except Exception as e:
                    logger.error(f"Error closing pool for {domain}: {e}")
            self._pools.clear()
            if self._default_pool:
                try:
                    await self._default_pool.aclose()
                except Exception:
                    pass
                self._default_pool = None


class HTTPUtils:
    """HTTP utilities with retry logic and JavaScript rendering fallback."""
    
    def __init__(self):
        self.config = get_config()
        self._client: Optional[httpx.AsyncClient] = None  # Legacy fallback
        self._domain_pool: Optional[DomainConnectionPool] = None  # New per-domain pools
        self._playwright = None
        self._browser = None
        self._browser_pool = None
        self._browser_pool_lock = asyncio.Lock()
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        ]
        
        # HTML caching for JavaScript renders
        self._js_cache: Dict[str, Tuple[float, str, List[str]]] = {}  # url -> (timestamp, html, network_images)
        self._js_cache_ttl: float = 300.0  # 5 minutes cache TTL
        self._js_cache_lock = asyncio.Lock()  # Thread-safe cache access
    
    def _get_domain_pool(self) -> DomainConnectionPool:
        """Get or create domain connection pool manager."""
        if self._domain_pool is None:
            self._domain_pool = DomainConnectionPool(self.config)
        return self._domain_pool
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with realistic headers and redirect control (legacy fallback)."""
        if self._client is None:
            # Calculate connection limits based on extractor concurrency
            # Add buffer for crawlers and JS rendering (estimate ~150)
            total_extractor_concurrency = self.config.nc_extractor_concurrency
            total_connections = total_extractor_concurrency + 150
            max_keepalive = min(800, int(total_connections * 0.8))  # 80% keepalive, cap at 800
            
            # Improved connection pooling for higher concurrency
            limits = httpx.Limits(
                max_connections=total_connections,  # Match extractor capacity + buffer
                max_keepalive_connections=max_keepalive,  # 80% keepalive for better connection reuse
                keepalive_expiry=60.0  # Keep connections alive longer
            )
            logger.info(f"HTTP client limits: {total_connections} max connections "
                       f"({max_keepalive} keepalive) for {total_extractor_concurrency} extractor concurrency")
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
    
    async def _get_client_for_url(self, url: str) -> httpx.AsyncClient:
        """Get appropriate HTTP client for URL (per-domain pool or fallback)."""
        try:
            pool = self._get_domain_pool()
            # Check circuit breaker first
            if await pool.is_circuit_open(url):
                logger.debug(f"Circuit breaker open for {url}, using fallback client")
                return await self._get_client()
            return await pool.get_client_for_url(url)
        except Exception as e:
            logger.warning(f"Error getting domain pool for {url}: {e}, using fallback")
            return await self._get_client()
    
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
    
    async def fetch_html(self, url: str, use_js_fallback: bool = True, force_compare_first_visit: bool = False) -> Tuple[Optional[str], str, Optional[Dict[str, int]]]:
        """Fetch HTML content with optional JavaScript rendering fallback.
        If force_compare_first_visit is True, fetch both HTTP and JS and choose the better one by heuristic.
        
        Returns: (html, status_message, comparison_stats)
        comparison_stats is None unless force_compare_first_visit=True, then it's {'http_count': N, 'js_count': M}
        """
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
                
                # Return comparison stats WITHOUT storing strategy yet
                comparison_stats = {'http_count': http_count, 'js_count': js_count}
                
                # Return the better HTML
                if js_count >= max(5, 2 * http_count):
                    return js_html, "SUCCESS", comparison_stats
                return (http_html or js_html), ("SUCCESS" if (http_html or js_html) else (http_err or js_err or "FAILED")), comparison_stats

            # Try standard HTTP first
            html, error = await self._fetch_with_redirects(url)
            
            if html and not self._needs_js_rendering(html):
                logger.debug(f"Standard HTTP successful for {url}")
                return html, "SUCCESS", None
            
            # Try JavaScript rendering if needed and enabled
            if use_js_fallback and PLAYWRIGHT_AVAILABLE:
                logger.info(f"Standard HTTP failed or needs JS rendering for {url}")
                js_html, js_error = await self._fetch_with_js(url)
                
                if js_html:
                    logger.info(f"JavaScript rendering successful for {url}")
                    return js_html, "SUCCESS", None
                else:
                    logger.warning(f"JavaScript rendering also failed for {url}: {js_error}")
            
            # Return whatever we got from standard HTTP
            return html, error or "FAILED", None
            
        except Exception as e:
            logger.error(f"Error fetching HTML from {url}: {e}")
            return None, f"ERROR: {str(e)}", None

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
        client = await self._get_client_for_url(url)
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
            
            # Check cache first
            async with self._js_cache_lock:
                if url in self._js_cache:
                    cached_time, cached_html, cached_network = self._js_cache[url]
                    if time.time() - cached_time < self._js_cache_ttl:
                        logger.debug(f"Returning cached JS render for {url}")
                        return cached_html, "CACHED"
            
            # Ensure browser pool is initialized atomically
            # Initialize if pool is missing or browsers aren't ready yet
            if self._browser_pool is None or not getattr(self._browser_pool, '_browsers', []):
                async with self._browser_pool_lock:
                    # Double-check after acquiring lock in case another coroutine initialized it
                    if self._browser_pool is None or not getattr(self._browser_pool, '_browsers', []):
                        logger.info("Initializing browser pool for JavaScript rendering")
                        pool = BrowserPool()
                        await pool.__aenter__()
                        self._browser_pool = pool
                        logger.info(f"Browser pool initialized with {len(pool._browsers)} browsers")
            
            html, network_images = await self._browser_pool.render_page(url, timeout=self.config.nc_js_render_timeout)
            
            # Store in cache
            if html:
                async with self._js_cache_lock:
                    self._js_cache[url] = (time.time(), html, network_images)
                    # Clean old entries if cache gets too large (keep last 1000)
                    if len(self._js_cache) > 1000:
                        # Remove oldest 20% of entries
                        sorted_items = sorted(self._js_cache.items(), key=lambda x: x[1][0])
                        for key, _ in sorted_items[:200]:
                            del self._js_cache[key]
            
            # Log network-captured images for debugging
            if network_images:
                logger.debug(f"Captured {len(network_images)} image URLs from network requests for {url}")
            
            return html, "SUCCESS"
            
        except Exception as e:
            logger.error(f"JavaScript rendering failed for {url}: {e}")
            return None, f"JS_ERROR: {str(e)}"
    
    async def head_check(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """Perform HEAD request to check if URL is accessible."""
        request_start = time.time()
        success = False
        try:
            client = await self._get_client_for_url(url)
            headers = self._get_headers() if self.config.nc_realistic_headers else {}
            response = await client.head(url, headers=headers)
            
            response_time = time.time() - request_start
            success = response.status_code == 200
            
            info = {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'content_length': response.headers.get('content-length'),
                'last_modified': response.headers.get('last-modified'),
                'etag': response.headers.get('etag')
            }
            
            # Record metrics for adaptive behavior
            pool = self._get_domain_pool()
            await pool.record_request(url, response_time, success)
            
            # Check if it's an image
            content_type = info['content_type'].lower()
            is_image = any(img_type in content_type for img_type in ['image/', 'jpeg', 'jpg', 'png', 'gif', 'webp'])
            
            # Disallow SVG (often useless for faces)
            if 'svg' in content_type:
                return False, info
            
            # Enforce size bounds (mirror extractor defaults or keep local)
            min_bytes = getattr(self.config, 'nc_min_image_bytes', 4_096)
            max_bytes = getattr(self.config, 'nc_max_image_bytes', 10_000_000)
            content_length = info['content_length']
            if content_length:
                try:
                    size = int(content_length)
                    if size < min_bytes or size > max_bytes:
                        return False, info
                except (ValueError, TypeError):
                    pass
            
            return response.status_code == 200 and is_image, info
            
        except Exception as e:
            response_time = time.time() - request_start
            # Record failure
            pool = self._get_domain_pool()
            await pool.record_request(url, response_time, False)
            logger.error(f"HEAD check failed for {url}: {e}")
            return False, {'error': str(e)}
    
    async def download_to_temp(self, url: str, temp_dir: str = None) -> Tuple[Optional[str], Dict[str, Any]]:
        """Download image to temporary file with exponential backoff retry and redirect handling."""
        import tempfile
        import os
        
        max_retries = self.config.nc_max_retries
        base_delay = self.config.nc_retry_base_delay
        max_delay = self.config.nc_retry_max_delay
        jitter = self.config.nc_retry_jitter
        
        for attempt in range(max_retries + 1):
            request_start = time.time()
            success = False
            try:
                client = await self._get_client_for_url(url)
                headers = self._get_headers() if self.config.nc_realistic_headers else {}
                
                # Follow redirects manually (same logic as _fetch_with_redirects)
                current_url = url
                redirect_count = 0
                max_redirects = self.config.nc_max_redirects
                original_domain = urlparse(url).netloc
                response = None
                
                while redirect_count <= max_redirects:
                    response = await client.get(current_url, headers=headers)
                    
                    # Handle redirects
                    if response.status_code in {301, 302, 303, 307, 308}:
                        redirect_url = response.headers.get('location')
                        if not redirect_url:
                            logger.debug(f"No location header in redirect response for {current_url}")
                            break  # No location header, treat as failure
                        
                        # Resolve relative redirects
                        redirect_url = urljoin(current_url, redirect_url)
                        redirect_domain = urlparse(redirect_url).netloc
                        
                        # Check same-origin policy
                        if self.config.nc_same_origin_redirects_only and redirect_domain != original_domain:
                            logger.debug(f"Cross-domain redirect blocked: {current_url} -> {redirect_url}")
                            break
                        
                        # Check blocklist
                        if redirect_domain in self.config.nc_blocklist_redirect_hosts:
                            logger.debug(f"Redirect to blocked host: {redirect_url}")
                            break
                        
                        if redirect_count >= max_redirects:
                            logger.debug(f"Redirect limit reached for {url}")
                            break
                        
                        current_url = redirect_url
                        redirect_count += 1
                        logger.debug(f"Following redirect {redirect_count}/{max_redirects}: {redirect_url}")
                        continue
                    
                    # If we got here and status is 200, process the image
                    if response.status_code == 200:
                        break
                    else:
                        # Non-200 and not a redirect - treat as failure
                        break
                
                response_time = time.time() - request_start
                success = response and response.status_code == 200
                
                # Record metrics for adaptive behavior
                pool = self._get_domain_pool()
                await pool.record_request(url, response_time, success)
                
                if not response or response.status_code != 200:
                    if attempt < max_retries:
                        # Exponential backoff with jitter
                        delay = min(base_delay * (2 ** attempt) + random.uniform(0, jitter), max_delay)
                        status_code = response.status_code if response else "NO_RESPONSE"
                        logger.debug(f"Download failed for {url} (HTTP {status_code}), retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(delay)
                        continue
                    status_code = response.status_code if response else "NO_RESPONSE"
                    return None, {'error': f'HTTP {status_code}'}
                
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
                
                if redirect_count > 0:
                    logger.debug(f"Downloaded {url} (via {redirect_count} redirect(s)) to {temp_path}")
                else:
                    logger.debug(f"Downloaded {url} to {temp_path}")
                return temp_path, info
                
            except (httpx.TimeoutException, httpx.ConnectError) as e:
                response_time = time.time() - request_start
                # Record failure
                pool = self._get_domain_pool()
                await pool.record_request(url, response_time, False)
                
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, jitter), max_delay)
                    logger.debug(f"Download failed for {url} ({type(e).__name__}), retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Failed to download {url} after {max_retries} retries: {e}")
                return None, {'error': str(e)}
            except Exception as e:
                response_time = time.time() - request_start
                # Record failure
                pool = self._get_domain_pool()
                await pool.record_request(url, response_time, False)
                
                if attempt < max_retries:
                    # Exponential backoff with jitter
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, jitter), max_delay)
                    logger.debug(f"Download failed for {url} ({type(e).__name__}), retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Failed to download {url}: {e}")
                return None, {'error': str(e)}
        
        return None, {'error': 'Max retries exceeded'}
    
    async def close(self):
        """Close HTTP client and browser pool."""
        try:
            if self._domain_pool:
                await self._domain_pool.close_all()
                self._domain_pool = None
            
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
    
    def __init__(self, max_browsers: int = None):
        from .config import get_config
        cfg = get_config()
        self.max_browsers = max_browsers or getattr(cfg, 'nc_js_browser_pool_size', 8)
        self._browsers: List[Browser] = []
        self._playwright = None
        self._lock = asyncio.Lock()
        self._js_semaphore: Optional[asyncio.Semaphore] = None
        self._contexts: List[BrowserContext] = []  # Add context pool
        self._context_lock = asyncio.Lock()  # For thread-safe context access
        self._browser_idx = 0  # Round-robin counter
    
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
                
                # Create reusable context pool (2 contexts per browser)
                contexts_per_browser = 2
                default_ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                for browser in self._browsers:
                    for _ in range(contexts_per_browser):
                        context = await browser.new_context(
                            java_script_enabled=True,
                            user_agent=default_ua
                        )
                        self._contexts.append(context)
                logger.info(f"Created {len(self._contexts)} browser contexts ({contexts_per_browser} per browser)")
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with cleanup."""
        async with self._lock:
            # Close all contexts before browsers
            for context in self._contexts:
                try:
                    await context.close()
                except Exception as e:
                    logger.warning(f"Error closing context: {e}")
            self._contexts.clear()
            
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
        
        # Round-robin browser selection
        async with self._lock:
            self._browser_idx = (self._browser_idx + 1) % len(self._browsers)
            browser = self._browsers[self._browser_idx]
        
        try:
            # Concurrency gate for JS rendering
            sem = self._js_semaphore or asyncio.Semaphore(4)
            await sem.acquire()
            # Get context from pool (thread-safe)
            async with self._context_lock:
                if not self._contexts:
                    # Fallback: create new context if pool is empty
                    pw_ua = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
                    try:
                        # Try to get UA from HTTPUtils if available (for consistency)
                        from .http_utils import get_http_utils
                        http_utils = get_http_utils()
                        if hasattr(http_utils, 'user_agents') and http_utils.user_agents:
                            pw_ua = http_utils.user_agents[0]
                    except Exception:
                        pass
                    context = await browser.new_context(
                        java_script_enabled=True,
                        user_agent=pw_ua
                    )
                else:
                    context = self._contexts.pop(0)
                    # Clear cookies to prevent state pollution between renders
                    try:
                        await context.clear_cookies()
                    except Exception as e:
                        logger.debug(f"Failed to clear cookies: {e}")
            
            page = await context.new_page()
            page.set_default_timeout(timeout * 1000)
            
            # Navigate to blank page first to ensure clean state, then to target URL
            # This prevents JavaScript state pollution from previous renders
            try:
                await page.goto('about:blank', wait_until='domcontentloaded')
                await page.wait_for_timeout(100)  # Brief pause for cleanup
            except Exception:
                pass  # Continue if blank navigation fails
            
            logger.info(f"Rendering JavaScript for URL: {url}")
            
            # Set up network capture for image URLs
            image_requests = set()
            if config.nc_capture_network_images:
                async def handle_response(response):
                    try:
                        url_lower = response.url.lower()
                        # Only check direct image URLs (skip JSON parsing for speed)
                        if any(url_lower.endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.webp', '.gif']):
                            image_requests.add(response.url)
                    except Exception:
                        pass  # Ignore errors
                
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
            
            # Return context to pool instead of closing
            await page.close()  # Only close page
            async with self._context_lock:
                self._contexts.append(context)  # Return to pool
            
            logger.info(f"JavaScript rendering completed for {url} ({len(html_content)} chars, {len(image_requests)} network images)")
            return html_content, list(image_requests)
            
        except Exception as e:
            logger.warning(f"JavaScript rendering failed for {url}: {e}")
            # Ensure context is returned to pool even on error
            try:
                if 'context' in locals() and 'page' in locals():
                    try:
                        await page.close()
                    except Exception:
                        pass
                    async with self._context_lock:
                        self._contexts.append(context)
            except Exception:
                pass
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
