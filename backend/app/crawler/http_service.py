"""
Centralized HTTP service for optimized request handling.

This module provides a singleton HTTP service that:
- Consolidates HTTP client instances
- Implements intelligent connection pooling
- Provides consistent retry logic
- Handles redirects efficiently
- Caches responses when appropriate
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse
import httpx
from ..core.config import get_settings
# URL security validation (moved from redirect_utils.py)

logger = logging.getLogger(__name__)

# Security constants
MALICIOUS_SCHEMES = {'javascript', 'data', 'file', 'ftp'}
BLOCKED_HOSTS = set()  # Can be populated with blocked hosts
BLOCKED_TLDS = set()   # Can be populated with blocked TLDs


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

@dataclass
class RequestConfig:
    """Configuration for HTTP requests."""
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    max_redirects: int = 3
    follow_redirects: bool = True
    verify_ssl: bool = True
    user_agent: str = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

@dataclass
class ResponseCache:
    """Simple in-memory response cache."""
    url: str
    content: bytes
    content_type: str
    timestamp: float
    ttl: float = 300.0  # 5 minutes default TTL

class HTTPService:
    """
    Centralized HTTP service with connection pooling and intelligent caching.
    """
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._response_cache: Dict[str, ResponseCache] = {}
        self._cache_cleanup_interval = 60.0  # Clean cache every minute
        self._last_cache_cleanup = 0.0
        self._settings = get_settings()
        
        # Connection pool configuration
        self._limits = httpx.Limits(
            max_keepalive_connections=200,  # Increased for better reuse
            max_connections=500,            # Increased for higher concurrency
            keepalive_expiry=30.0           # Keep connections alive longer
        )
        
        # Request configuration
        self._default_config = RequestConfig()
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def _ensure_client(self):
        """Ensure HTTP client is initialized."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self._default_config.timeout,
                    write=10.0,
                    pool=5.0
                ),
                verify=self._default_config.verify_ssl,
                http2=True,
                limits=self._limits,
                headers={
                    'User-Agent': self._default_config.user_agent
                },
                follow_redirects=self._default_config.follow_redirects
            )
            logger.info("HTTP service client initialized")
    
    async def close(self):
        """Close the HTTP client and clean up resources."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            logger.info("HTTP service client closed")
    
    def _cleanup_cache(self):
        """Clean up expired cache entries."""
        current_time = time.time()
        if current_time - self._last_cache_cleanup < self._cache_cleanup_interval:
            return
            
        expired_keys = []
        for key, cache_entry in self._response_cache.items():
            if current_time - cache_entry.timestamp > cache_entry.ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self._response_cache[key]
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
        
        self._last_cache_cleanup = current_time
    
    def _get_cache_key(self, url: str, method: str = "GET") -> str:
        """Generate cache key for URL and method."""
        return f"{method}:{url}"
    
    async def get(self, url: str, config: Optional[RequestConfig] = None, use_cache: bool = True, as_text: bool = False) -> Tuple[Optional[bytes], str]:
        """
        Perform GET request with caching and retry logic.
        
        Args:
            url: URL to fetch
            config: Request configuration (uses default if None)
            use_cache: Whether to use response caching
            as_text: Whether to return content as text instead of bytes
            
        Returns:
            Tuple of (content, status_code)
        """
        return await self._request("GET", url, config=config, use_cache=use_cache, as_text=as_text)
    
    async def head(self, url: str, config: Optional[RequestConfig] = None) -> Tuple[Optional[httpx.Response], str]:
        """
        Perform HEAD request.
        
        Args:
            url: URL to fetch
            config: Request configuration (uses default if None)
            
        Returns:
            Tuple of (response, status_code)
        """
        return await self._request("HEAD", url, config=config, return_response=True)
    
    async def stream(self, url: str, config: Optional[RequestConfig] = None) -> Tuple[Optional[httpx.Response], str]:
        """
        Perform streaming GET request.
        
        Args:
            url: URL to fetch
            config: Request configuration (uses default if None)
            
        Returns:
            Tuple of (response, status_code)
        """
        return await self._request("GET", url, config=config, return_response=True, stream=True)
    
    async def _request(
        self, 
        method: str, 
        url: str, 
        config: Optional[RequestConfig] = None,
        use_cache: bool = True,
        return_response: bool = False,
        stream: bool = False,
        as_text: bool = False
    ) -> Tuple[Any, str]:
        """
        Perform HTTP request with retry logic and caching.
        
        Args:
            method: HTTP method
            url: URL to fetch
            config: Request configuration
            use_cache: Whether to use response caching
            return_response: Whether to return the response object
            stream: Whether to use streaming
            
        Returns:
            Tuple of (content/response, status_code)
        """
        if config is None:
            config = self._default_config
        
        await self._ensure_client()
        
        # Check cache for GET requests
        if method == "GET" and use_cache and not return_response:
            cache_key = self._get_cache_key(url, method)
            if cache_key in self._response_cache:
                cache_entry = self._response_cache[cache_key]
                if time.time() - cache_entry.timestamp < cache_entry.ttl:
                    logger.debug(f"Cache hit for {url}")
                    return cache_entry.content, "200"
        
        # Validate URL security
        is_safe, reason = validate_url_security(url)
        if not is_safe:
            logger.warning(f"URL rejected: {reason} - {url}")
            return None, f"BLOCKED_{reason}"
        
        # Retry logic with exponential backoff
        last_exception = None
        for attempt in range(config.max_retries + 1):
            try:
                if stream:
                    # For streaming, use the async context manager properly
                    async with self._client.stream(method, url) as response:
                        # Handle redirects manually if follow_redirects is False
                        if not config.follow_redirects and response.status_code in (301, 302, 303, 307, 308):
                            redirect_url = response.headers.get('location')
                            if redirect_url:
                                redirect_url = urljoin(url, redirect_url)
                                # Validate redirect security
                                is_safe, reason = validate_url_security(redirect_url)
                                if not is_safe:
                                    logger.warning(f"Redirect blocked: {reason} - {redirect_url}")
                                    return None, f"REDIRECT_BLOCKED_{reason}"
                                # Follow redirect
                                url = redirect_url
                                continue
                        
                        if return_response:
                            return response, str(response.status_code)
                        
                        # For streaming GET requests, cache the response
                        if method == "GET" and use_cache and response.status_code == 200:
                            content = response.content
                            content_type = response.headers.get('content-type', '')
                            
                            cache_key = self._get_cache_key(url, method)
                            self._response_cache[cache_key] = ResponseCache(
                                url=url,
                                content=content.encode('utf-8') if as_text else content,
                                content_type=content_type,
                                timestamp=time.time()
                            )
                            
                            # Cleanup cache periodically
                            self._cleanup_cache()
                            
                            return content, str(response.status_code)
                        
                        # Return content as text or bytes based on as_text parameter
                        if as_text:
                            return response.text if hasattr(response, 'text') else None, str(response.status_code)
                        else:
                            return response.content if hasattr(response, 'content') else None, str(response.status_code)
                else:
                    response = await self._client.request(method, url)
                
                # Handle redirects manually if follow_redirects is False
                if not config.follow_redirects and response.status_code in (301, 302, 303, 307, 308):
                    redirect_url = response.headers.get('location')
                    if redirect_url:
                        redirect_url = urljoin(url, redirect_url)
                        # Validate redirect security
                        is_safe, reason = validate_url_security(redirect_url)
                        if not is_safe:
                            logger.warning(f"Redirect blocked: {reason} - {redirect_url}")
                            return None, f"REDIRECT_BLOCKED_{reason}"
                        # Follow redirect
                        url = redirect_url
                        continue
                
                if return_response:
                    return response, str(response.status_code)
                
                # For GET requests, cache the response
                if method == "GET" and use_cache and response.status_code == 200:
                    content = response.text if as_text else response.content
                    content_type = response.headers.get('content-type', '')
                    
                    cache_key = self._get_cache_key(url, method)
                    self._response_cache[cache_key] = ResponseCache(
                        url=url,
                        content=content.encode('utf-8') if as_text else content,
                        content_type=content_type,
                        timestamp=time.time()
                    )
                    
                    # Cleanup cache periodically
                    self._cleanup_cache()
                    
                    return content, str(response.status_code)
                
                # Return content as text or bytes based on as_text parameter
                if as_text:
                    return response.text if hasattr(response, 'text') else None, str(response.status_code)
                else:
                    return response.content if hasattr(response, 'content') else None, str(response.status_code)
                
            except httpx.HTTPError as e:
                last_exception = e
                if attempt < config.max_retries:
                    delay = config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"HTTP error on attempt {attempt + 1}/{config.max_retries + 1}: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"HTTP error after {config.max_retries + 1} attempts: {e}")
                    return None, f"HTTP_ERROR_{e.response.status_code if hasattr(e, 'response') else 'UNKNOWN'}"
            
            except Exception as e:
                last_exception = e
                if attempt < config.max_retries:
                    delay = config.retry_delay * (2 ** attempt)
                    logger.warning(f"Request error on attempt {attempt + 1}/{config.max_retries + 1}: {e}. Retrying in {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Request error after {config.max_retries + 1} attempts: {e}")
                    return None, f"REQUEST_ERROR_{str(e)}"
        
        return None, f"MAX_RETRIES_EXCEEDED_{str(last_exception)}"
    
    async def download_image(self, url: str, max_size: int = 10 * 1024 * 1024) -> Tuple[Optional[bytes], str]:
        """
        Download image with size limits and content validation using streaming.
        
        Args:
            url: Image URL
            max_size: Maximum file size in bytes
            
        Returns:
            Tuple of (image_content, status_code)
        """
        config = RequestConfig(
            timeout=60.0,  # Longer timeout for image downloads
            max_retries=2,  # Fewer retries for images
            retry_delay=2.0
        )
        
        try:
            # Use streaming for image downloads with size limits
            async with self._client.stream("GET", url) as response:
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                    logger.warning(f"Non-image content type: {content_type}")
                    return None, "INVALID_CONTENT_TYPE"
                
                # Check content length
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > max_size:
                    logger.warning(f"Image too large: {content_length} bytes")
                    return None, "CONTENT_TOO_LARGE"
                
                # Stream content with size limit
                content = b""
                async for chunk in response.aiter_bytes():
                    content += chunk
                    if len(content) > max_size:
                        logger.warning(f"Image size exceeded limit during download")
                        return None, "CONTENT_TOO_LARGE"
                
                return content, str(response.status_code)
            
        except Exception as e:
            logger.error(f"Error downloading image from {url}: {e}")
            return None, f"DOWNLOAD_ERROR_{str(e)}"
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        current_time = time.time()
        total_entries = len(self._response_cache)
        expired_entries = sum(
            1 for cache_entry in self._response_cache.values()
            if current_time - cache_entry.timestamp > cache_entry.ttl
        )
        
        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "cache_hit_rate": "N/A"  # Would need to track hits/misses
        }
    
    def clear_cache(self):
        """Clear the response cache."""
        self._response_cache.clear()
        logger.info("HTTP service cache cleared")

# Global HTTP service instance
_http_service: Optional[HTTPService] = None

async def get_http_service() -> HTTPService:
    """Get the global HTTP service instance."""
    global _http_service
    if _http_service is None:
        _http_service = HTTPService()
    return _http_service

async def close_http_service():
    """Close the global HTTP service."""
    global _http_service
    if _http_service is not None:
        await _http_service.close()
        _http_service = None
