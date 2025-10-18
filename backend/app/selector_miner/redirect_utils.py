"""
Redirect handling utilities for safer HTTP requests.

This module provides utilities for manual redirect handling with security guards
to prevent malicious redirects and enforce redirect limits.
"""

import logging
from typing import Optional, Tuple, List
from urllib.parse import urljoin, urlparse
import httpx

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


async def fetch_with_redirects(
    url: str, 
    client: httpx.AsyncClient, 
    max_hops: int = 3,
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
            if response.status_code in (301, 302, 303, 307, 308):
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
    max_hops: int = 3
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
    max_hops: int = 3
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


def create_safe_client(**kwargs) -> httpx.AsyncClient:
    """
    Create an httpx.AsyncClient with safe defaults for redirect handling.
    
    Args:
        **kwargs: Additional arguments for httpx.AsyncClient
        
    Returns:
        httpx.AsyncClient with follow_redirects=False
    """
    # Ensure follow_redirects is False
    kwargs['follow_redirects'] = False
    
    # Set safe defaults if not provided
    if 'timeout' not in kwargs:
        kwargs['timeout'] = httpx.Timeout(30.0)
    
    if 'verify' not in kwargs:
        kwargs['verify'] = True
    
    return httpx.AsyncClient(**kwargs)
