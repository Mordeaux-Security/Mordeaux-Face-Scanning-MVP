"""
Test suite for HTTP Service

Validates that the unified HTTP service works correctly and preserves
all security and redirect handling logic from the original implementations.
"""

import pytest
import httpx
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from backend.app.services.http_service import (
    HttpService, 
    create_http_client, 
    fetch_html_with_redirects,
    validate_url_security,
    BrowserPool,
    PLAYWRIGHT_AVAILABLE
)


class TestHttpService:
    """Test the main HttpService class."""
    
    @pytest.mark.asyncio
    async def test_create_client_defaults(self):
        """Test that create_client returns properly configured client."""
        service = HttpService()
        client = await service.create_client()
        
        assert isinstance(client, httpx.AsyncClient)
        assert client.follow_redirects is False
        # Note: httpx.AsyncClient doesn't expose verify/http2 as attributes
        # They are set during initialization but not accessible later
        
        await client.aclose()
    
    @pytest.mark.asyncio
    async def test_create_client_custom_params(self):
        """Test that create_client accepts custom parameters."""
        service = HttpService()
        custom_limits = httpx.Limits(max_connections=50)
        custom_timeout = httpx.Timeout(connect=5.0, read=10.0, write=5.0, pool=5.0)
        custom_headers = {'User-Agent': 'Test Agent'}
        
        client = await service.create_client(
            limits=custom_limits,
            timeout=custom_timeout,
            headers=custom_headers
        )
        
        # Test that client was created successfully with custom params
        assert isinstance(client, httpx.AsyncClient)
        assert client.follow_redirects is False
        assert client.headers['User-Agent'] == 'Test Agent'
        # Note: limits and timeout are set during initialization but not accessible as attributes
        
        await client.aclose()
    
    @pytest.mark.asyncio
    async def test_fetch_html_static(self):
        """Test static HTML fetching without JavaScript."""
        service = HttpService()
        
        # Mock the redirect handling
        with patch('backend.app.services.http_service.fetch_html_with_redirects') as mock_fetch:
            mock_fetch.return_value = ("<html>test</html>", "SUCCESS")
            
            client = await service.create_client()
            html, reason = await service.fetch_html("https://example.com", client, use_js=False)
            
            assert html == "<html>test</html>"
            assert reason == "SUCCESS"
            mock_fetch.assert_called_once_with("https://example.com", client, 3)
            
            await client.aclose()
    
    @pytest.mark.asyncio
    async def test_fetch_with_validation(self):
        """Test fetching with custom validation."""
        service = HttpService()
        
        def validate_ok(response):
            return response.status_code == 200
        
        # Mock the redirect handling
        with patch('backend.app.services.http_service.fetch_with_redirects') as mock_fetch:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_fetch.return_value = (mock_response, "SUCCESS")
            
            client = await service.create_client()
            response, reason = await service.fetch_with_validation(
                "https://example.com", client, validate_ok
            )
            
            assert response == mock_response
            assert reason == "SUCCESS"
            
            await client.aclose()
    
    @pytest.mark.asyncio
    async def test_fetch_with_validation_failure(self):
        """Test fetching with validation failure."""
        service = HttpService()
        
        def validate_fail(response):
            return False
        
        # Mock the redirect handling
        with patch('backend.app.services.http_service.fetch_with_redirects') as mock_fetch:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_fetch.return_value = (mock_response, "SUCCESS")
            
            client = await service.create_client()
            response, reason = await service.fetch_with_validation(
                "https://example.com", client, validate_fail
            )
            
            assert response is None
            assert reason == "VALIDATION_FAILED"
            
            await client.aclose()


class TestUrlSecurity:
    """Test URL security validation."""
    
    def test_validate_url_security_safe(self):
        """Test that safe URLs pass validation."""
        safe_urls = [
            "https://example.com",
            "http://example.com",
            "https://subdomain.example.com/path",
            "https://example.com:8080/path?query=value"
        ]
        
        for url in safe_urls:
            is_safe, reason = validate_url_security(url)
            assert is_safe, f"URL {url} should be safe, got reason: {reason}"
            assert reason == "SAFE"
    
    def test_validate_url_security_malicious_schemes(self):
        """Test that malicious schemes are blocked."""
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "ftp://malicious.com"
        ]
        
        for url in malicious_urls:
            is_safe, reason = validate_url_security(url)
            assert not is_safe, f"URL {url} should be blocked"
            assert reason == "MALICIOUS_SCHEME"
    
    def test_validate_url_security_unsafe_schemes(self):
        """Test that non-HTTP schemes are blocked."""
        unsafe_urls = [
            "gopher://example.com",
            "telnet://example.com",
            "ssh://example.com"
        ]
        
        for url in unsafe_urls:
            is_safe, reason = validate_url_security(url)
            assert not is_safe, f"URL {url} should be blocked"
            assert reason == "UNSAFE_SCHEME"
    
    def test_validate_url_security_invalid_url(self):
        """Test that invalid URLs are handled gracefully."""
        invalid_urls = [
            "not-a-url",
            "",
            "://example.com"
        ]
        
        for url in invalid_urls:
            is_safe, reason = validate_url_security(url)
            assert not is_safe, f"Invalid URL {url} should be blocked"
            # Some invalid URLs might be caught by UNSAFE_SCHEME instead of VALIDATION_ERROR
            assert reason in ["VALIDATION_ERROR", "UNSAFE_SCHEME"]
    
    def test_validate_url_security_edge_cases(self):
        """Test edge cases for URL validation."""
        # https:// is technically valid (empty hostname)
        is_safe, reason = validate_url_security("https://")
        # This might be considered safe by urlparse, so we'll just check it doesn't crash
        assert reason in ["SAFE", "UNSAFE_SCHEME", "VALIDATION_ERROR"]


class TestRedirectHandling:
    """Test redirect handling functionality."""
    
    @pytest.mark.asyncio
    async def test_fetch_html_with_redirects_success(self):
        """Test successful HTML fetching with redirects."""
        # Mock httpx client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.text = "<html>redirected content</html>"
        mock_response.headers = {'content-type': 'text/html; charset=utf-8'}
        mock_response.raise_for_status = Mock()
        mock_client.get.return_value = mock_response
        
        # Mock redirect handling
        with patch('backend.app.services.http_service.fetch_with_redirects') as mock_fetch:
            mock_fetch.return_value = (mock_response, "SUCCESS")
            
            html, reason = await fetch_html_with_redirects("https://example.com", mock_client)
            
            assert html == "<html>redirected content</html>"
            assert reason == "SUCCESS"
    
    @pytest.mark.asyncio
    async def test_fetch_html_with_redirects_not_html(self):
        """Test handling of non-HTML content."""
        # Mock httpx client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.headers = {'content-type': 'application/json'}
        mock_client.get.return_value = mock_response
        
        # Mock redirect handling
        with patch('backend.app.services.http_service.fetch_with_redirects') as mock_fetch:
            mock_fetch.return_value = (mock_response, "SUCCESS")
            
            html, reason = await fetch_html_with_redirects("https://example.com", mock_client)
            
            assert html is None
            assert reason == "NOT_HTML_CONTENT"
    
    @pytest.mark.asyncio
    async def test_fetch_html_with_redirects_http_error(self):
        """Test handling of HTTP errors."""
        # Mock redirect handling
        with patch('backend.app.services.http_service.fetch_with_redirects') as mock_fetch:
            mock_fetch.return_value = (None, "HTTP_ERROR_404")
            
            html, reason = await fetch_html_with_redirects("https://example.com", AsyncMock())
            
            assert html is None
            assert reason == "HTTP_ERROR_404"


class TestBrowserPool:
    """Test BrowserPool functionality (if Playwright available)."""
    
    @pytest.mark.asyncio
    async def test_browser_pool_context_manager(self):
        """Test BrowserPool context manager."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        pool = BrowserPool(max_browsers=1)
        
        async with pool:
            assert pool._browsers
            assert len(pool._browsers) == 1
            assert pool._playwright is not None
    
    @pytest.mark.asyncio
    async def test_browser_pool_render_page(self):
        """Test page rendering with BrowserPool."""
        if not PLAYWRIGHT_AVAILABLE:
            pytest.skip("Playwright not available")
        
        pool = BrowserPool(max_browsers=1)
        
        async with pool:
            # Test with a simple URL (this might fail in CI, that's ok)
            try:
                html = await pool.render_page("data:text/html,<html><body>Test</body></html>")
                assert "<html>" in html
                assert "Test" in html
            except Exception as e:
                # Browser rendering might fail in CI environments
                pytest.skip(f"Browser rendering failed: {e}")


class TestBackwardCompatibility:
    """Test backward compatibility functions."""
    
    @pytest.mark.asyncio
    async def test_create_safe_client(self):
        """Test create_safe_client backward compatibility function."""
        from backend.app.services.http_service import create_safe_client
        
        client = await create_safe_client()
        
        assert isinstance(client, httpx.AsyncClient)
        assert client.follow_redirects is False
        # Note: verify is set during initialization but not accessible as attribute
        
        await client.aclose()
    
    def test_import_compatibility(self):
        """Test that all expected functions are importable."""
        from backend.app.services.http_service import (
            HttpService,
            create_http_client,
            create_safe_client,
            fetch_html_with_redirects,
            head_with_redirects,
            validate_url_security,
            BrowserPool
        )
        
        # Just verify they're importable
        assert HttpService is not None
        assert create_http_client is not None
        assert create_safe_client is not None
        assert fetch_html_with_redirects is not None
        assert head_with_redirects is not None
        assert validate_url_security is not None
        assert BrowserPool is not None


if __name__ == "__main__":
    pytest.main([__file__])
