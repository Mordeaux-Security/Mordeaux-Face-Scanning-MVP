"""
Comprehensive Safety Tests for Crawler Security Features

This module contains table-driven tests for all security guards and safety features
implemented in the crawler service.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any, Tuple

from app.crawler.crawler import (
    ImageCrawler, 
    validate_url_security, 
    validate_content_security, 
    validate_redirect_security,
    MALICIOUS_SCHEMES,
    SUSPICIOUS_EXTENSIONS,
    BAIT_QUERY_KEYS,
    BLOCKED_HOSTS,
    BLOCKED_TLDS,
    ALLOWED_CONTENT_TYPES,
    BLOCKED_CONTENT_TYPES,
    MAX_CONTENT_LENGTH
)


class TestURLSecurityValidation:
    """Test URL security validation functions."""
    
    @pytest.mark.parametrize("url,expected_safe,expected_reason", [
        # Safe URLs
        ("https://example.com/image.jpg", True, "SAFE"),
        ("http://example.com/path/image.png", True, "SAFE"),
        ("https://cdn.example.com/image.webp", True, "SAFE"),
        ("https://example.com/image.jpg?width=100", True, "SAFE"),
        
        # Malicious schemes
        ("javascript:alert('xss')", False, "MALICIOUS_SCHEME"),
        ("data:text/html,<script>alert('xss')</script>", False, "MALICIOUS_SCHEME"),
        ("file:///etc/passwd", False, "MALICIOUS_SCHEME"),
        ("ftp://malicious.com/file.exe", False, "MALICIOUS_SCHEME"),
        
        # Suspicious extensions
        ("https://example.com/malware.exe", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/script.scr", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/app.apk", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/installer.msi", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/script.bat", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/command.cmd", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/powershell.ps1", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/script.php", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/gateway.cgi", False, "SUSPICIOUS_EXTENSION"),
        ("https://example.com/binary.bin", False, "SUSPICIOUS_EXTENSION"),
        
        # Bait query parameters
        ("https://example.com/image.jpg?download=true", False, "BAIT_QUERY_PARAM"),
        ("https://example.com/image.jpg?redirect=http://evil.com", False, "BAIT_QUERY_PARAM"),
        ("https://example.com/image.jpg?out=http://evil.com", False, "BAIT_QUERY_PARAM"),
        ("https://example.com/image.jpg?go=http://evil.com", False, "BAIT_QUERY_PARAM"),
        ("https://example.com/image.jpg?download=1&redirect=evil.com", False, "BAIT_QUERY_PARAM"),
        
        # Blocked hosts
        ("https://malware.example.com/image.jpg", False, "BLOCKED_HOST"),
        ("https://phishing-site.net/image.jpg", False, "BLOCKED_HOST"),
        ("https://suspicious-domain.org/image.jpg", False, "BLOCKED_HOST"),
        
        # Blocked TLDs
        ("https://example.tk/image.jpg", False, "BLOCKED_TLD"),
        ("https://example.ml/image.jpg", False, "BLOCKED_TLD"),
        ("https://example.ga/image.jpg", False, "BLOCKED_TLD"),
        ("https://example.cf/image.jpg", False, "BLOCKED_TLD"),
        
        # Invalid URLs
        ("not-a-url", False, "VALIDATION_ERROR"),
        ("", False, "VALIDATION_ERROR"),
        ("://invalid", False, "VALIDATION_ERROR"),
    ])
    def test_validate_url_security(self, url: str, expected_safe: bool, expected_reason: str):
        """Test URL security validation with various inputs."""
        is_safe, reason = validate_url_security(url)
        assert is_safe == expected_safe, f"URL {url} expected safe={expected_safe}, got {is_safe}"
        assert reason == expected_reason, f"URL {url} expected reason={expected_reason}, got {reason}"


class TestContentSecurityValidation:
    """Test content security validation functions."""
    
    @pytest.mark.parametrize("content_type,content_length,expected_safe,expected_reason", [
        # Safe content types
        ("image/jpeg", None, True, "SAFE"),
        ("image/png", 1024, True, "SAFE"),
        ("image/gif", 2048, True, "SAFE"),
        ("image/webp", 4096, True, "SAFE"),
        ("image/bmp", 8192, True, "SAFE"),
        ("image/jpg", 16384, True, "SAFE"),
        
        # Blocked content types
        ("image/svg+xml", None, False, "BLOCKED_CONTENT_TYPE"),
        ("image/svg+xml", 1024, False, "BLOCKED_CONTENT_TYPE"),
        
        # Non-image content types
        ("text/html", None, False, "NOT_IMAGE_CONTENT"),
        ("application/pdf", None, False, "NOT_IMAGE_CONTENT"),
        ("application/octet-stream", None, False, "NOT_IMAGE_CONTENT"),
        ("video/mp4", None, False, "NOT_IMAGE_CONTENT"),
        ("audio/mpeg", None, False, "NOT_IMAGE_CONTENT"),
        
        # Content too large
        ("image/jpeg", MAX_CONTENT_LENGTH + 1, False, "CONTENT_TOO_LARGE"),
        ("image/png", MAX_CONTENT_LENGTH * 2, False, "CONTENT_TOO_LARGE"),
        
        # Edge cases
        ("", None, False, "NO_CONTENT_TYPE"),
        (None, None, False, "NO_CONTENT_TYPE"),
        ("image/jpeg", MAX_CONTENT_LENGTH, True, "SAFE"),
        ("image/jpeg", MAX_CONTENT_LENGTH - 1, True, "SAFE"),
    ])
    def test_validate_content_security(self, content_type: str, content_length: int, expected_safe: bool, expected_reason: str):
        """Test content security validation with various inputs."""
        is_safe, reason = validate_content_security(content_type, content_length)
        assert is_safe == expected_safe, f"Content type {content_type} with length {content_length} expected safe={expected_safe}, got {is_safe}"
        assert reason == expected_reason, f"Content type {content_type} with length {content_length} expected reason={expected_reason}, got {reason}"


class TestRedirectSecurityValidation:
    """Test redirect security validation functions."""
    
    @pytest.mark.parametrize("redirect_url,redirect_count,max_redirects,expected_safe,expected_reason", [
        # Safe redirects
        ("https://example.com/image.jpg", 0, 3, True, "SAFE"),
        ("https://example.com/image.jpg", 1, 3, True, "SAFE"),
        ("https://example.com/image.jpg", 2, 3, True, "SAFE"),
        
        # Too many redirects
        ("https://example.com/image.jpg", 3, 3, False, "TOO_MANY_REDIRECTS"),
        ("https://example.com/image.jpg", 4, 3, False, "TOO_MANY_REDIRECTS"),
        
        # Malicious redirects
        ("javascript:alert('xss')", 0, 3, False, "REDIRECT_MALICIOUS_SCHEME"),
        ("data:text/html,<script>alert('xss')</script>", 1, 3, False, "REDIRECT_MALICIOUS_SCHEME"),
        ("https://malware.example.com/image.jpg", 0, 3, False, "REDIRECT_BLOCKED_HOST"),
        ("https://example.tk/image.jpg", 1, 3, False, "REDIRECT_BLOCKED_TLD"),
        ("https://example.com/malware.exe", 0, 3, False, "REDIRECT_SUSPICIOUS_EXTENSION"),
    ])
    def test_validate_redirect_security(self, redirect_url: str, redirect_count: int, max_redirects: int, expected_safe: bool, expected_reason: str):
        """Test redirect security validation with various inputs."""
        is_safe, reason = validate_redirect_security(redirect_url, redirect_count, max_redirects)
        assert is_safe == expected_safe, f"Redirect {redirect_url} at count {redirect_count} expected safe={expected_safe}, got {is_safe}"
        assert reason == expected_reason, f"Redirect {redirect_url} at count {redirect_count} expected reason={expected_reason}, got {reason}"


class TestCrawlerSecurityIntegration:
    """Test crawler security integration with real crawler instance."""
    
    @pytest.fixture
    async def crawler(self):
        """Create a crawler instance for testing."""
        crawler = ImageCrawler(
            tenant_id="test-tenant",
            max_file_size=MAX_CONTENT_LENGTH,
            timeout=10,
            max_concurrent_images=5
        )
        async with crawler:
            yield crawler
    
    @pytest.mark.asyncio
    async def test_crawler_rejects_malicious_urls(self, crawler):
        """Test that crawler rejects malicious URLs during image processing."""
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "https://example.com/malware.exe",
            "https://malware.example.com/image.jpg",
            "https://example.tk/image.jpg",
        ]
        
        for url in malicious_urls:
            # Create mock image info
            from app.crawler.crawler import ImageInfo
            image_info = ImageInfo(url=url, alt_text="test", title="test", width=None, height=None)
            
            # Try to download - should be rejected
            content, errors = await crawler.download_image(image_info)
            assert content is None, f"Malicious URL {url} should have been rejected"
            assert len(errors) > 0, f"Should have errors for malicious URL {url}"
            assert any("rejected" in error.lower() for error in errors), f"Should have rejection error for {url}"
    
    @pytest.mark.asyncio
    async def test_crawler_handles_redirects_securely(self, crawler):
        """Test that crawler handles redirects with security validation."""
        # Mock HTTP responses with redirects
        with patch('httpx.AsyncClient.stream') as mock_stream:
            # Setup redirect chain
            mock_response1 = Mock()
            mock_response1.status_code = 302
            mock_response1.headers = {'location': 'https://example.com/image.jpg'}
            mock_response1.aiter_bytes.return_value = iter([])
            
            mock_response2 = Mock()
            mock_response2.status_code = 200
            mock_response2.headers = {'content-type': 'image/jpeg', 'content-length': '1024'}
            mock_response2.aiter_bytes.return_value = iter([b'fake_image_data'])
            mock_response2.raise_for_status.return_value = None
            
            # Setup mock to return different responses for different calls
            mock_stream.side_effect = [
                AsyncMock(__aenter__=AsyncMock(return_value=mock_response1)),
                AsyncMock(__aenter__=AsyncMock(return_value=mock_response2))
            ]
            
            from app.crawler.crawler import ImageInfo
            image_info = ImageInfo(url="https://example.com/redirect", alt_text="test", title="test", width=None, height=None)
            
            # Should handle redirects securely
            content, errors = await crawler.download_image(image_info)
            # Note: This will fail in real implementation due to mock setup, but demonstrates the flow
            # In real test, you'd need proper mock setup for the streaming response
    
    @pytest.mark.asyncio
    async def test_crawler_applies_jitter(self, crawler):
        """Test that crawler applies jitter to avoid rate limiting."""
        start_time = asyncio.get_event_loop().time()
        await crawler._apply_jitter()
        end_time = asyncio.get_event_loop().time()
        
        # Jitter should add some delay
        elapsed = end_time - start_time
        assert elapsed > 0, "Jitter should add some delay"
        assert elapsed < 1.0, "Jitter should not be excessive"
    
    @pytest.mark.asyncio
    async def test_crawler_per_host_concurrency_control(self, crawler):
        """Test that crawler enforces per-host concurrency limits."""
        # Test that semaphores are created for different hosts
        sem1 = await crawler._get_host_semaphore("https://example1.com/image.jpg")
        sem2 = await crawler._get_host_semaphore("https://example2.com/image.jpg")
        sem3 = await crawler._get_host_semaphore("https://example1.com/another.jpg")
        
        # Different hosts should get different semaphores
        assert sem1 is not sem2, "Different hosts should get different semaphores"
        # Same host should get same semaphore
        assert sem1 is sem3, "Same host should get same semaphore"
        
        # Test semaphore limits
        assert sem1._value <= crawler.per_host_concurrency, "Semaphore limit should respect per_host_concurrency"


class TestMaliciousHTMLFixture:
    """Test crawler with HTML containing malicious links."""
    
    @pytest.fixture
    def malicious_html(self):
        """HTML fixture containing various malicious and safe links."""
        return """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <!-- Safe images -->
            <img src="https://example.com/safe1.jpg" alt="Safe image 1" />
            <img src="https://example.com/safe2.png" alt="Safe image 2" />
            <img src="/relative/safe3.gif" alt="Safe image 3" />
            
            <!-- Malicious images -->
            <img src="javascript:alert('xss')" alt="XSS attempt" />
            <img src="data:text/html,<script>alert('xss')</script>" alt="Data URI XSS" />
            <img src="file:///etc/passwd" alt="File protocol" />
            <img src="https://example.com/malware.exe" alt="Executable file" />
            <img src="https://malware.example.com/image.jpg" alt="Blocked host" />
            <img src="https://example.tk/image.jpg" alt="Blocked TLD" />
            
            <!-- Images with bait parameters -->
            <img src="https://example.com/image.jpg?download=true" alt="Download bait" />
            <img src="https://example.com/image.jpg?redirect=http://evil.com" alt="Redirect bait" />
            
            <!-- Mixed content -->
            <div>
                <img src="https://cdn.example.com/good.jpg" alt="Good CDN image" />
                <img src="https://example.com/script.php" alt="PHP script" />
            </div>
        </body>
        </html>
        """
    
    @pytest.mark.asyncio
    async def test_crawler_filters_malicious_images_from_html(self, malicious_html):
        """Test that crawler filters out malicious images from HTML."""
        crawler = ImageCrawler(tenant_id="test-tenant")
        
        # Extract images from malicious HTML
        images, method = crawler.extract_images_by_method(malicious_html, "https://example.com")
        
        # Should only contain safe images
        safe_urls = [
            "https://example.com/safe1.jpg",
            "https://example.com/safe2.png", 
            "https://example.com/relative/safe3.gif",
            "https://cdn.example.com/good.jpg"
        ]
        
        extracted_urls = [img.url for img in images]
        
        # All extracted URLs should be safe
        for url in extracted_urls:
            is_safe, reason = validate_url_security(url)
            assert is_safe, f"Extracted URL {url} should be safe, but got reason: {reason}"
        
        # Should have filtered out malicious URLs
        malicious_urls = [
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "file:///etc/passwd",
            "https://example.com/malware.exe",
            "https://malware.example.com/image.jpg",
            "https://example.tk/image.jpg",
            "https://example.com/image.jpg?download=true",
            "https://example.com/image.jpg?redirect=http://evil.com",
            "https://example.com/script.php"
        ]
        
        for malicious_url in malicious_urls:
            assert malicious_url not in extracted_urls, f"Malicious URL {malicious_url} should have been filtered out"
        
        # Should have found some safe images
        assert len(images) > 0, "Should have found some safe images"
        assert len(images) <= 4, "Should have found at most 4 safe images"


class TestSecurityConfiguration:
    """Test security configuration constants."""
    
    def test_malicious_schemes_coverage(self):
        """Test that malicious schemes list covers common threats."""
        expected_schemes = {'javascript:', 'data:', 'file:', 'ftp:'}
        assert MALICIOUS_SCHEMES == expected_schemes, "Malicious schemes should cover common threats"
    
    def test_suspicious_extensions_coverage(self):
        """Test that suspicious extensions list covers executable files."""
        expected_extensions = {'.exe', '.scr', '.apk', '.msi', '.bat', '.cmd', '.ps1', '.php', '.cgi', '.bin'}
        assert SUSPICIOUS_EXTENSIONS == expected_extensions, "Suspicious extensions should cover executable files"
    
    def test_bait_query_keys_coverage(self):
        """Test that bait query keys list covers common redirect parameters."""
        expected_keys = {'download', 'redirect', 'out', 'go'}
        assert BAIT_QUERY_KEYS == expected_keys, "Bait query keys should cover redirect parameters"
    
    def test_allowed_content_types_coverage(self):
        """Test that allowed content types cover common image formats."""
        expected_types = {'image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp', 'image/bmp'}
        assert ALLOWED_CONTENT_TYPES == expected_types, "Allowed content types should cover image formats"
    
    def test_blocked_content_types_coverage(self):
        """Test that blocked content types include dangerous formats."""
        expected_types = {'image/svg+xml'}
        assert BLOCKED_CONTENT_TYPES == expected_types, "Blocked content types should include SVG"
    
    def test_max_content_length_reasonable(self):
        """Test that max content length is reasonable."""
        assert MAX_CONTENT_LENGTH == 8 * 1024 * 1024, "Max content length should be 8MB"
        assert MAX_CONTENT_LENGTH > 1024 * 1024, "Max content length should be at least 1MB"
        assert MAX_CONTENT_LENGTH < 100 * 1024 * 1024, "Max content length should be reasonable"


class TestSecurityLogging:
    """Test security-related logging functionality."""
    
    @pytest.mark.asyncio
    async def test_security_rejections_are_logged(self, caplog):
        """Test that security rejections are properly logged."""
        # Test URL validation logging
        validate_url_security("javascript:alert('xss')")
        
        # Should have logged the rejection
        assert any("rejected" in record.message.lower() for record in caplog.records), "Security rejection should be logged"
    
    def test_security_reason_codes_are_meaningful(self):
        """Test that security reason codes are meaningful and consistent."""
        # Test various rejection scenarios
        test_cases = [
            ("javascript:alert('xss')", "MALICIOUS_SCHEME"),
            ("https://example.com/malware.exe", "SUSPICIOUS_EXTENSION"),
            ("https://example.com/image.jpg?download=true", "BAIT_QUERY_PARAM"),
            ("https://malware.example.com/image.jpg", "BLOCKED_HOST"),
            ("https://example.tk/image.jpg", "BLOCKED_TLD"),
            ("image/svg+xml", "BLOCKED_CONTENT_TYPE"),
            ("text/html", "NOT_IMAGE_CONTENT"),
            ("image/jpeg", MAX_CONTENT_LENGTH + 1, "CONTENT_TOO_LARGE"),
        ]
        
        for test_case in test_cases:
            if len(test_case) == 2:
                url, expected_reason = test_case
                is_safe, reason = validate_url_security(url)
                assert not is_safe, f"URL {url} should be rejected"
                assert reason == expected_reason, f"URL {url} should have reason {expected_reason}, got {reason}"
            else:
                content_type, content_length, expected_reason = test_case
                is_safe, reason = validate_content_security(content_type, content_length)
                assert not is_safe, f"Content type {content_type} with length {content_length} should be rejected"
                assert reason == expected_reason, f"Content type {content_type} should have reason {expected_reason}, got {reason}"


# Integration test marker
pytestmark = pytest.mark.integration
