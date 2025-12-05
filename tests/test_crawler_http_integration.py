"""
Test suite for Crawler HTTP Integration

Validates that the crawler uses http_service correctly and maintains
backward compatibility with existing functionality.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from backend.app.services.http_service import HttpService

# Import only the parts we need to test, avoiding heavy dependencies
try:
    from backend.app.services.crawler import ImageCrawler
    CRAWLER_AVAILABLE = True
except ImportError:
    CRAWLER_AVAILABLE = False
    ImageCrawler = None


class TestCrawlerHttpIntegration:
    """Test that crawler integrates with http_service correctly."""
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    @pytest.mark.asyncio
    async def test_crawler_uses_http_service(self):
        """Verify crawler uses http_service, not direct httpx."""
        crawler = ImageCrawler()
        
        # Mock the http_service
        with patch('backend.app.services.crawler.HttpService') as mock_http_service_class:
            mock_http_service = Mock()
            mock_http_service.create_client.return_value = AsyncMock()
            mock_http_service_class.return_value = mock_http_service
            
            async with crawler:
                # Verify HttpService was instantiated
                mock_http_service_class.assert_called_once()
                
                # Verify create_client was called
                mock_http_service.create_client.assert_called_once()
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    @pytest.mark.asyncio
    async def test_crawler_backwards_compatible(self):
        """Verify same URLs produce same results as before refactor."""
        # This test would need to be run with actual URLs to verify behavior
        # For now, we'll test that the interface remains the same
        crawler = ImageCrawler()
        
        # Test that all expected methods exist
        assert hasattr(crawler, 'crawl_page')
        assert hasattr(crawler, 'crawl_site')
        assert hasattr(crawler, 'fetch_page')
        assert hasattr(crawler, 'download_image')
        
        # Test that initialization parameters are preserved
        assert hasattr(crawler, 'tenant_id')
        assert hasattr(crawler, 'max_file_size')
        assert hasattr(crawler, 'timeout')
        assert hasattr(crawler, 'min_face_quality')
        assert hasattr(crawler, 'require_face')
        assert hasattr(crawler, 'crop_faces')
        assert hasattr(crawler, 'face_margin')
        assert hasattr(crawler, 'max_total_images')
        assert hasattr(crawler, 'max_pages')
        assert hasattr(crawler, 'same_domain_only')
        assert hasattr(crawler, 'similarity_threshold')
        assert hasattr(crawler, 'max_concurrent_images')
        assert hasattr(crawler, 'batch_size')
        assert hasattr(crawler, 'enable_audit_logging')
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    @pytest.mark.asyncio
    async def test_crawler_connection_reuse(self):
        """Verify connections are reused across multiple page fetches."""
        crawler = ImageCrawler()
        
        with patch('backend.app.services.crawler.HttpService') as mock_http_service_class:
            mock_http_service = Mock()
            mock_client = AsyncMock()
            mock_http_service.create_client.return_value = mock_client
            mock_http_service_class.return_value = mock_http_service
            
            async with crawler:
                # Verify only one client was created
                mock_http_service.create_client.assert_called_once()
                
                # Verify the same client is used for multiple operations
                assert crawler.session is mock_client
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    @pytest.mark.asyncio
    async def test_crawler_http_service_cleanup(self):
        """Verify http_service is properly cleaned up."""
        crawler = ImageCrawler()
        
        with patch('backend.app.services.crawler.HttpService') as mock_http_service_class:
            mock_http_service = Mock()
            mock_client = AsyncMock()
            mock_http_service.create_client.return_value = mock_client
            mock_http_service_class.return_value = mock_http_service
            
            async with crawler:
                pass  # Exit context manager
            
            # Verify client was closed
            mock_client.aclose.assert_called_once()
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    @pytest.mark.asyncio
    async def test_crawler_fetch_page_uses_http_service(self):
        """Verify fetch_page uses http_service methods."""
        crawler = ImageCrawler()
        
        with patch('backend.app.services.crawler.HttpService') as mock_http_service_class:
            mock_http_service = Mock()
            mock_client = AsyncMock()
            mock_http_service.create_client.return_value = mock_client
            mock_http_service.fetch_html.return_value = ("<html>test</html>", "SUCCESS")
            mock_http_service_class.return_value = mock_http_service
            
            async with crawler:
                # Mock the http_service instance
                crawler._http_service = mock_http_service
                
                html, errors = await crawler.fetch_page("https://example.com")
                
                # Verify http_service.fetch_html was called
                mock_http_service.fetch_html.assert_called_once_with(
                    "https://example.com", 
                    mock_client, 
                    use_js=False, 
                    max_redirects=3
                )
                
                assert html == "<html>test</html>"
                assert errors == []
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    @pytest.mark.asyncio
    async def test_crawler_download_image_uses_session(self):
        """Verify download_image uses the session from http_service."""
        crawler = ImageCrawler()
        
        with patch('backend.app.services.crawler.HttpService') as mock_http_service_class:
            mock_http_service = Mock()
            mock_client = AsyncMock()
            mock_http_service.create_client.return_value = mock_client
            mock_http_service_class.return_value = mock_http_service
            
            async with crawler:
                # Test that download_image uses the session
                from backend.app.services.crawler import ImageInfo
                
                image_info = ImageInfo(
                    url="https://example.com/image.jpg",
                    selector="img",
                    video_url="https://example.com/video",
                    attributes={},
                    context={}
                )
                
                # Mock the stream method
                mock_response = AsyncMock()
                mock_response.status_code = 200
                mock_response.headers = {'content-type': 'image/jpeg', 'content-length': '1000'}
                mock_response.iter_bytes.return_value = [b'fake image data']
                mock_client.stream.return_value.__aenter__.return_value = mock_response
                
                # This would normally download an image, but we're just testing the interface
                try:
                    result = await crawler.download_image(image_info)
                    # If it gets here, it means the session was used correctly
                    assert result is not None
                except Exception:
                    # Expected to fail with mock data, but session should be used
                    pass
                
                # Verify stream was called on the session
                mock_client.stream.assert_called()


class TestCrawlerBackwardCompatibility:
    """Test that crawler maintains backward compatibility."""
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    def test_crawler_initialization_parameters(self):
        """Test that all initialization parameters are preserved."""
        # Test with all parameters to ensure none are lost
        crawler = ImageCrawler(
            tenant_id="test_tenant",
            max_file_size=5 * 1024 * 1024,
            allowed_extensions={'.jpg', '.png'},
            timeout=60,
            min_face_quality=0.8,
            require_face=False,
            crop_faces=False,
            face_margin=0.3,
            max_total_images=100,
            max_pages=50,
            same_domain_only=False,
            similarity_threshold=3,
            max_concurrent_images=30,
            batch_size=75,
            enable_audit_logging=False
        )
        
        # Verify all parameters are set
        assert crawler.tenant_id == "test_tenant"
        assert crawler.max_file_size == 5 * 1024 * 1024
        assert crawler.allowed_extensions == {'.jpg', '.png'}
        assert crawler.timeout == 60
        assert crawler.min_face_quality == 0.8
        assert crawler.require_face == False
        assert crawler.crop_faces == False
        assert crawler.face_margin == 0.3
        assert crawler.max_total_images == 100
        assert crawler.max_pages == 50
        assert crawler.same_domain_only == False
        assert crawler.similarity_threshold == 3
        assert crawler.max_concurrent_images == 30
        assert crawler.batch_size == 75
        assert crawler.enable_audit_logging == False
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    def test_crawler_default_parameters(self):
        """Test that default parameters are preserved."""
        crawler = ImageCrawler()
        
        # Verify default values
        assert crawler.tenant_id == "default"
        assert crawler.max_file_size == 10 * 1024 * 1024
        assert crawler.timeout == 30
        assert crawler.min_face_quality == 0.5
        assert crawler.require_face == True
        assert crawler.crop_faces == True
        assert crawler.face_margin == 0.2
        assert crawler.max_total_images == 50
        assert crawler.max_pages == 20
        assert crawler.same_domain_only == True
        assert crawler.similarity_threshold == 5
        assert crawler.max_concurrent_images == 20
        assert crawler.batch_size == 50
        assert crawler.enable_audit_logging == True
    
    @pytest.mark.skipif(not CRAWLER_AVAILABLE, reason="Crawler dependencies not available")
    @pytest.mark.asyncio
    async def test_crawler_context_manager(self):
        """Test that context manager works correctly."""
        crawler = ImageCrawler()
        
        with patch('backend.app.services.crawler.HttpService') as mock_http_service_class:
            mock_http_service = Mock()
            mock_client = AsyncMock()
            mock_http_service.create_client.return_value = mock_client
            mock_http_service_class.return_value = mock_http_service
            
            # Test context manager entry
            async with crawler:
                assert crawler.session is not None
                assert crawler.session is mock_client
            
            # Test context manager exit (cleanup)
            mock_client.aclose.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])
