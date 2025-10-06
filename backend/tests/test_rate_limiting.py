import pytest
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image

from app.main import app
from app.core.rate_limiter import RateLimiter, get_rate_limiter

client = TestClient(app)

def create_test_image():
    """Create a test image for upload tests."""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

def create_large_test_image():
    """Create a large test image (>10MB) for size validation tests."""
    # Create a large image (2000x2000 pixels)
    img = Image.new('RGB', (2000, 2000), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG', quality=95)
    img_bytes.seek(0)
    return img_bytes.getvalue()

@pytest.fixture
def test_image():
    """Fixture for test image."""
    return create_test_image()

@pytest.fixture
def large_test_image():
    """Fixture for large test image."""
    return create_large_test_image()

@pytest.fixture
def valid_tenant_headers():
    """Fixture for valid tenant headers."""
    return {"X-Tenant-ID": "test-tenant-123"}

class TestRateLimiting:
    """Test rate limiting functionality."""
    
    def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
        assert rate_limiter.requests_per_minute == 60
        assert rate_limiter.requests_per_hour == 1000
    
    @patch('app.core.rate_limiter.get_redis_client')
    def test_rate_limiter_with_redis(self, mock_redis):
        """Test rate limiter with Redis backend."""
        # Mock Redis client
        mock_redis_client = MagicMock()
        mock_redis_client.get.return_value = None  # No existing count
        mock_redis_client.pipeline.return_value.__enter__.return_value = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
        
        # Should not be rate limited initially
        assert not rate_limiter.is_rate_limited("tenant-1")
        
        # Increment counter
        rate_limiter.increment_counter("tenant-1")
        
        # Verify Redis operations were called
        mock_redis_client.pipeline.assert_called()
    
    @patch('app.core.rate_limiter.get_redis_client')
    def test_rate_limiter_exceeds_minute_limit(self, mock_redis):
        """Test rate limiter when minute limit is exceeded."""
        # Mock Redis client to return count above limit
        mock_redis_client = MagicMock()
        mock_redis_client.get.side_effect = lambda key: "61" if "minute" in key else None
        mock_redis.return_value = mock_redis_client
        
        rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
        
        # Should be rate limited
        assert rate_limiter.is_rate_limited("tenant-1")
    
    @patch('app.core.rate_limiter.get_redis_client')
    def test_rate_limiter_exceeds_hour_limit(self, mock_redis):
        """Test rate limiter when hour limit is exceeded."""
        # Mock Redis client to return count above limit
        mock_redis_client = MagicMock()
        mock_redis_client.get.side_effect = lambda key: "1001" if "hour" in key else None
        mock_redis.return_value = mock_redis_client
        
        rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
        
        # Should be rate limited
        assert rate_limiter.is_rate_limited("tenant-1")
    
    def test_rate_limiter_per_tenant_isolation(self):
        """Test that rate limiting is isolated per tenant."""
        with patch('app.core.rate_limiter.get_redis_client') as mock_redis:
            # Mock Redis client
            mock_redis_client = MagicMock()
            mock_redis_client.get.return_value = None
            mock_redis_client.pipeline.return_value.__enter__.return_value = MagicMock()
            mock_redis.return_value = mock_redis_client
            
            rate_limiter = RateLimiter(requests_per_minute=60, requests_per_hour=1000)
            
            # Both tenants should not be rate limited initially
            assert not rate_limiter.is_rate_limited("tenant-1")
            assert not rate_limiter.is_rate_limited("tenant-2")
            
            # Increment counter for tenant-1
            rate_limiter.increment_counter("tenant-1")
            
            # tenant-2 should still not be rate limited
            assert not rate_limiter.is_rate_limited("tenant-2")

class TestRateLimitingEndpoints:
    """Test rate limiting on API endpoints."""
    
    @patch('app.core.rate_limiter.get_redis_client')
    def test_rate_limiting_on_index_face(self, mock_redis, test_image, valid_tenant_headers):
        """Test rate limiting on index_face endpoint."""
        # Mock Redis client to simulate rate limit exceeded
        mock_redis_client = MagicMock()
        mock_redis_client.get.side_effect = lambda key: "61" if "minute" in key else None
        mock_redis.return_value = mock_redis_client
        
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.storage.save_raw_and_thumb') as mock_storage, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            response = client.post(
                "/api/index_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Should return 429 Too Many Requests
            assert response.status_code == 429
            assert "Rate limit exceeded" in response.json()["detail"]
    
    @patch('app.core.rate_limiter.get_redis_client')
    def test_rate_limiting_on_search_face(self, mock_redis, test_image, valid_tenant_headers):
        """Test rate limiting on search_face endpoint."""
        # Mock Redis client to simulate rate limit exceeded
        mock_redis_client = MagicMock()
        mock_redis_client.get.side_effect = lambda key: "61" if "minute" in key else None
        mock_redis.return_value = mock_redis_client
        
        response = client.post(
            "/api/search_face",
            headers=valid_tenant_headers,
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        
        # Should return 429 Too Many Requests
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
    
    @patch('app.core.rate_limiter.get_redis_client')
    def test_rate_limiting_on_compare_face(self, mock_redis, test_image, valid_tenant_headers):
        """Test rate limiting on compare_face endpoint."""
        # Mock Redis client to simulate rate limit exceeded
        mock_redis_client = MagicMock()
        mock_redis_client.get.side_effect = lambda key: "61" if "minute" in key else None
        mock_redis.return_value = mock_redis_client
        
        response = client.post(
            "/api/compare_face",
            headers=valid_tenant_headers,
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        
        # Should return 429 Too Many Requests
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]
    
    def test_no_rate_limiting_on_health_endpoints(self):
        """Test that health endpoints are not rate limited."""
        response = client.get("/healthz")
        assert response.status_code == 200
        
        response = client.get("/healthz/detailed")
        # May return 503 if services are down, but not 429 for rate limiting
        assert response.status_code != 429
    
    def test_no_rate_limiting_on_config_endpoint(self):
        """Test that config endpoint is not rate limited."""
        response = client.get("/config")
        assert response.status_code == 200
        assert response.status_code != 429

class TestRequestSizeValidation:
    """Test request size validation functionality."""
    
    def test_valid_image_size_accepted(self, test_image, valid_tenant_headers):
        """Test that valid image size is accepted."""
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.storage.save_raw_and_thumb') as mock_storage, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock services
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100], "det_score": 0.9}]
            mock_face_service.return_value = mock_face
            
            mock_storage.return_value = ("raw-key", "raw-url", "thumb-key", "thumb-url")
            
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vector.return_value = mock_vec
            
            response = client.post(
                "/api/index_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Should not return 413 (Payload Too Large)
            assert response.status_code != 413
    
    def test_large_image_size_rejected(self, large_test_image, valid_tenant_headers):
        """Test that large image size is rejected."""
        # Create headers with content-length indicating large size
        headers = valid_tenant_headers.copy()
        headers["content-length"] = str(11 * 1024 * 1024)  # 11MB
        
        response = client.post(
            "/api/index_face",
            headers=headers,
            files={"file": ("large_test.jpg", large_test_image, "image/jpeg")}
        )
        
        # Should return 413 Payload Too Large
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"] and "10MB" in response.json()["detail"]
    
    def test_size_validation_configurable(self):
        """Test that size validation is configurable."""
        from app.core.config import get_settings
        
        settings = get_settings()
        assert hasattr(settings, 'max_image_size_mb')
        assert settings.max_image_size_mb == 10
        
        # Test that the limit is used in middleware
        assert hasattr(settings, 'max_image_size_bytes')
        assert settings.max_image_size_bytes == 10 * 1024 * 1024
    
    def test_size_validation_on_all_endpoints(self, large_test_image, valid_tenant_headers):
        """Test that size validation applies to all upload endpoints."""
        headers = valid_tenant_headers.copy()
        headers["content-length"] = str(11 * 1024 * 1024)  # 11MB
        
        endpoints = ["/api/index_face", "/api/search_face", "/api/compare_face"]
        
        for endpoint in endpoints:
            response = client.post(
                endpoint,
                headers=headers,
                files={"file": ("large_test.jpg", large_test_image, "image/jpeg")}
            )
            
            # All should return 413 Payload Too Large
            assert response.status_code == 413, f"Endpoint {endpoint} did not reject large file"
            assert "exceeds" in response.json()["detail"]

class TestRateLimitingConfiguration:
    """Test rate limiting configuration."""
    
    def test_rate_limiting_config_from_env(self):
        """Test that rate limiting configuration comes from environment."""
        from app.core.config import get_settings
        
        settings = get_settings()
        assert hasattr(settings, 'rate_limit_per_minute')
        assert hasattr(settings, 'rate_limit_per_hour')
        assert settings.rate_limit_per_minute == 60
        assert settings.rate_limit_per_hour == 1000
    
    def test_rate_limiter_uses_config(self):
        """Test that rate limiter uses configuration values."""
        rate_limiter = get_rate_limiter()
        assert rate_limiter.requests_per_minute == 60
        assert rate_limiter.requests_per_hour == 1000

class TestRateLimitingIntegration:
    """Test rate limiting integration with other middleware."""
    
    @patch('app.core.rate_limiter.get_redis_client')
    def test_rate_limiting_with_tenant_validation(self, mock_redis, test_image):
        """Test that rate limiting works with tenant validation."""
        # Mock Redis client
        mock_redis_client = MagicMock()
        mock_redis_client.get.return_value = None
        mock_redis_client.pipeline.return_value.__enter__.return_value = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        # Test without tenant ID (should fail tenant validation first)
        response = client.post(
            "/api/index_face",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        
        # Should return 400 for missing tenant ID, not 429 for rate limiting
        assert response.status_code == 400
        assert "X-Tenant-ID header is required" in response.json()["detail"]
    
    @patch('app.core.rate_limiter.get_redis_client')
    def test_rate_limiting_with_size_validation(self, mock_redis, large_test_image, valid_tenant_headers):
        """Test that rate limiting works with size validation."""
        # Mock Redis client
        mock_redis_client = MagicMock()
        mock_redis_client.get.return_value = None
        mock_redis_client.pipeline.return_value.__enter__.return_value = MagicMock()
        mock_redis.return_value = mock_redis_client
        
        # Test with large file (should fail size validation first)
        headers = valid_tenant_headers.copy()
        headers["content-length"] = str(11 * 1024 * 1024)  # 11MB
        
        response = client.post(
            "/api/index_face",
            headers=headers,
            files={"file": ("large_test.jpg", large_test_image, "image/jpeg")}
        )
        
        # Should return 413 for size limit, not 429 for rate limiting
        assert response.status_code == 413
        assert "exceeds" in response.json()["detail"] and "10MB" in response.json()["detail"]
    
    def test_rate_limiting_metrics_tracking(self):
        """Test that rate limiting violations are tracked in metrics."""
        from app.core.metrics import get_metrics
        
        metrics = get_metrics()
        
        # Record a rate limit violation
        metrics.record_rate_limit_violation("test-tenant")
        
        # Check that violation is tracked
        assert metrics.get_rate_limit_violations("test-tenant") == 1
        assert metrics.get_rate_limit_violations() == 1
        
        # Record another violation
        metrics.record_rate_limit_violation("test-tenant")
        assert metrics.get_rate_limit_violations("test-tenant") == 2
