import pytest
import time
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import io
from PIL import Image
import statistics

from app.main import app
from app.core.metrics import PerformanceMetrics, get_metrics

client = TestClient(app)

def create_test_image():
    """Create a test image for upload tests."""
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    return img_bytes.getvalue()

@pytest.fixture
def test_image():
    """Fixture for test image."""
    return create_test_image()

@pytest.fixture
def valid_tenant_headers():
    """Fixture for valid tenant headers."""
    return {"X-Tenant-ID": "test-tenant-123"}

class TestPerformanceMetrics:
    """Test performance metrics collection and P95 latency tracking."""
    
    def test_metrics_initialization(self):
        """Test that metrics are properly initialized."""
        metrics = PerformanceMetrics(max_samples=100)
        assert metrics.max_samples == 100
        assert len(metrics.request_times['all']) == 0
        assert len(metrics.endpoint_times) == 0
        assert len(metrics.tenant_times) == 0
    
    def test_record_request(self):
        """Test recording request metrics."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Record some requests
        metrics.record_request("POST /api/index_face", "tenant-1", 1.0, 200, "req-1")
        metrics.record_request("POST /api/index_face", "tenant-1", 2.0, 200, "req-2")
        metrics.record_request("POST /api/search_face", "tenant-2", 1.5, 200, "req-3")
        
        # Check counts
        assert metrics.get_request_count() == 3
        assert metrics.get_request_count("POST /api/index_face") == 2
        assert metrics.get_request_count("tenant-1") == 2
        assert metrics.get_request_count("tenant-2") == 1
        
        # Check timing data
        assert len(metrics.request_times['all']) == 3
        assert len(metrics.endpoint_times["POST /api/index_face"]) == 2
        assert len(metrics.tenant_times["tenant-1"]) == 2
    
    def test_p95_latency_calculation(self):
        """Test P95 latency calculation."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Add 100 requests with known latencies
        latencies = [i * 0.1 for i in range(1, 101)]  # 0.1s to 10.0s
        for i, latency in enumerate(latencies):
            metrics.record_request("POST /api/test", "tenant-1", latency, 200, f"req-{i}")
        
        # P95 should be around 9.5s (95th percentile of 0.1s to 10.0s)
        p95 = metrics.get_p95_latency()
        assert p95 is not None
        assert 9.4 <= p95 <= 9.6  # Allow small tolerance
    
    def test_p99_latency_calculation(self):
        """Test P99 latency calculation."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Add 100 requests with known latencies
        latencies = [i * 0.1 for i in range(1, 101)]  # 0.1s to 10.0s
        for i, latency in enumerate(latencies):
            metrics.record_request("POST /api/test", "tenant-1", latency, 200, f"req-{i}")
        
        # P99 should be around 9.9s
        p99 = metrics.get_p99_latency()
        assert p99 is not None
        assert 9.8 <= p99 <= 10.0  # Allow small tolerance
    
    def test_avg_latency_calculation(self):
        """Test average latency calculation."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Add requests with known latencies
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, latency in enumerate(latencies):
            metrics.record_request("POST /api/test", "tenant-1", latency, 200, f"req-{i}")
        
        # Average should be 3.0
        avg = metrics.get_avg_latency()
        assert avg is not None
        assert abs(avg - 3.0) < 0.01
    
    def test_median_latency_calculation(self):
        """Test median latency calculation."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Add requests with known latencies
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]
        for i, latency in enumerate(latencies):
            metrics.record_request("POST /api/test", "tenant-1", latency, 200, f"req-{i}")
        
        # Median should be 3.0
        median = metrics.get_median_latency()
        assert median is not None
        assert abs(median - 3.0) < 0.01
    
    def test_threshold_exceeded_detection(self):
        """Test P95 threshold exceeded detection."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Add requests that exceed threshold (default 5.0s)
        latencies = [6.0, 7.0, 8.0, 9.0, 10.0]  # All above 5.0s
        for i, latency in enumerate(latencies):
            metrics.record_request("POST /api/test", "tenant-1", latency, 200, f"req-{i}")
        
        # Should detect threshold exceeded
        assert metrics.is_p95_threshold_exceeded()
        
        # Add requests below threshold
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]  # All at or below 5.0s
        for i, latency in enumerate(latencies):
            metrics.record_request("POST /api/test", "tenant-1", latency, 200, f"req-{i+5}")
        
        # Should still detect threshold exceeded due to high P95
        assert metrics.is_p95_threshold_exceeded()
    
    def test_error_counting(self):
        """Test error counting."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Add requests with different status codes
        metrics.record_request("POST /api/test", "tenant-1", 1.0, 200, "req-1")
        metrics.record_request("POST /api/test", "tenant-1", 1.0, 400, "req-2")
        metrics.record_request("POST /api/test", "tenant-1", 1.0, 500, "req-3")
        
        # Should count 2 errors (400 and 500)
        assert metrics.get_error_count() == 2
        assert metrics.get_error_count("POST /api/test:400") == 1
        assert metrics.get_error_count("POST /api/test:500") == 1
    
    def test_rate_limit_violation_tracking(self):
        """Test rate limit violation tracking."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Record rate limit violations
        metrics.record_rate_limit_violation("tenant-1")
        metrics.record_rate_limit_violation("tenant-1")
        metrics.record_rate_limit_violation("tenant-2")
        
        # Check counts
        assert metrics.get_rate_limit_violations() == 3
        assert metrics.get_rate_limit_violations("tenant-1") == 2
        assert metrics.get_rate_limit_violations("tenant-2") == 1
    
    def test_metrics_summary(self):
        """Test comprehensive metrics summary."""
        metrics = PerformanceMetrics(max_samples=100)
        
        # Add some test data
        metrics.record_request("POST /api/index_face", "tenant-1", 1.0, 200, "req-1")
        metrics.record_request("POST /api/index_face", "tenant-1", 2.0, 200, "req-2")
        metrics.record_request("POST /api/search_face", "tenant-2", 1.5, 400, "req-3")
        metrics.record_rate_limit_violation("tenant-1")
        
        summary = metrics.get_metrics_summary()
        
        # Check structure
        assert 'latency' in summary
        assert 'requests' in summary
        assert 'endpoints' in summary
        assert 'tenants' in summary
        
        # Check latency metrics
        assert 'p95' in summary['latency']
        assert 'p99' in summary['latency']
        assert 'avg' in summary['latency']
        assert 'median' in summary['latency']
        assert 'threshold_exceeded' in summary['latency']
        
        # Check request counts
        assert summary['requests']['total'] == 3
        assert summary['requests']['errors'] == 1
        assert summary['requests']['rate_limit_violations'] == 1
        
        # Check endpoint data
        assert "POST /api/index_face" in summary['endpoints']
        assert "POST /api/search_face" in summary['endpoints']
        
        # Check tenant data
        assert "tenant-1" in summary['tenants']
        assert "tenant-2" in summary['tenants']

class TestPerformanceEndpoints:
    """Test performance-related API endpoints."""
    
    def test_metrics_endpoint(self, valid_tenant_headers):
        """Test /metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        
        data = response.json()
        assert 'latency' in data
        assert 'requests' in data
        assert 'endpoints' in data
        assert 'tenants' in data
    
    def test_p95_metrics_endpoint(self, valid_tenant_headers):
        """Test /metrics/p95 endpoint."""
        response = client.get("/metrics/p95")
        assert response.status_code == 200
        
        data = response.json()
        assert 'p95_latency' in data
        assert 'p99_latency' in data
        assert 'avg_latency' in data
        assert 'median_latency' in data
        assert 'threshold_exceeded' in data
        assert 'threshold_seconds' in data
    
    def test_performance_under_load(self, test_image, valid_tenant_headers):
        """Test performance under simulated load."""
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.storage.save_raw_and_thumb') as mock_storage, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock services to simulate realistic processing time
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100], "det_score": 0.9}]
            mock_face_service.return_value = mock_face
            
            mock_storage.return_value = ("raw-key", "raw-url", "thumb-key", "thumb-url")
            
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vector.return_value = mock_vec
            
            # Simulate multiple requests
            start_time = time.time()
            response_times = []
            
            for i in range(10):
                response = client.post(
                    "/api/index_face",
                    headers=valid_tenant_headers,
                    files={"file": ("test.jpg", test_image, "image/jpeg")}
                )
                response_times.append(response.elapsed.total_seconds())
            
            total_time = time.time() - start_time
            
            # Check that all requests succeeded
            assert all(rt < 5.0 for rt in response_times), f"Some requests exceeded 5s: {response_times}"
            
            # Check that total time is reasonable
            assert total_time < 10.0, f"Total time exceeded 10s: {total_time}"
            
            # Check metrics endpoint shows the requests
            metrics_response = client.get("/metrics")
            assert metrics_response.status_code == 200
            metrics_data = metrics_response.json()
            assert metrics_data['requests']['total'] >= 10
    
    def test_p95_threshold_configuration(self):
        """Test that P95 threshold is configurable."""
        from app.core.config import get_settings
        
        settings = get_settings()
        assert hasattr(settings, 'p95_latency_threshold_seconds')
        assert settings.p95_latency_threshold_seconds == 5.0
        
        # Test that threshold is used in metrics
        metrics = get_metrics()
        assert metrics.settings.p95_latency_threshold_seconds == 5.0

class TestPerformanceMonitoring:
    """Test performance monitoring integration."""
    
    def test_middleware_performance_tracking(self, test_image, valid_tenant_headers):
        """Test that middleware tracks performance metrics."""
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
            
            # Make request
            response = client.post(
                "/api/index_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Check that request ID is in response headers
            assert "X-Request-ID" in response.headers
            
            # Check that metrics were recorded
            metrics_response = client.get("/metrics")
            assert metrics_response.status_code == 200
            metrics_data = metrics_response.json()
            assert metrics_data['requests']['total'] >= 1
    
    def test_performance_warning_logging(self):
        """Test that performance warnings are logged when threshold is exceeded."""
        # This would require more complex setup to test actual logging
        # For now, we'll test that the metrics system can detect threshold violations
        metrics = PerformanceMetrics(max_samples=100)
        
        # Add requests that exceed threshold
        latencies = [6.0, 7.0, 8.0, 9.0, 10.0]
        for i, latency in enumerate(latencies):
            metrics.record_request("POST /api/test", "tenant-1", latency, 200, f"req-{i}")
        
        # Should detect threshold exceeded
        assert metrics.is_p95_threshold_exceeded()
        
        # Check that threshold exceeded is in summary
        summary = metrics.get_metrics_summary()
        assert summary['latency']['threshold_exceeded'] is True
