import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import tempfile
import io
from PIL import Image

from app.main import app
from app.core.config import get_settings

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

@pytest.fixture
def invalid_tenant_headers():
    """Fixture for invalid tenant headers."""
    return {"X-Tenant-ID": "ab"}  # Too short

class TestTenantScoping:
    """Test tenant scoping functionality."""
    
    def test_missing_tenant_id_returns_400(self, test_image):
        """Test that missing X-Tenant-ID header returns 400."""
        response = client.post(
            "/api/index_face",
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "X-Tenant-ID header is required" in response.json()["detail"]
    
    def test_invalid_tenant_id_returns_400(self, test_image, invalid_tenant_headers):
        """Test that invalid X-Tenant-ID header returns 400."""
        response = client.post(
            "/api/index_face",
            headers=invalid_tenant_headers,
            files={"file": ("test.jpg", test_image, "image/jpeg")}
        )
        assert response.status_code == 400
        assert "X-Tenant-ID must be at least 3 characters long" in response.json()["detail"]
    
    def test_valid_tenant_id_accepted(self, test_image, valid_tenant_headers):
        """Test that valid X-Tenant-ID header is accepted."""
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.storage.save_raw_and_thumb') as mock_storage, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock face service
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100], "det_score": 0.9}]
            mock_face_service.return_value = mock_face
            
            # Mock storage
            mock_storage.return_value = ("raw-key", "raw-url", "thumb-key", "thumb-url")
            
            # Mock vector client
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vector.return_value = mock_vec
            
            response = client.post(
                "/api/index_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Should not return 400 (tenant validation passed)
            assert response.status_code != 400
    
    def test_tenant_id_passed_to_services(self, test_image, valid_tenant_headers):
        """Test that tenant_id is passed to storage and vector services."""
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.storage.save_raw_and_thumb') as mock_storage, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock face service
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100], "det_score": 0.9}]
            mock_face_service.return_value = mock_face
            
            # Mock storage
            mock_storage.return_value = ("raw-key", "raw-url", "thumb-key", "thumb-url")
            
            # Mock vector client
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vector.return_value = mock_vec
            
            response = client.post(
                "/api/index_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Verify storage was called with tenant_id
            mock_storage.assert_called_once()
            call_args = mock_storage.call_args
            assert call_args[0][1] == "test-tenant-123"  # tenant_id parameter
            
            # Verify vector client was called with tenant_id
            mock_vec.upsert_embeddings.assert_called_once()
            call_args = mock_vec.upsert_embeddings.call_args
            assert call_args[0][1] == "test-tenant-123"  # tenant_id parameter
    
    def test_tenant_isolation_in_search(self, test_image, valid_tenant_headers):
        """Test that search results are tenant-isolated."""
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.storage.save_raw_and_thumb') as mock_storage, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock face service
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100], "det_score": 0.9}]
            mock_face_service.return_value = mock_face
            
            # Mock storage
            mock_storage.return_value = ("raw-key", "raw-url", "thumb-key", "thumb-url")
            
            # Mock vector client
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vec.search_similar.return_value = [{"id": "face-1", "score": 0.9, "metadata": {}}]
            mock_vector.return_value = mock_vec
            
            response = client.post(
                "/api/search_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Verify search was called with tenant_id
            mock_vec.search_similar.assert_called_once()
            call_args = mock_vec.search_similar.call_args
            assert call_args[0][1] == "test-tenant-123"  # tenant_id parameter
    
    def test_tenant_isolation_in_compare(self, test_image, valid_tenant_headers):
        """Test that compare results are tenant-isolated."""
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock face service
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100], "det_score": 0.9}]
            mock_face_service.return_value = mock_face
            
            # Mock vector client
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vec.search_similar.return_value = [{"id": "face-1", "score": 0.9, "metadata": {}}]
            mock_vector.return_value = mock_vec
            
            response = client.post(
                "/api/compare_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Verify search was called with tenant_id
            mock_vec.search_similar.assert_called_once()
            call_args = mock_vec.search_similar.call_args
            assert call_args[0][1] == "test-tenant-123"  # tenant_id parameter
    
    def test_health_endpoints_dont_require_tenant(self):
        """Test that health check endpoints don't require tenant ID."""
        response = client.get("/healthz")
        assert response.status_code == 200
        
        response = client.get("/healthz/detailed")
        # May return 503 if services are down, but not 400 for missing tenant
        assert response.status_code != 400
    
    def test_config_endpoint_dont_require_tenant(self):
        """Test that config endpoint doesn't require tenant ID."""
        response = client.get("/config")
        assert response.status_code == 200
    
    def test_different_tenants_isolated(self, test_image):
        """Test that different tenants are properly isolated."""
        tenant1_headers = {"X-Tenant-ID": "tenant-1"}
        tenant2_headers = {"X-Tenant-ID": "tenant-2"}
        
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.storage.save_raw_and_thumb') as mock_storage, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock face service
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100], "det_score": 0.9}]
            mock_face_service.return_value = mock_face
            
            # Mock storage
            mock_storage.return_value = ("raw-key", "raw-url", "thumb-key", "thumb-url")
            
            # Mock vector client
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vector.return_value = mock_vec
            
            # Make request with tenant 1
            response1 = client.post(
                "/api/index_face",
                headers=tenant1_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Make request with tenant 2
            response2 = client.post(
                "/api/index_face",
                headers=tenant2_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Both should succeed
            assert response1.status_code != 400
            assert response2.status_code != 400
            
            # Verify different tenant IDs were passed to services
            assert mock_storage.call_count == 2
            assert mock_vec.upsert_embeddings.call_count == 2
            
            # Check that different tenant IDs were used
            call1_tenant = mock_storage.call_args_list[0][0][1]
            call2_tenant = mock_storage.call_args_list[1][0][1]
            assert call1_tenant == "tenant-1"
            assert call2_tenant == "tenant-2"
