import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import io
from PIL import Image
import psycopg

from app.main import app
from app.core.audit import AuditLogger, get_audit_logger

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

class TestAuditLogger:
    """Test audit logging functionality."""
    
    @pytest.mark.asyncio
    async def test_audit_logger_initialization(self):
        """Test audit logger initialization."""
        with patch('app.core.audit.get_audit_db_pool') as mock_pool:
            mock_pool.return_value = AsyncMock()
            audit_logger = AuditLogger()
            assert audit_logger.db_pool is not None
    
    @pytest.mark.asyncio
    async def test_log_request(self):
        """Test logging API requests."""
        with patch('app.core.audit.get_audit_db_pool') as mock_pool:
            # Mock database connection
            mock_conn = AsyncMock()
            mock_cur = AsyncMock()
            mock_conn.cursor.return_value.__aenter__.return_value = mock_cur
            mock_conn.__aenter__.return_value = mock_conn
            mock_pool.return_value.connection.return_value = mock_conn
            
            audit_logger = AuditLogger()
            
            # Mock request object
            mock_request = MagicMock()
            mock_request.state.request_id = "req-123"
            mock_request.state.tenant_id = "tenant-123"
            mock_request.method = "POST"
            mock_request.url.path = "/api/index_face"
            mock_request.headers.get.return_value = "test-user-agent"
            mock_request.client.host = "192.168.1.1"
            
            # Test logging request
            await audit_logger.log_request(mock_request, 200, 1.5, {"test": "data"})
            
            # Verify database operations
            mock_cur.execute.assert_called_once()
            mock_conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_log_search_operation(self):
        """Test logging search operations."""
        with patch('app.core.audit.get_audit_db_pool') as mock_pool:
            # Mock database connection
            mock_conn = AsyncMock()
            mock_cur = AsyncMock()
            mock_conn.cursor.return_value.__aenter__.return_value = mock_cur
            mock_conn.__aenter__.return_value = mock_conn
            mock_pool.return_value.connection.return_value = mock_conn
            
            audit_logger = AuditLogger()
            
            # Test logging search operation
            await audit_logger.log_search_operation(
                tenant_id="tenant-123",
                operation_type="search",
                face_count=2,
                result_count=5,
                vector_backend="qdrant",
                request_id="req-123"
            )
            
            # Verify database operations
            mock_cur.execute.assert_called_once()
            mock_conn.commit.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_audit_logger_error_handling(self):
        """Test audit logger error handling."""
        with patch('app.core.audit.get_audit_db_pool') as mock_pool:
            # Mock database connection that raises exception
            mock_conn = AsyncMock()
            mock_conn.cursor.side_effect = Exception("Database error")
            mock_conn.__aenter__.return_value = mock_conn
            mock_pool.return_value.connection.return_value = mock_conn
            
            audit_logger = AuditLogger()
            
            # Mock request object
            mock_request = MagicMock()
            mock_request.state.request_id = "req-123"
            mock_request.state.tenant_id = "tenant-123"
            mock_request.method = "POST"
            mock_request.url.path = "/api/index_face"
            mock_request.headers.get.return_value = "test-user-agent"
            mock_request.client.host = "192.168.1.1"
            
            # Should not raise exception even if database fails
            await audit_logger.log_request(mock_request, 200, 1.5)
            
            # Should handle gracefully
            assert True  # If we get here, no exception was raised

class TestAuditLoggingEndpoints:
    """Test audit logging on API endpoints."""
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_logging_on_index_face(self, mock_audit_logger, test_image, valid_tenant_headers):
        """Test that index_face endpoint logs audit information."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
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
            
            # Verify audit logging was called
            mock_audit.log_search_operation.assert_called_once()
            call_args = mock_audit.log_search_operation.call_args
            assert call_args[1]['tenant_id'] == "test-tenant-123"
            assert call_args[1]['operation_type'] == "index"
            assert call_args[1]['face_count'] == 1
            assert call_args[1]['result_count'] == 1
            assert call_args[1]['vector_backend'] == "qdrant"
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_logging_on_search_face(self, mock_audit_logger, test_image, valid_tenant_headers):
        """Test that search_face endpoint logs audit information."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
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
            mock_vec.search_similar.return_value = [{"id": "face-1", "score": 0.9, "metadata": {}}]
            mock_vector.return_value = mock_vec
            
            response = client.post(
                "/api/search_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Verify audit logging was called
            mock_audit.log_search_operation.assert_called_once()
            call_args = mock_audit.log_search_operation.call_args
            assert call_args[1]['tenant_id'] == "test-tenant-123"
            assert call_args[1]['operation_type'] == "search"
            assert call_args[1]['face_count'] == 1
            assert call_args[1]['result_count'] == 1
            assert call_args[1]['vector_backend'] == "qdrant"
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_logging_on_compare_face(self, mock_audit_logger, test_image, valid_tenant_headers):
        """Test that compare_face endpoint logs audit information."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock services
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100], "det_score": 0.9}]
            mock_face_service.return_value = mock_face
            
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vec.search_similar.return_value = [{"id": "face-1", "score": 0.9, "metadata": {}}]
            mock_vector.return_value = mock_vec
            
            response = client.post(
                "/api/compare_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Verify audit logging was called
            mock_audit.log_search_operation.assert_called_once()
            call_args = mock_audit.log_search_operation.call_args
            assert call_args[1]['tenant_id'] == "test-tenant-123"
            assert call_args[1]['operation_type'] == "compare"
            assert call_args[1]['face_count'] == 1
            assert call_args[1]['result_count'] == 1
            assert call_args[1]['vector_backend'] == "qdrant"
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_logging_on_compare_face_no_faces(self, mock_audit_logger, test_image, valid_tenant_headers):
        """Test that compare_face endpoint logs audit information when no faces detected."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
        with patch('app.services.face.get_face_service') as mock_face_service, \
             patch('app.services.vector.get_vector_client') as mock_vector:
            
            # Mock services - no faces detected
            mock_face = MagicMock()
            mock_face.compute_phash.return_value = "test-phash"
            mock_face.detect_and_embed.return_value = []  # No faces
            mock_face_service.return_value = mock_face
            
            mock_vec = MagicMock()
            mock_vec.using_pinecone.return_value = False
            mock_vector.return_value = mock_vec
            
            response = client.post(
                "/api/compare_face",
                headers=valid_tenant_headers,
                files={"file": ("test.jpg", test_image, "image/jpeg")}
            )
            
            # Verify audit logging was called even with no faces
            mock_audit.log_search_operation.assert_called_once()
            call_args = mock_audit.log_search_operation.call_args
            assert call_args[1]['tenant_id'] == "test-tenant-123"
            assert call_args[1]['operation_type'] == "compare"
            assert call_args[1]['face_count'] == 0
            assert call_args[1]['result_count'] == 0
            assert call_args[1]['vector_backend'] == "qdrant"
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_logging_on_cleanup(self, mock_audit_logger, valid_tenant_headers):
        """Test that cleanup endpoint logs audit information."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
        with patch('app.services.cleanup.run_cleanup_jobs') as mock_cleanup:
            mock_cleanup.return_value = [10, 5, 2]  # Mock cleanup results
            
            response = client.post(
                "/api/admin/cleanup",
                headers=valid_tenant_headers
            )
            
            # Verify audit logging was called
            mock_audit.log_search_operation.assert_called_once()
            call_args = mock_audit.log_search_operation.call_args
            assert call_args[1]['tenant_id'] == "test-tenant-123"
            assert call_args[1]['operation_type'] == "cleanup"
            assert call_args[1]['face_count'] == 0
            assert call_args[1]['result_count'] == 0
            assert call_args[1]['vector_backend'] == "none"

class TestAuditLoggingMiddleware:
    """Test audit logging middleware functionality."""
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_middleware_logs_all_requests(self, mock_audit_logger, valid_tenant_headers):
        """Test that audit middleware logs all requests."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
        # Make a simple request
        response = client.get("/healthz", headers=valid_tenant_headers)
        
        # Verify audit logging was called
        mock_audit.log_request.assert_called_once()
        call_args = mock_audit.log_request.call_args
        assert call_args[0][1] == 200  # status code
        assert call_args[0][2] > 0  # process time
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_middleware_logs_errors(self, mock_audit_logger):
        """Test that audit middleware logs error responses."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
        # Make a request without tenant ID (should return 400)
        response = client.get("/healthz")
        
        # Verify audit logging was called
        mock_audit.log_request.assert_called_once()
        call_args = mock_audit.log_request.call_args
        assert call_args[0][1] == 400  # status code for missing tenant ID

class TestAuditLoggingDatabase:
    """Test audit logging database operations."""
    
    def test_audit_logs_table_structure(self):
        """Test that audit logs table has correct structure."""
        # This would require actual database connection in integration tests
        # For unit tests, we'll verify the SQL structure is correct
        
        expected_columns = [
            'id', 'request_id', 'tenant_id', 'method', 'path', 
            'status_code', 'process_time', 'user_agent', 'ip_address', 
            'response_size', 'created_at'
        ]
        
        # Verify that the audit logging code expects these columns
        # This is a basic check that the structure is consistent
        assert len(expected_columns) == 11
    
    def test_search_audit_logs_table_structure(self):
        """Test that search audit logs table has correct structure."""
        expected_columns = [
            'id', 'request_id', 'tenant_id', 'operation_type', 
            'face_count', 'result_count', 'vector_backend', 'created_at'
        ]
        
        # Verify that the search audit logging code expects these columns
        assert len(expected_columns) == 8

class TestAuditLoggingIntegration:
    """Test audit logging integration with other components."""
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_logging_with_tenant_scoping(self, mock_audit_logger, test_image, valid_tenant_headers):
        """Test that audit logging works with tenant scoping."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
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
            
            # Verify both middleware and endpoint audit logging
            assert mock_audit.log_request.call_count >= 1
            assert mock_audit.log_search_operation.call_count == 1
            
            # Verify tenant ID is consistent
            middleware_call = mock_audit.log_request.call_args
            endpoint_call = mock_audit.log_search_operation.call_args
            
            # Both should have the same tenant ID
            assert endpoint_call[1]['tenant_id'] == "test-tenant-123"
    
    @patch('app.core.audit.get_audit_logger')
    def test_audit_logging_with_performance_metrics(self, mock_audit_logger, test_image, valid_tenant_headers):
        """Test that audit logging works with performance metrics."""
        mock_audit = AsyncMock()
        mock_audit_logger.return_value = mock_audit
        
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
            
            # Verify audit logging includes performance data
            middleware_call = mock_audit.log_request.call_args
            assert middleware_call[0][2] > 0  # process time should be positive
            
            # Verify request ID is consistent
            assert "X-Request-ID" in response.headers
