"""
Tests for readiness endpoint.

Tests dependency health checks for all system components.
"""

import pytest
from unittest.mock import patch, MagicMock
from main import ready


class TestReadinessEndpoint:
    """Test cases for readiness endpoint."""
    
    def test_ready_all_healthy(self):
        """Test readiness when all dependencies are healthy."""
        # Mock all dependencies as healthy
        with patch('main.load_detector') as mock_load_detector, \
             patch('main.load_model') as mock_load_model, \
             patch('main.get_minio_client') as mock_get_minio, \
             patch('main.get_qdrant_client') as mock_get_qdrant, \
             patch('main.get_redis_client') as mock_get_redis:
            
            # Mock successful operations
            mock_minio_client = MagicMock()
            mock_minio_client.list_buckets.return_value = []
            mock_get_minio.return_value = mock_minio_client
            
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = []
            mock_get_qdrant.return_value = mock_qdrant_client
            
            mock_redis_client = MagicMock()
            mock_redis_client.ping.return_value = True
            mock_get_redis.return_value = mock_redis_client
            
            result = ready()
            
            assert result["ready"] is True
            assert result["reason"] == "ok"
            assert result["checks"]["models"] is True
            assert result["checks"]["storage"] is True
            assert result["checks"]["vector_db"] is True
            assert result["checks"]["redis"] is True
    
    def test_ready_models_failed(self):
        """Test readiness when models fail to load."""
        with patch('main.load_detector', side_effect=Exception("Model load failed")), \
             patch('main.load_model'), \
             patch('main.get_minio_client') as mock_get_minio, \
             patch('main.get_qdrant_client') as mock_get_qdrant, \
             patch('main.get_redis_client') as mock_get_redis:
            
            # Mock other dependencies as healthy
            mock_minio_client = MagicMock()
            mock_minio_client.list_buckets.return_value = []
            mock_get_minio.return_value = mock_minio_client
            
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = []
            mock_get_qdrant.return_value = mock_qdrant_client
            
            mock_redis_client = MagicMock()
            mock_redis_client.ping.return_value = True
            mock_get_redis.return_value = mock_redis_client
            
            result = ready()
            
            assert result["ready"] is False
            assert "models_not_ready" in result["reason"]
            assert result["checks"]["models"] is False
            assert result["checks"]["storage"] is True
            assert result["checks"]["vector_db"] is True
            assert result["checks"]["redis"] is True
    
    def test_ready_storage_failed(self):
        """Test readiness when MinIO storage fails."""
        with patch('main.load_detector'), \
             patch('main.load_model'), \
             patch('main.get_minio_client') as mock_get_minio, \
             patch('main.get_qdrant_client') as mock_get_qdrant, \
             patch('main.get_redis_client') as mock_get_redis:
            
            # Mock storage failure
            mock_get_minio.side_effect = Exception("MinIO connection failed")
            
            # Mock other dependencies as healthy
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = []
            mock_get_qdrant.return_value = mock_qdrant_client
            
            mock_redis_client = MagicMock()
            mock_redis_client.ping.return_value = True
            mock_get_redis.return_value = mock_redis_client
            
            result = ready()
            
            assert result["ready"] is False
            assert "storage_not_ready" in result["reason"]
            assert result["checks"]["models"] is True
            assert result["checks"]["storage"] is False
            assert result["checks"]["vector_db"] is True
            assert result["checks"]["redis"] is True
    
    def test_ready_vector_db_failed(self):
        """Test readiness when Qdrant vector database fails."""
        with patch('main.load_detector'), \
             patch('main.load_model'), \
             patch('main.get_minio_client') as mock_get_minio, \
             patch('main.get_qdrant_client') as mock_get_qdrant, \
             patch('main.get_redis_client') as mock_get_redis:
            
            # Mock other dependencies as healthy
            mock_minio_client = MagicMock()
            mock_minio_client.list_buckets.return_value = []
            mock_get_minio.return_value = mock_minio_client
            
            # Mock vector DB failure
            mock_get_qdrant.side_effect = Exception("Qdrant connection failed")
            
            mock_redis_client = MagicMock()
            mock_redis_client.ping.return_value = True
            mock_get_redis.return_value = mock_redis_client
            
            result = ready()
            
            assert result["ready"] is False
            assert "vector_db_not_ready" in result["reason"]
            assert result["checks"]["models"] is True
            assert result["checks"]["storage"] is True
            assert result["checks"]["vector_db"] is False
            assert result["checks"]["redis"] is True
    
    def test_ready_redis_failed(self):
        """Test readiness when Redis fails."""
        with patch('main.load_detector'), \
             patch('main.load_model'), \
             patch('main.get_minio_client') as mock_get_minio, \
             patch('main.get_qdrant_client') as mock_get_qdrant, \
             patch('main.get_redis_client') as mock_get_redis:
            
            # Mock other dependencies as healthy
            mock_minio_client = MagicMock()
            mock_minio_client.list_buckets.return_value = []
            mock_get_minio.return_value = mock_minio_client
            
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = []
            mock_get_qdrant.return_value = mock_qdrant_client
            
            # Mock Redis failure
            mock_get_redis.side_effect = Exception("Redis connection failed")
            
            result = ready()
            
            assert result["ready"] is False
            assert "redis_not_ready" in result["reason"]
            assert result["checks"]["models"] is True
            assert result["checks"]["storage"] is True
            assert result["checks"]["vector_db"] is True
            assert result["checks"]["redis"] is False
    
    def test_ready_multiple_failures(self):
        """Test readiness when multiple dependencies fail."""
        with patch('main.load_detector', side_effect=Exception("Model failed")), \
             patch('main.load_model'), \
             patch('main.get_minio_client', side_effect=Exception("MinIO failed")), \
             patch('main.get_qdrant_client') as mock_get_qdrant, \
             patch('main.get_redis_client') as mock_get_redis:
            
            # Mock other dependencies as healthy
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = []
            mock_get_qdrant.return_value = mock_qdrant_client
            
            mock_redis_client = MagicMock()
            mock_redis_client.ping.return_value = True
            mock_get_redis.return_value = mock_redis_client
            
            result = ready()
            
            assert result["ready"] is False
            assert "models_not_ready" in result["reason"]
            assert "storage_not_ready" in result["reason"]
            assert result["checks"]["models"] is False
            assert result["checks"]["storage"] is False
            assert result["checks"]["vector_db"] is True
            assert result["checks"]["redis"] is True
    
    def test_ready_partial_failure(self):
        """Test readiness with partial dependency failures."""
        with patch('main.load_detector'), \
             patch('main.load_model'), \
             patch('main.get_minio_client') as mock_get_minio, \
             patch('main.get_qdrant_client') as mock_get_qdrant, \
             patch('main.get_redis_client') as mock_get_redis:
            
            # Mock storage with connection but operation failure
            mock_minio_client = MagicMock()
            mock_minio_client.list_buckets.side_effect = Exception("List buckets failed")
            mock_get_minio.return_value = mock_minio_client
            
            # Mock other dependencies as healthy
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = []
            mock_get_qdrant.return_value = mock_qdrant_client
            
            mock_redis_client = MagicMock()
            mock_redis_client.ping.return_value = True
            mock_get_redis.return_value = mock_redis_client
            
            result = ready()
            
            assert result["ready"] is False
            assert "storage_not_ready" in result["reason"]
            assert result["checks"]["models"] is True
            assert result["checks"]["storage"] is False
            assert result["checks"]["vector_db"] is True
            assert result["checks"]["redis"] is True
    
    def test_ready_response_structure(self):
        """Test that readiness response has correct structure."""
        with patch('main.load_detector'), \
             patch('main.load_model'), \
             patch('main.get_minio_client') as mock_get_minio, \
             patch('main.get_qdrant_client') as mock_get_qdrant, \
             patch('main.get_redis_client') as mock_get_redis:
            
            # Mock all dependencies as healthy
            mock_minio_client = MagicMock()
            mock_minio_client.list_buckets.return_value = []
            mock_get_minio.return_value = mock_minio_client
            
            mock_qdrant_client = MagicMock()
            mock_qdrant_client.get_collections.return_value = []
            mock_get_qdrant.return_value = mock_qdrant_client
            
            mock_redis_client = MagicMock()
            mock_redis_client.ping.return_value = True
            mock_get_redis.return_value = mock_redis_client
            
            result = ready()
            
            # Check response structure
            assert "ready" in result
            assert "reason" in result
            assert "checks" in result
            
            # Check checks structure
            checks = result["checks"]
            assert "models" in checks
            assert "storage" in checks
            assert "vector_db" in checks
            assert "redis" in checks
            
            # Check types
            assert isinstance(result["ready"], bool)
            assert isinstance(result["reason"], str)
            assert isinstance(checks["models"], bool)
            assert isinstance(checks["storage"], bool)
            assert isinstance(checks["vector_db"], bool)
            assert isinstance(checks["redis"], bool)
    
    def test_ready_import_errors(self):
        """Test readiness when import errors occur."""
        with patch('main.load_detector', side_effect=ImportError("Module not found")):
            result = ready()
            
            assert result["ready"] is False
            assert "models_not_ready" in result["reason"]
            assert "ImportError" in result["reason"]
    
    def test_ready_timeout_errors(self):
        """Test readiness when timeout errors occur."""
        with patch('main.get_minio_client') as mock_get_minio:
            mock_minio_client = MagicMock()
            mock_minio_client.list_buckets.side_effect = TimeoutError("Connection timeout")
            mock_get_minio.return_value = mock_minio_client
            
            result = ready()
            
            assert result["ready"] is False
            assert "storage_not_ready" in result["reason"]
            assert "TimeoutError" in result["reason"]
