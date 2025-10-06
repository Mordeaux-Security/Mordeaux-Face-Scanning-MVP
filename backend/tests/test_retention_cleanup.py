import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock, AsyncMock
import tempfile
import os

from app.services.cleanup import CleanupService, get_cleanup_service, run_cleanup_jobs
from app.core.config import get_settings

class TestCleanupService:
    """Test cleanup service functionality."""
    
    def test_cleanup_service_initialization(self):
        """Test cleanup service initialization."""
        cleanup_service = CleanupService()
        assert cleanup_service.settings is not None
        assert hasattr(cleanup_service.settings, 'crawled_thumbs_retention_days')
        assert hasattr(cleanup_service.settings, 'user_query_images_retention_hours')
    
    def test_is_crawled_thumbnail(self):
        """Test identification of crawled thumbnails."""
        cleanup_service = CleanupService()
        
        # Test crawled thumbnail patterns
        assert cleanup_service._is_crawled_thumbnail("crawled/2024-01-01/image.jpg")
        assert cleanup_service._is_crawled_thumbnail("web/2024-01-01/image.jpg")
        assert not cleanup_service._is_crawled_thumbnail("tenant-123/image.jpg")
        assert not cleanup_service._is_crawled_thumbnail("query/2024-01-01/image.jpg")
    
    def test_is_user_query_image(self):
        """Test identification of user query images."""
        cleanup_service = CleanupService()
        
        # Test user query image patterns
        assert cleanup_service._is_user_query_image("query/2024-01-01/image.jpg")
        assert cleanup_service._is_user_query_image("search/2024-01-01/image.jpg")
        assert not cleanup_service._is_user_query_image("crawled/2024-01-01/image.jpg")
        assert not cleanup_service._is_user_query_image("tenant-123/image.jpg")
    
    def test_should_delete_crawled_thumbnail(self):
        """Test logic for deleting crawled thumbnails."""
        cleanup_service = CleanupService()
        cutoff_date = datetime.now() - timedelta(days=90)
        
        # Test old crawled thumbnail (should be deleted)
        old_key = "crawled/2023-01-01/image.jpg"
        assert cleanup_service._should_delete_crawled_thumbnail(old_key, cutoff_date)
        
        # Test recent crawled thumbnail (should not be deleted)
        recent_key = "crawled/2024-01-01/image.jpg"
        assert not cleanup_service._should_delete_crawled_thumbnail(recent_key, cutoff_date)
        
        # Test invalid format (should not be deleted for safety)
        invalid_key = "crawled/invalid-format/image.jpg"
        assert not cleanup_service._should_delete_crawled_thumbnail(invalid_key, cutoff_date)
    
    def test_should_delete_user_query_image(self):
        """Test logic for deleting user query images."""
        cleanup_service = CleanupService()
        cutoff_date = datetime.now() - timedelta(hours=24)
        
        # Test old user query image (should be deleted)
        old_key = "query/2023-01-01/image.jpg"
        assert cleanup_service._should_delete_user_query_image(old_key, cutoff_date)
        
        # Test recent user query image (should not be deleted)
        recent_key = "query/2024-01-01/image.jpg"
        assert not cleanup_service._should_delete_user_query_image(recent_key, cutoff_date)
    
    @patch('app.services.cleanup.list_objects')
    @patch('app.services.cleanup.get_object_from_storage')
    async def test_cleanup_crawled_thumbnails(self, mock_get_object, mock_list_objects):
        """Test cleanup of crawled thumbnails."""
        cleanup_service = CleanupService()
        
        # Mock list_objects to return test objects
        mock_list_objects.return_value = [
            "crawled/2023-01-01/old_image.jpg",
            "crawled/2024-01-01/recent_image.jpg",
            "tenant-123/regular_image.jpg"
        ]
        
        with patch.object(cleanup_service, '_delete_object') as mock_delete:
            deleted_count = await cleanup_service.cleanup_crawled_thumbnails()
            
            # Should delete only the old crawled thumbnail
            assert deleted_count == 1
            mock_delete.assert_called_once_with(
                cleanup_service.settings.s3_bucket_thumbs,
                "crawled/2023-01-01/old_image.jpg"
            )
    
    @patch('app.services.cleanup.list_objects')
    @patch('app.services.cleanup.get_object_from_storage')
    async def test_cleanup_user_query_images(self, mock_get_object, mock_list_objects):
        """Test cleanup of user query images."""
        cleanup_service = CleanupService()
        
        # Mock list_objects to return test objects
        mock_list_objects.return_value = [
            "query/2023-01-01/old_image.jpg",
            "query/2024-01-01/recent_image.jpg",
            "tenant-123/regular_image.jpg"
        ]
        
        with patch.object(cleanup_service, '_delete_object') as mock_delete:
            deleted_count = await cleanup_service.cleanup_user_query_images()
            
            # Should delete only the old user query image
            assert deleted_count == 1
            mock_delete.assert_called_once_with(
                cleanup_service.settings.s3_bucket_raw,
                "query/2023-01-01/old_image.jpg"
            )
    
    @patch('app.services.cleanup.psycopg.AsyncConnectionPool')
    async def test_cleanup_audit_logs(self, mock_pool):
        """Test cleanup of audit logs."""
        cleanup_service = CleanupService()
        
        # Mock database connection
        mock_conn = AsyncMock()
        mock_cur = AsyncMock()
        mock_cur.rowcount = 5  # Mock deleted rows
        mock_conn.cursor.return_value.__aenter__.return_value = mock_cur
        mock_conn.__aenter__.return_value = mock_conn
        mock_pool.return_value.connection.return_value = mock_conn
        
        deleted_count = await cleanup_service.cleanup_audit_logs(retention_days=30)
        
        # Should return total deleted count
        assert deleted_count == 10  # 5 + 5 from both tables
        assert mock_cur.execute.call_count == 2  # Two DELETE statements
        mock_conn.commit.assert_called_once()
    
    @patch('app.services.cleanup.psycopg.AsyncConnectionPool')
    async def test_cleanup_audit_logs_error_handling(self, mock_pool):
        """Test cleanup of audit logs with error handling."""
        cleanup_service = CleanupService()
        
        # Mock database connection that raises exception
        mock_conn = AsyncMock()
        mock_conn.cursor.side_effect = Exception("Database error")
        mock_conn.__aenter__.return_value = mock_conn
        mock_pool.return_value.connection.return_value = mock_conn
        
        # Should handle error gracefully
        deleted_count = await cleanup_service.cleanup_audit_logs(retention_days=30)
        assert deleted_count == 0
    
    @patch('app.services.cleanup.Minio')
    async def test_delete_object_minio(self, mock_minio_class):
        """Test deleting object from MinIO."""
        cleanup_service = CleanupService()
        cleanup_service.settings.using_minio = True
        cleanup_service.settings.s3_endpoint = "http://minio:9000"
        cleanup_service.settings.s3_access_key = "test-key"
        cleanup_service.settings.s3_secret_key = "test-secret"
        cleanup_service.settings.s3_use_ssl = False
        
        # Mock MinIO client
        mock_client = MagicMock()
        mock_minio_class.return_value = mock_client
        
        await cleanup_service._delete_object("test-bucket", "test-key")
        
        # Verify MinIO client was called
        mock_minio_class.assert_called_once()
        mock_client.remove_object.assert_called_once_with("test-bucket", "test-key")
    
    @patch('app.services.cleanup.boto3')
    async def test_delete_object_s3(self, mock_boto3):
        """Test deleting object from S3."""
        cleanup_service = CleanupService()
        cleanup_service.settings.using_minio = False
        cleanup_service.settings.s3_region = "us-east-1"
        cleanup_service.settings.s3_access_key = "test-key"
        cleanup_service.settings.s3_secret_key = "test-secret"
        
        # Mock S3 client
        mock_s3_client = MagicMock()
        mock_boto3.client.return_value = mock_s3_client
        
        await cleanup_service._delete_object("test-bucket", "test-key")
        
        # Verify S3 client was called
        mock_boto3.client.assert_called_once()
        mock_s3_client.delete_object.assert_called_once_with(Bucket="test-bucket", Key="test-key")

class TestCleanupJobs:
    """Test cleanup job execution."""
    
    @patch('app.services.cleanup.get_cleanup_service')
    async def test_run_cleanup_jobs(self, mock_get_service):
        """Test running all cleanup jobs."""
        # Mock cleanup service
        mock_service = AsyncMock()
        mock_service.cleanup_crawled_thumbnails.return_value = 10
        mock_service.cleanup_user_query_images.return_value = 5
        mock_service.cleanup_audit_logs.return_value = 2
        mock_get_service.return_value = mock_service
        
        results = await run_cleanup_jobs()
        
        # Verify all cleanup methods were called
        mock_service.cleanup_crawled_thumbnails.assert_called_once()
        mock_service.cleanup_user_query_images.assert_called_once()
        mock_service.cleanup_audit_logs.assert_called_once()
        
        # Verify results
        assert results == [10, 5, 2]
    
    @patch('app.services.cleanup.get_cleanup_service')
    async def test_run_cleanup_jobs_with_exceptions(self, mock_get_service):
        """Test running cleanup jobs with exceptions."""
        # Mock cleanup service with one method raising exception
        mock_service = AsyncMock()
        mock_service.cleanup_crawled_thumbnails.return_value = 10
        mock_service.cleanup_user_query_images.side_effect = Exception("Cleanup error")
        mock_service.cleanup_audit_logs.return_value = 2
        mock_get_service.return_value = mock_service
        
        results = await run_cleanup_jobs()
        
        # Should handle exceptions gracefully
        assert len(results) == 3
        assert results[0] == 10  # First job succeeded
        assert isinstance(results[1], Exception)  # Second job failed
        assert results[2] == 2  # Third job succeeded

class TestCleanupEndpoints:
    """Test cleanup API endpoints."""
    
    def test_cleanup_endpoint_requires_tenant(self):
        """Test that cleanup endpoint requires tenant ID."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Test without tenant ID
        response = client.post("/api/admin/cleanup")
        assert response.status_code == 400
        assert "X-Tenant-ID header is required" in response.json()["detail"]
    
    @patch('app.services.cleanup.run_cleanup_jobs')
    def test_cleanup_endpoint_success(self, mock_cleanup_jobs):
        """Test successful cleanup endpoint execution."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        mock_cleanup_jobs.return_value = [10, 5, 2]
        
        response = client.post(
            "/api/admin/cleanup",
            headers={"X-Tenant-ID": "admin-tenant"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["results"] == [10, 5, 2]
        assert "Cleanup jobs completed successfully" in data["message"]
    
    @patch('app.services.cleanup.run_cleanup_jobs')
    def test_cleanup_endpoint_failure(self, mock_cleanup_jobs):
        """Test cleanup endpoint with failure."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        mock_cleanup_jobs.side_effect = Exception("Cleanup failed")
        
        response = client.post(
            "/api/admin/cleanup",
            headers={"X-Tenant-ID": "admin-tenant"}
        )
        
        assert response.status_code == 500
        assert "Cleanup failed" in response.json()["detail"]

class TestRetentionConfiguration:
    """Test retention configuration."""
    
    def test_retention_configuration(self):
        """Test that retention configuration is properly set."""
        settings = get_settings()
        
        # Test default values
        assert settings.crawled_thumbs_retention_days == 90
        assert settings.user_query_images_retention_hours == 24
        
        # Test that values are used in cleanup service
        cleanup_service = CleanupService()
        assert cleanup_service.settings.crawled_thumbs_retention_days == 90
        assert cleanup_service.settings.user_query_images_retention_hours == 24
    
    def test_retention_configuration_validation(self):
        """Test retention configuration validation."""
        settings = get_settings()
        
        # Test that values are positive
        assert settings.crawled_thumbs_retention_days > 0
        assert settings.user_query_images_retention_hours > 0
        
        # Test that crawled thumbnails have longer retention than user queries
        crawled_days = settings.crawled_thumbs_retention_days
        user_hours = settings.user_query_images_retention_hours
        assert crawled_days * 24 > user_hours

class TestCleanupIntegration:
    """Test cleanup integration with other components."""
    
    @patch('app.services.cleanup.get_cleanup_service')
    def test_cleanup_with_audit_logging(self, mock_get_service):
        """Test that cleanup operations are audited."""
        from fastapi.testclient import TestClient
        from app.main import app
        
        client = TestClient(app)
        
        # Mock cleanup service
        mock_service = AsyncMock()
        mock_service.cleanup_crawled_thumbnails.return_value = 10
        mock_service.cleanup_user_query_images.return_value = 5
        mock_service.cleanup_audit_logs.return_value = 2
        mock_get_service.return_value = mock_service
        
        with patch('app.core.audit.get_audit_logger') as mock_audit_logger:
            mock_audit = AsyncMock()
            mock_audit_logger.return_value = mock_audit
            
            response = client.post(
                "/api/admin/cleanup",
                headers={"X-Tenant-ID": "admin-tenant"}
            )
            
            # Verify audit logging was called
            mock_audit.log_search_operation.assert_called_once()
            call_args = mock_audit.log_search_operation.call_args
            assert call_args[1]['tenant_id'] == "admin-tenant"
            assert call_args[1]['operation_type'] == "cleanup"
            assert call_args[1]['vector_backend'] == "none"
    
    def test_cleanup_service_singleton(self):
        """Test that cleanup service is a singleton."""
        service1 = get_cleanup_service()
        service2 = get_cleanup_service()
        
        assert service1 is service2
    
    def test_cleanup_jobs_async_execution(self):
        """Test that cleanup jobs run asynchronously."""
        import asyncio
        
        async def test_async_cleanup():
            with patch('app.services.cleanup.get_cleanup_service') as mock_get_service:
                mock_service = AsyncMock()
                mock_service.cleanup_crawled_thumbnails.return_value = 10
                mock_service.cleanup_user_query_images.return_value = 5
                mock_service.cleanup_audit_logs.return_value = 2
                mock_get_service.return_value = mock_service
                
                results = await run_cleanup_jobs()
                return results
        
        # Run async test
        results = asyncio.run(test_async_cleanup())
        assert results == [10, 5, 2]
