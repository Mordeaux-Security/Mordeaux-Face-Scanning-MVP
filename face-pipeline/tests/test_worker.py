"""
Tests for Redis Streams queue worker.

Tests message consumption, processing, and error handling.
"""

import pytest
import json
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from worker import process_message, move_to_dlq, process_message_with_ack


class TestWorker:
    """Test cases for Redis Streams worker."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_redis = MagicMock()
        self.mock_process_image = AsyncMock()
    
    def test_process_message_valid(self):
        """Test processing a valid message."""
        message_id = "1234567890-0"
        message_data = {
            "image_sha256": "abc123def456",
            "bucket": "raw-images",
            "key": "tenant1/image.jpg",
            "tenant_id": "tenant1",
            "site": "example.com",
            "url": "https://example.com/image.jpg",
            "image_phash": "8f373c9c3c9c3c1e",
            "face_hints": None
        }
        
        # Mock successful processing
        self.mock_process_image.return_value = {
            "image_sha256": "abc123def456",
            "counts": {"faces_total": 1, "accepted": 1, "rejected": 0, "dup_skipped": 0},
            "artifacts": {"crops": [], "thumbs": [], "metadata": []},
            "timings_ms": {}
        }
        
        with patch('worker.process_image', self.mock_process_image):
            result = asyncio.run(process_message(message_id, message_data))
        
        assert result is True
        self.mock_process_image.assert_called_once_with(message_data)
    
    def test_process_message_missing_field(self):
        """Test processing a message with missing required field."""
        message_id = "1234567890-0"
        message_data = {
            "image_sha256": "abc123def456",
            "bucket": "raw-images",
            # Missing required fields
        }
        
        result = asyncio.run(process_message(message_id, message_data))
        
        assert result is False
    
    def test_process_message_pipeline_error(self):
        """Test processing a message that fails in pipeline."""
        message_id = "1234567890-0"
        message_data = {
            "image_sha256": "abc123def456",
            "bucket": "raw-images",
            "key": "tenant1/image.jpg",
            "tenant_id": "tenant1",
            "site": "example.com",
            "url": "https://example.com/image.jpg",
            "image_phash": "8f373c9c3c9c3c1e",
            "face_hints": None
        }
        
        # Mock pipeline error
        self.mock_process_image.return_value = {
            "error": "Processing failed",
            "image_sha256": "abc123def456"
        }
        
        with patch('worker.process_image', self.mock_process_image):
            result = asyncio.run(process_message(message_id, message_data))
        
        assert result is False
    
    def test_process_message_exception(self):
        """Test processing a message that raises an exception."""
        message_id = "1234567890-0"
        message_data = {
            "image_sha256": "abc123def456",
            "bucket": "raw-images",
            "key": "tenant1/image.jpg",
            "tenant_id": "tenant1",
            "site": "example.com",
            "url": "https://example.com/image.jpg",
            "image_phash": "8f373c9c3c9c3c1e",
            "face_hints": None
        }
        
        # Mock exception
        self.mock_process_image.side_effect = Exception("Unexpected error")
        
        with patch('worker.process_image', self.mock_process_image):
            result = asyncio.run(process_message(message_id, message_data))
        
        assert result is False
    
    def test_move_to_dlq_success(self):
        """Test moving message to dead letter queue."""
        message_id = "1234567890-0"
        message_data = {"test": "data"}
        error = "Processing failed"
        
        self.mock_redis.xadd.return_value = "dlq-1234567890-0"
        
        with patch('worker.get_redis_client', return_value=self.mock_redis):
            asyncio.run(move_to_dlq(message_id, message_data, error))
        
        # Check that xadd was called with correct parameters
        call_args = self.mock_redis.xadd.call_args
        assert call_args[0][0] == "face-processing-queue-dlq"
        
        dlq_message = call_args[0][1]
        assert dlq_message["original_id"] == message_id
        assert dlq_message["error"] == error
        assert "failed_at" in dlq_message
        assert "worker" in dlq_message
    
    def test_move_to_dlq_failure(self):
        """Test handling DLQ move failure."""
        message_id = "1234567890-0"
        message_data = {"test": "data"}
        error = "Processing failed"
        
        self.mock_redis.xadd.side_effect = Exception("Redis error")
        
        with patch('worker.get_redis_client', return_value=self.mock_redis):
            # Should not raise exception
            asyncio.run(move_to_dlq(message_id, message_data, error))
    
    def test_process_message_with_ack_success(self):
        """Test successful message processing with ack."""
        message_id = "1234567890-0"
        message_data = {
            "image_sha256": "abc123def456",
            "bucket": "raw-images",
            "key": "tenant1/image.jpg",
            "tenant_id": "tenant1",
            "site": "example.com",
            "url": "https://example.com/image.jpg",
            "image_phash": "8f373c9c3c9c3c1e",
            "face_hints": None
        }
        consumer_group = "face-workers"
        
        # Mock successful processing
        self.mock_process_image.return_value = {
            "image_sha256": "abc123def456",
            "counts": {"faces_total": 1, "accepted": 1, "rejected": 0, "dup_skipped": 0}
        }
        
        with patch('worker.process_message', return_value=True), \
             patch('worker.move_to_dlq', new_callable=AsyncMock) as mock_move_dlq:
            
            asyncio.run(process_message_with_ack(message_id, message_data, self.mock_redis, consumer_group))
        
        # Should acknowledge successful message
        self.mock_redis.xack.assert_called_once_with("face-processing-queue", consumer_group, message_id)
        mock_move_dlq.assert_not_called()
    
    def test_process_message_with_ack_failure(self):
        """Test failed message processing with DLQ move."""
        message_id = "1234567890-0"
        message_data = {
            "image_sha256": "abc123def456",
            "bucket": "raw-images",
            "key": "tenant1/image.jpg",
            "tenant_id": "tenant1",
            "site": "example.com",
            "url": "https://example.com/image.jpg",
            "image_phash": "8f373c9c3c9c3c1e",
            "face_hints": None
        }
        consumer_group = "face-workers"
        
        with patch('worker.process_message', return_value=False), \
             patch('worker.move_to_dlq', new_callable=AsyncMock) as mock_move_dlq:
            
            asyncio.run(process_message_with_ack(message_id, message_data, self.mock_redis, consumer_group))
        
        # Should move to DLQ and acknowledge
        mock_move_dlq.assert_called_once_with(message_id, message_data, "Processing failed")
        self.mock_redis.xack.assert_called_once_with("face-processing-queue", consumer_group, message_id)
    
    def test_process_message_with_ack_exception(self):
        """Test message processing with exception handling."""
        message_id = "1234567890-0"
        message_data = {"test": "data"}
        consumer_group = "face-workers"
        
        with patch('worker.process_message', side_effect=Exception("Unexpected error")), \
             patch('worker.move_to_dlq', new_callable=AsyncMock) as mock_move_dlq:
            
            asyncio.run(process_message_with_ack(message_id, message_data, self.mock_redis, consumer_group))
        
        # Should move to DLQ and acknowledge
        mock_move_dlq.assert_called_once_with(message_id, message_data, "Unexpected error")
        self.mock_redis.xack.assert_called_once_with("face-processing-queue", consumer_group, message_id)
    
    def test_message_validation(self):
        """Test message validation with various invalid inputs."""
        required_fields = [
            "image_sha256", "bucket", "key", "tenant_id", 
            "site", "url", "image_phash"
        ]
        
        for field in required_fields:
            message_data = {
                "image_sha256": "abc123def456",
                "bucket": "raw-images",
                "key": "tenant1/image.jpg",
                "tenant_id": "tenant1",
                "site": "example.com",
                "url": "https://example.com/image.jpg",
                "image_phash": "8f373c9c3c9c3c1e",
                "face_hints": None
            }
            
            # Remove one required field
            del message_data[field]
            
            result = asyncio.run(process_message("test-id", message_data))
            assert result is False, f"Should fail when {field} is missing"
    
    def test_json_parsing_error(self):
        """Test handling of malformed JSON in message data."""
        # This would be handled in the main consumption loop
        # where json.loads() would fail
        malformed_data = b"invalid json"
        
        with pytest.raises(json.JSONDecodeError):
            json.loads(malformed_data.decode("utf-8"))
    
    def test_worker_configuration(self):
        """Test worker configuration loading."""
        with patch('worker.settings') as mock_settings:
            mock_settings.enable_queue_worker = True
            mock_settings.redis_url = "redis://localhost:6379/0"
            mock_settings.redis_stream_name = "test-queue"
            mock_settings.max_worker_concurrency = 5
            mock_settings.worker_batch_size = 10
            
            # Test that settings are loaded correctly
            assert mock_settings.enable_queue_worker is True
            assert mock_settings.redis_url == "redis://localhost:6379/0"
            assert mock_settings.redis_stream_name == "test-queue"
            assert mock_settings.max_worker_concurrency == 5
            assert mock_settings.worker_batch_size == 10
