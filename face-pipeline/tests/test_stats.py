"""
Tests for statistics tracking service.

Tests Redis-based counter statistics functionality.
"""

import pytest
import redis
from unittest.mock import patch, MagicMock
from pipeline.stats import (
    increment_processed,
    increment_rejected,
    increment_dup_skipped,
    get_stats,
    reset_stats,
    get_all_tenant_stats,
    get_stats_summary,
    health_check
)


class TestStatsService:
    """Test cases for statistics service."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock Redis client
        self.mock_redis = MagicMock()
        self.mock_redis.incrby.return_value = 10
        self.mock_redis.get.return_value = b"5"
        self.mock_redis.delete.return_value = 3
        self.mock_redis.keys.return_value = [b"stats:tenant:tenant1:processed"]
        self.mock_redis.pipeline.return_value.execute.return_value = [10, 5]
        
        # Patch Redis client
        self.redis_patcher = patch('pipeline.stats.get_redis_client', return_value=self.mock_redis)
        self.redis_patcher.start()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.redis_patcher.stop()
    
    def test_increment_processed(self):
        """Test incrementing processed counter."""
        result = increment_processed(5, "tenant1")
        
        assert result == 10
        self.mock_redis.pipeline.assert_called_once()
        pipe = self.mock_redis.pipeline.return_value
        pipe.incrby.assert_any_call("stats:global:processed", 5)
        pipe.incrby.assert_any_call("stats:tenant:tenant1:processed", 5)
        pipe.execute.assert_called_once()
    
    def test_increment_rejected(self):
        """Test incrementing rejected counter."""
        result = increment_rejected(3, "tenant1")
        
        assert result == 10
        self.mock_redis.pipeline.assert_called_once()
        pipe = self.mock_redis.pipeline.return_value
        pipe.incrby.assert_any_call("stats:global:rejected", 3)
        pipe.incrby.assert_any_call("stats:tenant:tenant1:rejected", 3)
        pipe.execute.assert_called_once()
    
    def test_increment_dup_skipped(self):
        """Test incrementing dup_skipped counter."""
        result = increment_dup_skipped(2, "tenant1")
        
        assert result == 10
        self.mock_redis.pipeline.assert_called_once()
        pipe = self.mock_redis.pipeline.return_value
        pipe.incrby.assert_any_call("stats:global:dup_skipped", 2)
        pipe.incrby.assert_any_call("stats:tenant:tenant1:dup_skipped", 2)
        pipe.execute.assert_called_once()
    
    def test_get_stats_global(self):
        """Test getting global statistics."""
        self.mock_redis.pipeline.return_value.execute.return_value = [b"100", b"10", b"5"]
        
        result = get_stats()
        
        expected = {
            "processed": 100,
            "rejected": 10,
            "dup_skipped": 5
        }
        assert result == expected
    
    def test_get_stats_tenant(self):
        """Test getting tenant-specific statistics."""
        self.mock_redis.pipeline.return_value.execute.return_value = [b"50", b"5", b"2"]
        
        result = get_stats("tenant1")
        
        expected = {
            "processed": 50,
            "rejected": 5,
            "dup_skipped": 2
        }
        assert result == expected
    
    def test_get_stats_none_values(self):
        """Test handling of None values from Redis."""
        self.mock_redis.pipeline.return_value.execute.return_value = [None, None, None]
        
        result = get_stats()
        
        expected = {
            "processed": 0,
            "rejected": 0,
            "dup_skipped": 0
        }
        assert result == expected
    
    def test_reset_stats_global(self):
        """Test resetting global statistics."""
        result = reset_stats()
        
        assert result == 3
        self.mock_redis.delete.assert_called_once_with(
            "stats:global:processed",
            "stats:global:rejected", 
            "stats:global:dup_skipped"
        )
    
    def test_reset_stats_tenant(self):
        """Test resetting tenant statistics."""
        result = reset_stats("tenant1")
        
        assert result == 3
        self.mock_redis.delete.assert_called_once_with(
            "stats:tenant:tenant1:processed",
            "stats:tenant:tenant1:rejected",
            "stats:tenant:tenant1:dup_skipped"
        )
    
    def test_get_all_tenant_stats(self):
        """Test getting statistics for all tenants."""
        self.mock_redis.keys.return_value = [b"stats:tenant:tenant1:processed", b"stats:tenant:tenant2:processed"]
        self.mock_redis.pipeline.return_value.execute.side_effect = [
            [b"50", b"5", b"2"],  # tenant1
            [b"30", b"3", b"1"]   # tenant2
        ]
        
        result = get_all_tenant_stats()
        
        expected = {
            "tenant1": {"processed": 50, "rejected": 5, "dup_skipped": 2},
            "tenant2": {"processed": 30, "rejected": 3, "dup_skipped": 1}
        }
        assert result == expected
    
    def test_get_stats_summary(self):
        """Test getting comprehensive statistics summary."""
        # Mock get_stats calls
        with patch('pipeline.stats.get_stats') as mock_get_stats, \
             patch('pipeline.stats.get_all_tenant_stats') as mock_get_all_tenants:
            
            mock_get_stats.return_value = {"processed": 100, "rejected": 10, "dup_skipped": 5}
            mock_get_all_tenants.return_value = {
                "tenant1": {"processed": 60, "rejected": 6, "dup_skipped": 3},
                "tenant2": {"processed": 40, "rejected": 4, "dup_skipped": 2}
            }
            
            result = get_stats_summary()
            
            assert "global" in result
            assert "tenants" in result
            assert "tenant_count" in result
            assert "verification" in result
            assert result["tenant_count"] == 2
    
    def test_health_check_healthy(self):
        """Test health check when Redis is healthy."""
        self.mock_redis.ping.return_value = True
        self.mock_redis.incr.return_value = 1
        self.mock_redis.delete.return_value = 1
        
        result = health_check()
        
        expected = {
            "status": "healthy",
            "redis_connected": True,
            "operations_working": True
        }
        assert result == expected
    
    def test_health_check_unhealthy(self):
        """Test health check when Redis is unhealthy."""
        self.mock_redis.ping.side_effect = redis.RedisError("Connection failed")
        
        result = health_check()
        
        assert result["status"] == "unhealthy"
        assert result["redis_connected"] is False
        assert result["operations_working"] is False
        assert "error" in result
    
    def test_redis_error_handling(self):
        """Test handling of Redis errors."""
        self.mock_redis.pipeline.return_value.execute.side_effect = redis.RedisError("Connection failed")
        
        with pytest.raises(redis.RedisError):
            increment_processed(5, "tenant1")
    
    def test_pipeline_operations(self):
        """Test pipeline operations for counter increments."""
        # Mock pipeline operations
        mock_pipe = MagicMock()
        mock_pipe.incrby.return_value = mock_pipe
        mock_pipe.execute.return_value = [10, 5]
        self.mock_redis.pipeline.return_value = mock_pipe
        
        result = increment_processed(5, "tenant1")
        
        assert result == 10
        mock_pipe.incrby.assert_any_call("stats:global:processed", 5)
        mock_pipe.incrby.assert_any_call("stats:tenant:tenant1:processed", 5)
        mock_pipe.execute.assert_called_once()
    
    def test_batch_operations(self):
        """Test batch operations with multiple increments."""
        # Test multiple increments in sequence
        increment_processed(5, "tenant1")
        increment_rejected(2, "tenant1")
        increment_dup_skipped(1, "tenant1")
        
        # Should have called pipeline 3 times
        assert self.mock_redis.pipeline.call_count == 3
