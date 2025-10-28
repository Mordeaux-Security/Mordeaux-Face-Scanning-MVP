"""
Tests for global deduplication service.

Tests Redis-based pHash deduplication functionality.
"""

import pytest
import redis
from unittest.mock import patch, MagicMock
from pipeline.dedup import (
    is_duplicate, 
    mark_processed, 
    clear_dedup_cache, 
    get_dedup_stats,
    health_check
)
from config.settings import settings


class TestDedupService:
    """Test cases for deduplication service."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock Redis client
        self.mock_redis = MagicMock()
        self.mock_redis.sismember.return_value = False
        self.mock_redis.sadd.return_value = 1
        self.mock_redis.expire.return_value = True
        self.mock_redis.pipeline.return_value.execute.return_value = [1, True]
        
        # Patch Redis client
        self.redis_patcher = patch('pipeline.dedup.get_redis_client', return_value=self.mock_redis)
        self.redis_patcher.start()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.redis_patcher.stop()
    
    def test_is_duplicate_new_hash(self):
        """Test checking a new (non-duplicate) pHash."""
        result = is_duplicate("8f373c9c3c9c3c1e")
        
        assert result is False
        self.mock_redis.sismember.assert_called_once_with("dedup:phash:8f37", "8f373c9c3c9c3c1e")
    
    def test_is_duplicate_existing_hash(self):
        """Test checking an existing (duplicate) pHash."""
        self.mock_redis.sismember.return_value = True
        
        result = is_duplicate("8f373c9c3c9c3c1e")
        
        assert result is True
        self.mock_redis.sismember.assert_called_once_with("dedup:phash:8f37", "8f373c9c3c9c3c1e")
    
    def test_is_duplicate_disabled(self):
        """Test that dedup check returns False when disabled."""
        with patch.object(settings, 'enable_global_dedup', False):
            result = is_duplicate("8f373c9c3c9c3c1e")
            assert result is False
            self.mock_redis.sismember.assert_not_called()
    
    def test_is_duplicate_invalid_hash(self):
        """Test handling of invalid pHash format."""
        result = is_duplicate("invalid")
        assert result is False
        self.mock_redis.sismember.assert_not_called()
    
    def test_mark_processed_success(self):
        """Test marking a pHash as processed."""
        result = mark_processed("8f373c9c3c9c3c1e")
        
        assert result is True
        self.mock_redis.pipeline.assert_called_once()
        pipe = self.mock_redis.pipeline.return_value
        pipe.sadd.assert_called_once_with("dedup:phash:8f37", "8f373c9c3c9c3c1e")
        pipe.expire.assert_called_once_with("dedup:phash:8f37", settings.dedup_ttl_seconds)
    
    def test_mark_processed_disabled(self):
        """Test that mark_processed returns True when disabled."""
        with patch.object(settings, 'enable_global_dedup', False):
            result = mark_processed("8f373c9c3c9c3c1e")
            assert result is True
            self.mock_redis.pipeline.assert_not_called()
    
    def test_mark_processed_invalid_hash(self):
        """Test handling of invalid pHash format."""
        result = mark_processed("invalid")
        assert result is False
        self.mock_redis.pipeline.assert_not_called()
    
    def test_clear_dedup_cache(self):
        """Test clearing deduplication cache."""
        self.mock_redis.keys.return_value = [
            b"dedup:phash:8f37",
            b"dedup:phash:9f38"
        ]
        self.mock_redis.delete.return_value = 2
        
        result = clear_dedup_cache()
        
        assert result == 2
        self.mock_redis.keys.assert_called_once_with("dedup:phash:*")
        self.mock_redis.delete.assert_called_once_with("dedup:phash:8f37", "dedup:phash:9f38")
    
    def test_get_dedup_stats(self):
        """Test getting deduplication statistics."""
        self.mock_redis.keys.return_value = [b"dedup:phash:8f37", b"dedup:phash:9f38"]
        self.mock_redis.scard.side_effect = [5, 3]
        self.mock_redis.ttl.side_effect = [3600, 7200]
        
        result = get_dedup_stats()
        
        expected = {
            "total_keys": 2,
            "total_hashes": 8,
            "prefixes": [
                {"prefix": "8f37", "hash_count": 5, "ttl_seconds": 3600},
                {"prefix": "9f38", "hash_count": 3, "ttl_seconds": 7200}
            ]
        }
        assert result == expected
    
    def test_health_check_healthy(self):
        """Test health check when Redis is healthy."""
        self.mock_redis.ping.return_value = True
        
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
        self.mock_redis.sismember.side_effect = redis.RedisError("Connection failed")
        
        with pytest.raises(redis.RedisError):
            is_duplicate("8f373c9c3c9c3c1e")
    
    def test_pipeline_operations(self):
        """Test pipeline operations for mark_processed."""
        # Mock pipeline operations
        mock_pipe = MagicMock()
        mock_pipe.sadd.return_value = mock_pipe
        mock_pipe.expire.return_value = mock_pipe
        mock_pipe.execute.return_value = [1, True]
        self.mock_redis.pipeline.return_value = mock_pipe
        
        result = mark_processed("8f373c9c3c9c3c1e")
        
        assert result is True
        mock_pipe.sadd.assert_called_once_with("dedup:phash:8f37", "8f373c9c3c9c3c1e")
        mock_pipe.expire.assert_called_once_with("dedup:phash:8f37", settings.dedup_ttl_seconds)
        mock_pipe.execute.assert_called_once()


class TestNearDuplicateDedup:
    """Test cases for near-duplicate deduplication with Hamming distance."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mock_redis = MagicMock()
        self.mock_redis.smembers.return_value = set()
        self.mock_redis.scard.return_value = 0
        self.mock_redis.sadd.return_value = 1
        self.mock_redis.expire.return_value = True
        self.mock_redis.pipeline.return_value.execute.return_value = [1, True]
        
        self.redis_patcher = patch('pipeline.dedup.get_redis_client', return_value=self.mock_redis)
        self.redis_patcher.start()
    
    def teardown_method(self):
        """Clean up test environment."""
        self.redis_patcher.stop()
    
    def test_should_skip_no_matches(self):
        """Test should_skip with no similar hashes."""
        from pipeline.dedup import should_skip
        
        result = should_skip("tenant1", "8f37", "8f373c9c3c9c3c1e", max_dist=3)
        
        assert result is False
        self.mock_redis.smembers.assert_called_once_with("dedup:near:tenant1:8f37")
    
    def test_should_skip_exact_match(self):
        """Test should_skip with exact matching hash (distance=0)."""
        from pipeline.dedup import should_skip
        
        self.mock_redis.smembers.return_value = {b"8f373c9c3c9c3c1e"}
        
        result = should_skip("tenant1", "8f37", "8f373c9c3c9c3c1e", max_dist=3)
        
        assert result is True
    
    def test_should_skip_within_threshold(self):
        """Test should_skip with hash within Hamming distance threshold."""
        from pipeline.dedup import should_skip
        
        # These two hashes have Hamming distance of 2
        self.mock_redis.smembers.return_value = {b"8f373c9c3c9c3c1e"}
        
        result = should_skip("tenant1", "8f37", "8f373c9c3c9c3c1c", max_dist=3)
        
        assert result is True
    
    def test_should_skip_beyond_threshold(self):
        """Test should_skip with hash beyond Hamming distance threshold."""
        from pipeline.dedup import should_skip
        
        # Hash with larger Hamming distance
        self.mock_redis.smembers.return_value = {b"8f373c9c3c9c3c1e"}
        
        result = should_skip("tenant1", "8f37", "0000000000000000", max_dist=3)
        
        assert result is False
    
    def test_should_skip_disabled(self):
        """Test that should_skip returns False when feature is disabled."""
        from pipeline.dedup import should_skip
        
        with patch.object(settings, 'enable_global_dedup', False):
            result = should_skip("tenant1", "8f37", "8f373c9c3c9c3c1e", max_dist=3)
            assert result is False
            self.mock_redis.smembers.assert_not_called()
    
    def test_remember_success(self):
        """Test remembering a pHash for near-duplicate detection."""
        from pipeline.dedup import remember
        
        result = remember("tenant1", "8f37", "8f373c9c3c9c3c1e", max_size=1000, ttl=3600)
        
        assert result is True
        self.mock_redis.scard.assert_called_once_with("dedup:near:tenant1:8f37")
        pipe = self.mock_redis.pipeline.return_value
        pipe.sadd.assert_called_once_with("dedup:near:tenant1:8f37", "8f373c9c3c9c3c1e")
        pipe.expire.assert_called_once_with("dedup:near:tenant1:8f37", 3600)
    
    def test_remember_size_limiting(self):
        """Test that remember removes old entries when size limit is reached."""
        from pipeline.dedup import remember
        
        self.mock_redis.scard.return_value = 1000
        
        result = remember("tenant1", "8f37", "8f373c9c3c9c3c1e", max_size=1000, ttl=3600)
        
        # Should have called spop to remove an old entry
        self.mock_redis.spop.assert_called_once_with("dedup:near:tenant1:8f37")
        assert result is True
    
    def test_remember_disabled(self):
        """Test that remember returns True when feature is disabled."""
        from pipeline.dedup import remember
        
        with patch.object(settings, 'enable_global_dedup', False):
            result = remember("tenant1", "8f37", "8f373c9c3c9c3c1e")
            assert result is True
            self.mock_redis.pipeline.assert_not_called()
    
    def test_clear_near_dedup_cache_all(self):
        """Test clearing all near-dedup caches."""
        from pipeline.dedup import clear_near_dedup_cache
        
        self.mock_redis.keys.return_value = [
            b"dedup:near:tenant1:8f37",
            b"dedup:near:tenant2:9f38"
        ]
        self.mock_redis.delete.return_value = 2
        
        result = clear_near_dedup_cache()
        
        assert result == 2
        self.mock_redis.keys.assert_called_once_with("dedup:near:*")
        self.mock_redis.delete.assert_called_once()
    
    def test_clear_near_dedup_cache_tenant(self):
        """Test clearing near-dedup cache for specific tenant."""
        from pipeline.dedup import clear_near_dedup_cache
        
        self.mock_redis.keys.return_value = [b"dedup:near:tenant1:8f37"]
        self.mock_redis.delete.return_value = 1
        
        result = clear_near_dedup_cache(tenant_id="tenant1")
        
        assert result == 1
        self.mock_redis.keys.assert_called_once_with("dedup:near:tenant1:*")
    
    def test_hamming_distance_calculation(self):
        """Test Hamming distance calculation via XOR bitcount."""
        from pipeline.utils import hamming_distance_hex
        
        # Test identical hashes (distance = 0)
        assert hamming_distance_hex("8f37", "8f37") == 0
        
        # Test single bit difference (distance = 1)
        assert hamming_distance_hex("8f37", "8f36") == 1
        
        # Test known distances
        dist = hamming_distance_hex("ffff", "0000")
        assert dist == 16  # All 16 bits different
