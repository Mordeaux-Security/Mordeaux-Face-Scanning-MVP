"""
Redis Cache Test Utilities

This module provides utilities for managing Redis cache during testing,
including setup, teardown, and reset operations.
"""

import asyncio
import os
import redis
import pytest
from typing import Optional, Dict, Any
from app.services.cache import HybridCacheService


class RedisTestManager:
    """Manages Redis cache for testing scenarios."""
    
    def __init__(self, redis_url: str = "redis://localhost:6379/15"):
        """
        Initialize Redis test manager.
        
        Args:
            redis_url: Redis URL for test database (default uses DB 15)
        """
        self.redis_url = redis_url
        self.redis_client = redis.from_url(redis_url, decode_responses=True)
        self.cache_service = None
    
    async def setup_test_cache(self) -> HybridCacheService:
        """Set up cache service for testing."""
        self.cache_service = HybridCacheService(redis_url=self.redis_url)
        await self.clear_all_cache()
        return self.cache_service
    
    async def clear_all_cache(self) -> bool:
        """Clear all Redis cache data."""
        try:
            # Clear Redis
            self.redis_client.flushdb()
            
            # Clear cache service if available
            if self.cache_service:
                return await self.cache_service.clear_all_cache()
            
            return True
        except Exception as e:
            print(f"Failed to clear cache: {e}")
            return False
    
    async def clear_tenant_cache(self, tenant_id: str) -> int:
        """Clear cache for specific tenant."""
        if self.cache_service:
            return await self.cache_service.invalidate_tenant_cache(tenant_id)
        return 0
    
    def get_redis_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        try:
            info = self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "db_size": self.redis_client.dbsize()
            }
        except Exception as e:
            return {"error": str(e)}
    
    def get_cache_keys(self, pattern: str = "*") -> list:
        """Get all cache keys matching pattern."""
        try:
            return self.redis_client.keys(pattern)
        except Exception as e:
            print(f"Failed to get cache keys: {e}")
            return []
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()


# Test fixtures for different cache reset scenarios

@pytest.fixture
async def redis_test_manager():
    """Redis test manager for integration tests."""
    manager = RedisTestManager()
    await manager.setup_test_cache()
    
    yield manager
    
    # Cleanup
    await manager.clear_all_cache()
    manager.close()


@pytest.fixture
async def isolated_cache_service():
    """Isolated cache service for unit tests."""
    # Use a different Redis DB for isolation
    test_redis_url = "redis://localhost:6379/14"  # Use DB 14 for isolation
    
    cache_service = HybridCacheService(redis_url=test_redis_url)
    
    # Clear before test
    await cache_service.clear_all_cache()
    
    yield cache_service
    
    # Clear after test
    await cache_service.clear_all_cache()


@pytest.fixture
def redis_reset_script():
    """Generate Redis reset script for manual testing."""
    return """
# Redis Cache Reset Script for Testing

# 1. Clear all Redis data
redis-cli FLUSHALL

# 2. Clear specific database (if using multiple DBs)
redis-cli -n 0 FLUSHDB  # Main app DB
redis-cli -n 1 FLUSHDB  # Celery broker DB  
redis-cli -n 2 FLUSHDB  # Celery result DB
redis-cli -n 15 FLUSHDB # Test DB

# 3. Clear via API endpoints
curl -X DELETE http://localhost:8000/cache/all
curl -X DELETE http://localhost:8000/cache/tenant/test-tenant-123

# 4. Clear PostgreSQL cache tables
docker compose exec postgres psql -U mordeaux -d mordeaux -c "
    DELETE FROM face_embeddings_cache;
    DELETE FROM perceptual_hash_cache;
    DELETE FROM crawl_cache;
"

# 5. Reset cache statistics
python -c "
from app.services.cache import get_hybrid_cache_service
cache = get_hybrid_cache_service()
cache.reset_cache_stats()
print('Cache statistics reset')
"
"""


# Utility functions for test setup/teardown

async def reset_cache_before_test():
    """Reset cache before running a test."""
    cache_service = HybridCacheService()
    await cache_service.clear_all_cache()
    cache_service.reset_cache_stats()


async def reset_cache_after_test():
    """Reset cache after running a test."""
    cache_service = HybridCacheService()
    await cache_service.clear_all_cache()


def create_test_redis_url(db_number: int = 15) -> str:
    """
    Create Redis URL for test database.
    
    Args:
        db_number: Redis database number (0-15)
        
    Returns:
        Redis URL string
    """
    return f"redis://localhost:6379/{db_number}"


# Environment-specific cache reset strategies

class CacheResetStrategy:
    """Different strategies for cache reset based on test environment."""
    
    @staticmethod
    async def reset_for_unit_tests():
        """Reset strategy for unit tests (mocked services)."""
        # Unit tests use mocks, so no real reset needed
        pass
    
    @staticmethod
    async def reset_for_integration_tests():
        """Reset strategy for integration tests."""
        cache_service = HybridCacheService(redis_url=create_test_redis_url(15))
        await cache_service.clear_all_cache()
        cache_service.reset_cache_stats()
    
    @staticmethod
    async def reset_for_e2e_tests():
        """Reset strategy for end-to-end tests."""
        # Reset all databases
        for db_num in [0, 1, 2, 15]:
            cache_service = HybridCacheService(redis_url=create_test_redis_url(db_num))
            await cache_service.clear_all_cache()
    
    @staticmethod
    def reset_via_docker():
        """Reset cache using Docker commands."""
        return """
        # Reset Redis containers
        docker compose restart redis
        
        # Clear PostgreSQL cache
        docker compose exec postgres psql -U mordeaux -d mordeaux -c "
            TRUNCATE TABLE face_embeddings_cache;
            TRUNCATE TABLE perceptual_hash_cache;
            TRUNCATE TABLE crawl_cache;
        "
        
        # Reset via Makefile
        make reset-cache
        """


# Test markers and utilities

def mark_cache_reset_required(test_func):
    """Decorator to mark tests that require cache reset."""
    return pytest.mark.cache_reset(test_func)


def mark_cache_isolation_required(test_func):
    """Decorator to mark tests that require cache isolation."""
    return pytest.mark.cache_isolation(test_func)


# Usage examples in test files:

"""
# Example 1: Using fixtures in tests
async def test_cache_operations(redis_test_manager):
    cache_service = redis_test_manager.cache_service
    
    # Test cache operations
    await cache_service.cache_face_embeddings(b"test", "tenant1", [])
    
    # Verify cache
    result = await cache_service.get_cached_face_embeddings(b"test", "tenant1")
    assert result is not None

# Example 2: Manual cache reset in test
async def test_with_manual_reset():
    await reset_cache_before_test()
    
    # Run test
    cache_service = HybridCacheService()
    await cache_service.cache_face_embeddings(b"test", "tenant1", [])
    
    await reset_cache_after_test()

# Example 3: Using isolated cache
async def test_with_isolation(isolated_cache_service):
    # This test gets a completely isolated cache service
    await isolated_cache_service.cache_face_embeddings(b"test", "tenant1", [])
    
    # Cache is automatically cleared after test
"""
