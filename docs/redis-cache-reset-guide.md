# Redis Cache Reset Guide

This guide explains how to reset Redis cache during test runs and reruns in the Face Scanning Protection MVP application.

## Overview

The application uses a **Hybrid Cache Service** that combines:
- **Redis** (primary cache) - Fast, volatile storage for hot data
- **PostgreSQL** (secondary cache) - Persistent storage for reliability

## Quick Reference

### During Test Development
```bash
# Reset all cache data (if dependencies are installed)
python backend/scripts/reset_redis_cache.py --all

# Reset for specific tenant
python backend/scripts/reset_redis_cache.py --tenant test-tenant-123

# Use test database (DB 15)
python backend/scripts/reset_redis_cache.py --db 15 --all

# Show what would be cleared (dry run)
python backend/scripts/reset_redis_cache.py --all --dry-run

# If Python dependencies are missing, use Docker method:
docker compose exec redis redis-cli FLUSHDB

# Or try all methods automatically:
make reset-redis-all-methods
```

### Via API Endpoints
```bash
# Clear all cache
curl -X DELETE http://localhost:8000/cache/all

# Clear tenant cache
curl -X DELETE http://localhost:8000/cache/tenant/test-tenant-123
```

### Via Makefile
```bash
# Reset PostgreSQL cache only
make reset-cache

# Reset both cache and MinIO
make reset-both
```

## Detailed Methods

### 1. Standalone Reset Script

The most flexible method is using the standalone script:

```bash
# Basic usage
python backend/scripts/reset_redis_cache.py --all

# Advanced options
python backend/scripts/reset_redis_cache.py \
  --redis-url redis://localhost:6379/15 \
  --pattern "embedding:*" \
  --stats

# Get Redis information
python backend/scripts/reset_redis_cache.py --info

# List current cache keys
python backend/scripts/reset_redis_cache.py --list-keys
```

**Script Options:**
- `--all`: Clear all cache data
- `--tenant <id>`: Clear cache for specific tenant
- `--pattern <pattern>`: Clear keys matching pattern (e.g., "embedding:*")
- `--db <number>`: Use specific Redis database
- `--info`: Show Redis server information
- `--list-keys`: List current cache keys
- `--stats`: Reset cache statistics
- `--dry-run`: Show what would be done without executing

### 2. Programmatic Reset in Tests

#### Using Test Fixtures

```python
# In your test file
import pytest
from backend.tests.cache_test_utils import redis_test_manager, isolated_cache_service

# For integration tests
async def test_cache_operations(redis_test_manager):
    cache_service = redis_test_manager.cache_service
    
    # Cache is automatically cleared before/after test
    await cache_service.cache_face_embeddings(b"test", "tenant1", [])
    result = await cache_service.get_cached_face_embeddings(b"test", "tenant1")
    assert result is not None

# For isolated tests
async def test_with_isolation(isolated_cache_service):
    # Gets completely isolated cache service
    await isolated_cache_service.cache_face_embeddings(b"test", "tenant1", [])
```

#### Manual Reset in Tests

```python
from backend.tests.cache_test_utils import reset_cache_before_test, reset_cache_after_test
from app.services.cache import HybridCacheService

async def test_with_manual_reset():
    await reset_cache_before_test()
    
    # Run your test
    cache_service = HybridCacheService()
    await cache_service.cache_face_embeddings(b"test", "tenant1", [])
    
    await reset_cache_after_test()
```

#### Using Cache Reset Strategies

```python
from backend.tests.cache_test_utils import CacheResetStrategy

# For different test types
await CacheResetStrategy.reset_for_unit_tests()      # Mocked services
await CacheResetStrategy.reset_for_integration_tests()  # Real services, test DB
await CacheResetStrategy.reset_for_e2e_tests()       # All databases
```

### 3. API Endpoints

The application provides REST endpoints for cache management:

```python
# Clear all cache
DELETE /cache/all
Response: {"message": "All cache data cleared", "success": true}

# Clear tenant cache  
DELETE /cache/tenant/{tenant_id}
Response: {"message": "Cleared 10 cache entries for tenant test-tenant-123", "deleted_count": 10}
```

### 4. Direct Redis Commands

For manual debugging and testing:

```bash
# Connect to Redis CLI
docker compose exec redis redis-cli

# Clear all data in current database
FLUSHDB

# Clear all data in all databases
FLUSHALL

# Clear specific database
SELECT 15
FLUSHDB

# List all keys
KEYS *

# Delete specific keys
DEL key1 key2 key3

# Delete keys by pattern
EVAL "return redis.call('del', unpack(redis.call('keys', ARGV[1])))" 0 "embedding:*"
```

## Test Environment Setup

### Different Redis Databases

The application uses multiple Redis databases:

- **DB 0**: Main application cache
- **DB 1**: Celery broker
- **DB 2**: Celery results
- **DB 14**: Isolated test cache
- **DB 15**: Integration test cache

### Test Configuration

```python
# In conftest.py or test files
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use DB 15 for tests

# For complete isolation
ISOLATED_REDIS_URL = "redis://localhost:6379/14"  # Use DB 14 for isolation
```

### Environment Variables

```bash
# For testing
export REDIS_URL="redis://localhost:6379/15"
export ENVIRONMENT="testing"

# For development
export REDIS_URL="redis://redis:6379/0"
export ENVIRONMENT="development"
```

## Cache Reset Strategies

### 1. Before Each Test

```python
@pytest.fixture(autouse=True)
async def reset_cache_before_test():
    """Automatically reset cache before each test."""
    await reset_cache_before_test()
    yield
    await reset_cache_after_test()
```

### 2. Before Test Class

```python
@pytest.fixture(scope="class")
async def reset_cache_for_class():
    """Reset cache once per test class."""
    await reset_cache_before_test()
    yield
    await reset_cache_after_test()
```

### 3. Before Test Session

```python
@pytest.fixture(scope="session")
async def reset_cache_for_session():
    """Reset cache once per test session."""
    await reset_cache_before_test()
    yield
    await reset_cache_after_test()
```

### 4. Selective Reset

```python
async def reset_specific_cache_types():
    """Reset only specific cache types."""
    cache_service = HybridCacheService()
    
    # Clear only face embeddings
    await cache_service.clear_cache_pattern("embedding:*")
    
    # Clear only crawl cache
    await cache_service.clear_cache_pattern("crawl:*")
    
    # Clear only for specific tenant
    await cache_service.invalidate_tenant_cache("test-tenant")
```

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError: No module named 'redis'**
   ```bash
   # Install Redis Python client
   pip install redis
   
   # Or install all backend dependencies
   cd backend && pip install -r requirements.txt
   
   # Alternative: Use Docker method instead
   docker compose exec redis redis-cli FLUSHDB
   
   # Or use the multi-method script
   make reset-redis-all-methods
   ```

2. **Redis Connection Failed**
   ```bash
   # Check if Redis is running
   docker compose ps redis
   
   # Restart Redis
   docker compose restart redis
   ```

3. **Cache Not Clearing**
   ```bash
   # Check Redis databases
   docker compose exec redis redis-cli INFO keyspace
   
   # Clear specific database
   docker compose exec redis redis-cli -n 15 FLUSHDB
   ```

4. **Tests Failing Due to Cache**
   ```python
   # Use isolated cache service
   async def test_with_isolation(isolated_cache_service):
       # Test code here
   ```

### Debug Commands

```bash
# Check Redis status
python backend/scripts/reset_redis_cache.py --info

# List all cache keys
python backend/scripts/reset_redis_cache.py --list-keys

# Check specific pattern
docker compose exec redis redis-cli KEYS "embedding:*"

# Monitor Redis commands
docker compose exec redis redis-cli MONITOR
```

## Best Practices

### 1. Test Isolation

- Use separate Redis databases for different test types
- Clear cache before and after each test
- Use isolated cache services for unit tests

### 2. Performance

- Use `--dry-run` to preview operations
- Clear specific patterns instead of all cache when possible
- Reset cache statistics to avoid memory leaks

### 3. Development Workflow

```bash
# Start development with clean cache
make reset-both

# Run tests with isolated cache
python -m pytest backend/tests/ --redis-url redis://localhost:6379/15

# Debug cache issues
python backend/scripts/reset_redis_cache.py --info --list-keys
```

### 4. CI/CD Integration

```yaml
# In your CI pipeline
- name: Reset Redis Cache
  run: |
    python backend/scripts/reset_redis_cache.py --db 15 --all
    
- name: Run Tests
  run: |
    python -m pytest backend/tests/
```

## Examples

### Complete Test Setup

```python
import pytest
from backend.tests.cache_test_utils import redis_test_manager, CacheResetStrategy

class TestCacheOperations:
    
    @pytest.fixture(autouse=True)
    async def setup_test_cache(self, redis_test_manager):
        """Setup cache for each test."""
        self.cache_service = redis_test_manager.cache_service
        await CacheResetStrategy.reset_for_integration_tests()
    
    async def test_face_embedding_cache(self):
        """Test face embedding caching."""
        content = b"test_image_data"
        tenant_id = "test-tenant-123"
        embeddings = [{"embedding": [0.1] * 512, "bbox": [0, 0, 100, 100]}]
        
        # Cache embeddings
        await self.cache_service.cache_face_embeddings(content, tenant_id, embeddings)
        
        # Retrieve embeddings
        result = await self.cache_service.get_cached_face_embeddings(content, tenant_id)
        
        assert result is not None
        assert len(result) == 1
        assert result[0]["bbox"] == [0, 0, 100, 100]
    
    async def test_tenant_cache_isolation(self):
        """Test that tenant caches are isolated."""
        content = b"test_image_data"
        tenant1 = "tenant-1"
        tenant2 = "tenant-2"
        
        # Cache for tenant 1
        await self.cache_service.cache_face_embeddings(content, tenant1, [{"test": "data1"}])
        
        # Should not find data for tenant 2
        result = await self.cache_service.get_cached_face_embeddings(content, tenant2)
        assert result is None
        
        # Should find data for tenant 1
        result = await self.cache_service.get_cached_face_embeddings(content, tenant1)
        assert result is not None
```

This guide provides comprehensive methods for resetting Redis cache during test runs and reruns, ensuring clean test environments and reliable test results.
