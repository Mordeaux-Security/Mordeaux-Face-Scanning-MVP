# Hybrid Cache Service V2

## Overview

The Hybrid Cache Service V2 combines the best of both worlds:
- **Redis** (fast, volatile) for ultra-fast primary lookups
- **PostgreSQL** (persistent, reliable) for reliable secondary storage

This architecture provides:
- ⚡ **Speed**: Sub-millisecond Redis lookups for hot data
- 💾 **Persistence**: PostgreSQL storage survives restarts and cache clears
- 🔄 **Automatic Backfill**: PostgreSQL data → Redis for future fast access
- 🛡️ **Graceful Degradation**: Works even if Redis is unavailable
- 💰 **Cost Efficiency**: PostgreSQL handles bulk storage, Redis handles hot cache

## Architecture

```
┌─────────────────┐
│   Application   │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │   Redis   │ ← Primary Cache (fast, volatile)
    │  (RAM)    │
    └─────┬─────┘
          │ (miss)
    ┌─────▼─────┐
    │PostgreSQL │ ← Secondary Cache (persistent, reliable)
    │  (SSD)    │
    └───────────┘
```

## Use Cases

### 1. Face Embeddings Cache
- **Purpose**: Cache computed face embeddings to avoid re-computation
- **Redis TTL**: 1 hour (frequently accessed data)
- **PostgreSQL**: Permanent storage for reliability

### 2. Perceptual Hash Cache
- **Purpose**: Cache image hashes for duplicate detection
- **Redis TTL**: 2 hours (moderately accessed data)
- **PostgreSQL**: Permanent storage for cross-session deduplication

### 3. Crawl Cache (Duplicate Prevention)
- **Purpose**: Prevent reprocessing same images during web crawling
- **Redis TTL**: 24 hours (long-term but not permanent)
- **PostgreSQL**: Permanent storage for reliable duplicate prevention

## Quick Start

### 1. Setup Dependencies

```bash
# Install required packages
pip install redis psycopg[binary] asyncio

# Start Redis (if not already running)
docker run -d -p 6379:6379 redis:7-alpine

# Start PostgreSQL (if not already running)
docker run -d -p 5432:5432 -e POSTGRES_DB=mordeaux -e POSTGRES_USER=mordeaux -e POSTGRES_PASSWORD=password postgres:16-alpine
```

### 2. Run Database Migration

```bash
# Apply the hybrid cache migration
psql -h localhost -U mordeaux -d mordeaux -f migrations/004_hybrid_cache_tables.sql
```

### 3. Basic Usage

```python
from app.services.cache import get_hybrid_cache_service

# Get cache service
cache = get_hybrid_cache_service()

# Cache face embeddings
image_bytes = b"your_image_data"
embeddings = [{"bbox": [100, 150, 200, 250], "embedding": [...], "det_score": 0.95}]
await cache.cache_face_embeddings(image_bytes, "tenant_123", embeddings)

# Retrieve cached embeddings (checks Redis first, then PostgreSQL)
cached_embeddings = await cache.get_cached_face_embeddings(image_bytes, "tenant_123")

# Crawl cache for duplicate prevention
url = "https://example.com/image.jpg"
should_skip, raw_key = await cache.should_skip_crawled_image(url, image_bytes, "tenant_123")

if not should_skip:
    # Process the image
    raw_key = "processed_image_key"
    await cache.store_crawled_image(url, image_bytes, raw_key, "thumb_key", "tenant_123")
```

## Configuration

### Environment Variables

```bash
# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=mordeaux
POSTGRES_USER=mordeaux
POSTGRES_PASSWORD=password
```

### Cache TTL Settings

The service uses different TTL values for different cache types:

```python
redis_ttl_settings = {
    'embedding_cache': 3600,      # 1 hour
    'search_cache': 300,          # 5 minutes  
    'phash_cache': 7200,          # 2 hours
    'face_detection_cache': 3600, # 1 hour
    'crawl_cache': 86400,         # 24 hours
}
```

## API Reference

### Face Embeddings Cache

```python
# Cache face embeddings
await cache.cache_face_embeddings(content: bytes, tenant_id: str, embeddings: List[Dict]) -> bool

# Get cached face embeddings
await cache.get_cached_face_embeddings(content: bytes, tenant_id: str) -> Optional[List[Dict]]
```

### Perceptual Hash Cache

```python
# Cache perceptual hash
await cache.cache_perceptual_hash(content: bytes, tenant_id: str, phash: str) -> bool

# Get cached perceptual hash
await cache.get_cached_perceptual_hash(content: bytes, tenant_id: str) -> Optional[str]
```

### Crawl Cache (Duplicate Prevention)

```python
# Check if image should be skipped (duplicate prevention)
await cache.should_skip_crawled_image(url: str, image_bytes: bytes, tenant_id: str) -> Tuple[bool, Optional[str]]

# Store crawled image
await cache.store_crawled_image(url: str, image_bytes: bytes, raw_key: str, thumbnail_key: Optional[str], tenant_id: str) -> bool
```

### Cache Management

```python
# Invalidate all cache for a tenant
await cache.invalidate_tenant_cache(tenant_id: str) -> int

# Get cache statistics
await cache.get_cache_stats() -> Dict[str, Any]

# Clear all cache data
await cache.clear_all_cache() -> bool
```

## Performance Characteristics

### Redis (Primary Cache)
- **Read Latency**: ~0.1ms
- **Write Latency**: ~0.1ms
- **Throughput**: Millions of operations/second
- **Memory Usage**: All data in RAM
- **Durability**: Configurable (can be lost)

### PostgreSQL (Secondary Cache)
- **Read Latency**: ~1-10ms
- **Write Latency**: ~1-10ms
- **Throughput**: Thousands of operations/second
- **Storage**: Disk-based with memory caching
- **Durability**: ACID compliant, crash recovery

### Hybrid Performance
- **Cache Hit (Redis)**: ~0.1ms response time
- **Cache Hit (PostgreSQL)**: ~1-10ms response time + Redis backfill
- **Cache Miss**: No additional overhead

## Monitoring and Maintenance

### Cache Statistics

```python
stats = await cache.get_cache_stats()

# Redis stats
print(f"Redis memory usage: {stats['redis_stats']['used_memory_human']}")
print(f"Redis hit rate: {stats['redis_stats']['hit_rate']}%")

# PostgreSQL stats
print(f"Face embeddings cache: {stats['postgres_stats']['face_embeddings_cache_count']} entries")
print(f"Perceptual hash cache: {stats['postgres_stats']['perceptual_hash_cache_count']} entries")
print(f"Crawl cache: {stats['postgres_stats']['crawl_cache_count']} entries")
```

### Database Maintenance

```sql
-- View cache statistics
SELECT * FROM cache_stats;

-- Clean up old cache entries (keep 7 days)
SELECT * FROM cleanup_old_cache_entries(7);

-- Monitor index usage
SELECT * FROM get_cache_index_usage();
```

## Migration from Existing Cache Services

### From Redis-only Cache

```python
# Old code
from app.services.cache import get_cache_service
cache = get_cache_service()

# New code (compatible interface)
from app.services.cache import get_hybrid_cache_service
cache = get_hybrid_cache_service()  # Drop-in replacement
```

### From PostgreSQL-only Cache

```python
# Old code
from app.services.cache import get_cache_service
cache = get_cache_service()

# New code (enhanced with Redis)
from app.services.cache import get_hybrid_cache_service
cache = get_hybrid_cache_service()  # Same interface, better performance
```

## Troubleshooting

### Redis Connection Issues

```python
# The service gracefully degrades to PostgreSQL-only if Redis is unavailable
# Check logs for: "Redis cache unavailable: ... Falling back to PostgreSQL only."
```

### PostgreSQL Connection Issues

```python
# The service gracefully degrades to Redis-only if PostgreSQL is unavailable
# Check logs for: "PostgreSQL cache unavailable: ... Using Redis only."
```

### Performance Issues

1. **High Redis Memory Usage**: Adjust TTL settings or implement cache eviction policies
2. **Slow PostgreSQL Queries**: Check indexes and run `ANALYZE` on tables
3. **Cache Misses**: Monitor hit rates and adjust TTL values

### Data Consistency

- Redis and PostgreSQL are kept in sync automatically
- If Redis data is lost, it's automatically backfilled from PostgreSQL
- PostgreSQL is the source of truth for persistent data

## Examples

See `backend/examples/hybrid_cache_example.py` for comprehensive usage examples including:
- Face embeddings caching
- Perceptual hash caching
- Crawl duplicate prevention
- Cache statistics
- Tenant management

Run the examples:
```bash
python backend/examples/hybrid_cache_example.py
```

## Best Practices

1. **Use Appropriate TTL Values**: Balance memory usage with cache effectiveness
2. **Monitor Hit Rates**: Optimize TTL settings based on access patterns
3. **Regular Cleanup**: Use the cleanup functions to remove old entries
4. **Tenant Isolation**: Always specify tenant_id for multi-tenant applications
5. **Error Handling**: The service gracefully handles Redis/PostgreSQL failures
6. **Performance Monitoring**: Track both Redis and PostgreSQL performance metrics
