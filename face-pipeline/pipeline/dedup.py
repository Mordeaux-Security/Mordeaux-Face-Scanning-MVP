"""
Global Deduplication Service

Redis-based global deduplication using pHash prefix sets to prevent
duplicate face processing across the entire system.

Architecture:
- Uses Redis SET for pHash prefix storage: `dedup:phash:{prefix}` → set of full pHashes
- Checks duplicates before processing: `SISMEMBER dedup:phash:{prefix} {full_phash}`
- Adds new pHashes after indexing: `SADD dedup:phash:{prefix} {full_phash}`
- Optional TTL for cache management: `EXPIRE dedup:phash:{prefix} 86400`

Key Features:
- Thread-safe atomic operations
- Configurable TTL for cache management
- Tenant-scoped clearing for testing
- Minimal memory footprint with prefix-based sharding
"""

import logging
import redis
from typing import Optional
from config.settings import settings

logger = logging.getLogger(__name__)

# Redis client singleton
_redis_client: Optional[redis.Redis] = None


def get_redis_client() -> redis.Redis:
    """Get Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url)
    return _redis_client


def is_duplicate(phash: str) -> bool:
    """
    Check if a pHash already exists globally.
    
    Args:
        phash: Full perceptual hash (16-char hex string)
        
    Returns:
        True if pHash exists, False otherwise
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    if not settings.enable_global_dedup:
        return False
        
    if not phash or len(phash) < 4:
        logger.warning(f"Invalid pHash format: {phash}")
        return False
    
    # Extract 4-char prefix for sharding
    prefix = phash[:4]
    key = f"dedup:phash:{prefix}"
    
    try:
        r = get_redis_client()
        exists = r.sismember(key, phash)
        logger.debug(f"Duplicate check for {phash}: {exists}")
        return bool(exists)
    except redis.RedisError as e:
        logger.error(f"Redis error checking duplicate {phash}: {e}")
        raise


def mark_processed(phash: str) -> bool:
    """
    Mark a pHash as processed (add to global set).
    
    Args:
        phash: Full perceptual hash (16-char hex string)
        
    Returns:
        True if successfully added, False if already exists
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    if not settings.enable_global_dedup:
        return True
        
    if not phash or len(phash) < 4:
        logger.warning(f"Invalid pHash format: {phash}")
        return False
    
    # Extract 4-char prefix for sharding
    prefix = phash[:4]
    key = f"dedup:phash:{prefix}"
    
    try:
        r = get_redis_client()
        
        # Add to set and set TTL if not already set
        pipe = r.pipeline()
        pipe.sadd(key, phash)
        pipe.expire(key, settings.dedup_ttl_seconds)
        results = pipe.execute()
        
        # Check if this was a new addition (not duplicate)
        was_new = results[0] > 0
        logger.debug(f"Marked processed {phash}: new={was_new}")
        return was_new
        
    except redis.RedisError as e:
        logger.error(f"Redis error marking processed {phash}: {e}")
        raise


def should_skip(tenant_id: str, prefix: str, phash: str, max_dist: int | None = None) -> bool:
    """
    Check if a similar pHash exists within Hamming distance threshold.
    
    This provides near-duplicate detection by comparing against stored
    pHashes with the same prefix and checking if any are within max_dist.
    
    Args:
        tenant_id: Tenant identifier for scoping
        prefix: pHash prefix (4 chars) for efficient lookup
        phash: Full perceptual hash (16-char hex string)
        max_dist: Maximum Hamming distance to consider a duplicate (uses settings default if None)
        
    Returns:
        True if a similar pHash exists (should skip), False otherwise
        
    Note:
        Requires ENABLE_GLOBAL_DEDUP=true to be active.
    """
    if not settings.enable_global_dedup:
        return False
        
    if not phash or len(phash) < 4 or not prefix:
        logger.warning(f"Invalid pHash/prefix: {phash}/{prefix}")
        return False
    
    # Use settings default if not provided
    if max_dist is None:
        max_dist = settings.dedup_max_hamming
    
    # Import here to avoid circular dependency
    from pipeline.utils import hamming_distance_hex
    
    key = f"dedup:near:{tenant_id}:{prefix}"
    
    try:
        r = get_redis_client()
        stored_hashes = r.smembers(key)
        
        # Check Hamming distance against all stored hashes with this prefix
        for stored in stored_hashes:
            stored_str = stored.decode('utf-8') if isinstance(stored, bytes) else stored
            dist = hamming_distance_hex(phash, stored_str)
            if dist <= max_dist:
                logger.debug(f"Near-duplicate found: {phash} ~ {stored_str} (distance={dist})")
                return True
        
        return False
        
    except redis.RedisError as e:
        logger.error(f"Redis error in should_skip for {phash}: {e}")
        raise


def remember(tenant_id: str, prefix: str, phash: str, max_size: int = 1000, ttl: int | None = None) -> bool:
    """
    Store a pHash for future near-duplicate detection.
    
    Adds the pHash to the tenant-scoped prefix set. Implements size limiting
    by removing oldest entries when max_size is reached.
    
    Args:
        tenant_id: Tenant identifier for scoping
        prefix: pHash prefix (4 chars) for sharding
        phash: Full perceptual hash (16-char hex string)
        max_size: Maximum number of hashes per prefix set (default 1000)
        ttl: Time-to-live in seconds (uses settings default if None)
        
    Returns:
        True if successfully added, False otherwise
        
    Note:
        Requires ENABLE_GLOBAL_DEDUP=true to be active.
    """
    if not settings.enable_global_dedup:
        return True
        
    if not phash or len(phash) < 4 or not prefix:
        logger.warning(f"Invalid pHash/prefix: {phash}/{prefix}")
        return False
    
    # Use settings default if not provided
    if ttl is None:
        ttl = settings.dedup_ttl_seconds
    
    key = f"dedup:near:{tenant_id}:{prefix}"
    
    try:
        r = get_redis_client()
        
        # Check current size and trim if needed (simple FIFO approximation)
        current_size = r.scard(key)
        if current_size >= max_size:
            # Remove a random member to make space (Redis sets don't have order)
            r.spop(key)
        
        # Add new hash and set TTL
        pipe = r.pipeline()
        pipe.sadd(key, phash)
        pipe.expire(key, ttl)
        results = pipe.execute()
        
        was_new = results[0] > 0
        logger.debug(f"Remembered {phash} in {key}: new={was_new}")
        return was_new
        
    except redis.RedisError as e:
        logger.error(f"Redis error in remember for {phash}: {e}")
        raise


def clear_near_dedup_cache(tenant_id: Optional[str] = None) -> int:
    """
    Clear near-duplicate deduplication cache.
    
    Args:
        tenant_id: Optional tenant ID to clear only that tenant's data.
                  If None, clears all near-dedup caches.
                  
    Returns:
        Number of keys deleted
    """
    try:
        r = get_redis_client()
        
        if tenant_id:
            pattern = f"dedup:near:{tenant_id}:*"
        else:
            pattern = "dedup:near:*"
        
        keys = r.keys(pattern)
        
        if not keys:
            logger.info(f"No near-dedup cache keys found for pattern {pattern}")
            return 0
        
        deleted = r.delete(*keys)
        logger.info(f"Cleared {deleted} near-dedup cache keys")
        return deleted
        
    except redis.RedisError as e:
        logger.error(f"Redis error clearing near-dedup cache: {e}")
        raise


def clear_dedup_cache(tenant_id: Optional[str] = None) -> int:
    """
    Clear deduplication cache for testing or maintenance.
    
    Args:
        tenant_id: Optional tenant ID to clear only tenant-specific data
                  (currently not implemented - clears all)
                  
    Returns:
        Number of keys deleted
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        r = get_redis_client()
        
        # Find all dedup keys
        pattern = "dedup:phash:*"
        keys = r.keys(pattern)
        
        if not keys:
            logger.info("No dedup cache keys found")
            return 0
        
        # Delete all keys
        deleted = r.delete(*keys)
        logger.info(f"Cleared {deleted} dedup cache keys")
        return deleted
        
    except redis.RedisError as e:
        logger.error(f"Redis error clearing dedup cache: {e}")
        raise


def get_dedup_stats() -> dict:
    """
    Get deduplication cache statistics.
    
    Returns:
        Dictionary with cache statistics
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        r = get_redis_client()
        
        # Find all dedup keys
        pattern = "dedup:phash:*"
        keys = r.keys(pattern)
        
        if not keys:
            return {
                "total_keys": 0,
                "total_hashes": 0,
                "prefixes": []
            }
        
        # Get stats for each prefix
        prefixes = []
        total_hashes = 0
        
        for key in keys:
            prefix = key.decode('utf-8').split(':')[-1]
            count = r.scard(key)
            ttl = r.ttl(key)
            
            prefixes.append({
                "prefix": prefix,
                "hash_count": count,
                "ttl_seconds": ttl if ttl > 0 else None
            })
            total_hashes += count
        
        return {
            "total_keys": len(keys),
            "total_hashes": total_hashes,
            "prefixes": prefixes
        }
        
    except redis.RedisError as e:
        logger.error(f"Redis error getting dedup stats: {e}")
        raise


def health_check() -> dict:
    """
    Check deduplication service health.
    
    Returns:
        Health status dictionary
    """
    try:
        r = get_redis_client()
        r.ping()
        
        # Test basic operations
        test_key = "dedup:health:test"
        r.sadd(test_key, "test")
        r.delete(test_key)
        
        return {
            "status": "healthy",
            "redis_connected": True,
            "operations_working": True
        }
        
    except Exception as e:
        logger.error(f"Dedup health check failed: {e}")
        return {
            "status": "unhealthy",
            "redis_connected": False,
            "operations_working": False,
            "error": str(e)
        }
