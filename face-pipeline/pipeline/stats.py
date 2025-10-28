"""
Statistics Tracking Service

Redis-based statistics tracking with atomic counters for real-time
monitoring of face processing pipeline performance.

Architecture:
- Global counters: `stats:global:processed`, `stats:global:rejected`, `stats:global:dup_skipped`
- Tenant counters: `stats:tenant:{tenant_id}:processed`, etc.
- Atomic increments: `INCR` for thread-safety
- Batch operations: `INCRBY` for pipeline results
- Timing metrics: Hash-based storage for compact timing aggregation

Key Features:
- Thread-safe atomic operations
- Tenant-scoped statistics
- Real-time counter updates
- Batch processing support
- Admin utilities for testing
- Lightweight timing metrics with context managers
"""

import logging
import redis
import contextlib
import time
from typing import Optional, Dict, Any
from config.settings import settings

logger = logging.getLogger(__name__)

# Redis client singleton
_redis_client: Optional[redis.Redis] = None

# Namespacing (single hash per category keeps it compact)
KEY_COUNTERS = "stats:counters"
KEY_TIMINGS = "stats:timings_ms"


def get_redis_client() -> redis.Redis:
    """Get Redis client singleton."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(settings.redis_url)
    return _redis_client


# ============================================================================
# LIGHTWEIGHT TIMING METRICS (Hash-based storage for compact aggregation)
# ============================================================================

def inc(name: str, n: int = 1) -> None:
    """
    Increment counter using HINCRBY on stats:counters hash.
    
    Args:
        name: Counter name (e.g., "images_total", "faces_detected")
        n: Increment amount (default 1)
    """
    try:
        r = get_redis_client()
        r.hincrby(KEY_COUNTERS, name, n)
        logger.debug(f"Incremented counter {name} by {n}")
    except redis.RedisError as e:
        logger.error(f"Redis error incrementing counter {name}: {e}")


def add_time_ms(name: str, ms: float) -> None:
    """
    Accumulate timing using HINCRBYFLOAT on stats:timings_ms hash.
    
    Args:
        name: Timing name (e.g., "detect_ms", "embed_ms")
        ms: Milliseconds to add
    """
    try:
        r = get_redis_client()
        r.hincrbyfloat(KEY_TIMINGS, name, float(ms))
        logger.debug(f"Added {ms}ms to timing {name}")
    except redis.RedisError as e:
        logger.error(f"Redis error adding timing {name}: {e}")


@contextlib.contextmanager
def timer(name: str):
    """
    Context manager for automatic timing.
    
    Usage:
        with timer('detect_ms'):
            faces = detect_faces(img)
    
    Args:
        name: Timing name for the operation
    """
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        add_time_ms(name, dt)


def snapshot() -> dict:
    """
    Get combined counters and timings snapshot.
    
    Returns:
        Dictionary with 'counts' and 'timings_ms' keys containing
        all accumulated metrics from Redis hashes.
    """
    try:
        r = get_redis_client()
        
        # Atomic-ish read: HGETALL both hashes
        pipe = r.pipeline()
        pipe.hgetall(KEY_COUNTERS)
        pipe.hgetall(KEY_TIMINGS)
        c_raw, t_raw = pipe.execute()

        # Decode bytes -> proper types
        def decode_map(m):
            out = {}
            for k, v in m.items():
                key = k.decode() if isinstance(k, (bytes, bytearray)) else k
                val_s = v.decode() if isinstance(v, (bytes, bytearray)) else v
                # counters are ints where possible
                try:
                    val = int(val_s)
                except ValueError:
                    try:
                        val = float(val_s)
                    except ValueError:
                        val = val_s
                out[key] = val
            return out

        counters = decode_map(c_raw)
        timings = decode_map(t_raw)

        return {
            "counts": counters,
            "timings_ms": {k: round(float(v), 2) for k, v in timings.items()},
        }
        
    except redis.RedisError as e:
        logger.error(f"Redis error getting snapshot: {e}")
        return {"counts": {}, "timings_ms": {}}


def increment_processed(count: int, tenant_id: str) -> int:
    """
    Increment processed counter.
    
    Args:
        count: Number of faces processed
        tenant_id: Tenant identifier
        
    Returns:
        New counter value
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        r = get_redis_client()
        
        # Update both global and tenant counters
        pipe = r.pipeline()
        pipe.incrby("stats:global:processed", count)
        pipe.incrby(f"stats:tenant:{tenant_id}:processed", count)
        results = pipe.execute()
        
        global_count = results[0]
        tenant_count = results[1]
        
        logger.debug(f"Incremented processed: global={global_count}, tenant={tenant_id}={tenant_count}")
        return global_count
        
    except redis.RedisError as e:
        logger.error(f"Redis error incrementing processed: {e}")
        raise


def increment_rejected(count: int, tenant_id: str) -> int:
    """
    Increment rejected counter.
    
    Args:
        count: Number of faces rejected
        tenant_id: Tenant identifier
        
    Returns:
        New counter value
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        r = get_redis_client()
        
        # Update both global and tenant counters
        pipe = r.pipeline()
        pipe.incrby("stats:global:rejected", count)
        pipe.incrby(f"stats:tenant:{tenant_id}:rejected", count)
        results = pipe.execute()
        
        global_count = results[0]
        tenant_count = results[1]
        
        logger.debug(f"Incremented rejected: global={global_count}, tenant={tenant_id}={tenant_count}")
        return global_count
        
    except redis.RedisError as e:
        logger.error(f"Redis error incrementing rejected: {e}")
        raise


def increment_dup_skipped(count: int, tenant_id: str) -> int:
    """
    Increment duplicate skipped counter.
    
    Args:
        count: Number of faces skipped as duplicates
        tenant_id: Tenant identifier
        
    Returns:
        New counter value
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        r = get_redis_client()
        
        # Update both global and tenant counters
        pipe = r.pipeline()
        pipe.incrby("stats:global:dup_skipped", count)
        pipe.incrby(f"stats:tenant:{tenant_id}:dup_skipped", count)
        results = pipe.execute()
        
        global_count = results[0]
        tenant_count = results[1]
        
        logger.debug(f"Incremented dup_skipped: global={global_count}, tenant={tenant_id}={tenant_count}")
        return global_count
        
    except redis.RedisError as e:
        logger.error(f"Redis error incrementing dup_skipped: {e}")
        raise


def get_stats(tenant_id: Optional[str] = None) -> Dict[str, int]:
    """
    Get current statistics.
    
    Args:
        tenant_id: Optional tenant ID to get tenant-specific stats
                  If None, returns global stats
        
    Returns:
        Dictionary with statistics
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        r = get_redis_client()
        
        if tenant_id:
            # Get tenant-specific stats
            pipe = r.pipeline()
            pipe.get(f"stats:tenant:{tenant_id}:processed")
            pipe.get(f"stats:tenant:{tenant_id}:rejected")
            pipe.get(f"stats:tenant:{tenant_id}:dup_skipped")
            results = pipe.execute()
            
            return {
                "processed": int(results[0] or 0),
                "rejected": int(results[1] or 0),
                "dup_skipped": int(results[2] or 0)
            }
        else:
            # Get global stats
            pipe = r.pipeline()
            pipe.get("stats:global:processed")
            pipe.get("stats:global:rejected")
            pipe.get("stats:global:dup_skipped")
            results = pipe.execute()
            
            return {
                "processed": int(results[0] or 0),
                "rejected": int(results[1] or 0),
                "dup_skipped": int(results[2] or 0)
            }
            
    except redis.RedisError as e:
        logger.error(f"Redis error getting stats: {e}")
        raise


def reset_stats(tenant_id: Optional[str] = None) -> int:
    """
    Reset statistics for testing or maintenance.
    
    Args:
        tenant_id: Optional tenant ID to reset only tenant-specific stats
                  If None, resets global stats
        
    Returns:
        Number of keys deleted
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        r = get_redis_client()
        
        if tenant_id:
            # Reset tenant-specific stats
            keys = [
                f"stats:tenant:{tenant_id}:processed",
                f"stats:tenant:{tenant_id}:rejected",
                f"stats:tenant:{tenant_id}:dup_skipped"
            ]
        else:
            # Reset global stats
            keys = [
                "stats:global:processed",
                "stats:global:rejected",
                "stats:global:dup_skipped"
            ]
        
        # Delete keys
        deleted = r.delete(*keys)
        logger.info(f"Reset stats: deleted {deleted} keys for {'tenant ' + tenant_id if tenant_id else 'global'}")
        return deleted
        
    except redis.RedisError as e:
        logger.error(f"Redis error resetting stats: {e}")
        raise


def get_all_tenant_stats() -> Dict[str, Dict[str, int]]:
    """
    Get statistics for all tenants.
    
    Returns:
        Dictionary mapping tenant_id to their stats
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        r = get_redis_client()
        
        # Find all tenant stat keys
        pattern = "stats:tenant:*:processed"
        keys = r.keys(pattern)
        
        tenant_stats = {}
        
        for key in keys:
            # Extract tenant_id from key
            key_str = key.decode('utf-8')
            tenant_id = key_str.split(':')[2]
            
            # Get all stats for this tenant
            pipe = r.pipeline()
            pipe.get(f"stats:tenant:{tenant_id}:processed")
            pipe.get(f"stats:tenant:{tenant_id}:rejected")
            pipe.get(f"stats:tenant:{tenant_id}:dup_skipped")
            results = pipe.execute()
            
            tenant_stats[tenant_id] = {
                "processed": int(results[0] or 0),
                "rejected": int(results[1] or 0),
                "dup_skipped": int(results[2] or 0)
            }
        
        return tenant_stats
        
    except redis.RedisError as e:
        logger.error(f"Redis error getting all tenant stats: {e}")
        raise


def get_stats_summary() -> Dict[str, Any]:
    """
    Get comprehensive statistics summary.
    
    Returns:
        Dictionary with global stats, tenant breakdown, and metadata
        
    Raises:
        redis.RedisError: If Redis operation fails
    """
    try:
        # Get global stats
        global_stats = get_stats()
        
        # Get tenant stats
        tenant_stats = get_all_tenant_stats()
        
        # Calculate totals from tenant stats for verification
        total_processed = sum(stats["processed"] for stats in tenant_stats.values())
        total_rejected = sum(stats["rejected"] for stats in tenant_stats.values())
        total_dup_skipped = sum(stats["dup_skipped"] for stats in tenant_stats.values())
        
        return {
            "global": global_stats,
            "tenants": tenant_stats,
            "tenant_count": len(tenant_stats),
            "verification": {
                "tenant_total_processed": total_processed,
                "tenant_total_rejected": total_rejected,
                "tenant_total_dup_skipped": total_dup_skipped,
                "global_processed": global_stats["processed"],
                "global_rejected": global_stats["rejected"],
                "global_dup_skipped": global_stats["dup_skipped"]
            }
        }
        
    except redis.RedisError as e:
        logger.error(f"Redis error getting stats summary: {e}")
        raise


def health_check() -> Dict[str, Any]:
    """
    Check statistics service health.
    
    Returns:
        Health status dictionary
    """
    try:
        r = get_redis_client()
        r.ping()
        
        # Test basic operations
        test_key = "stats:health:test"
        r.incr(test_key)
        r.delete(test_key)
        
        return {
            "status": "healthy",
            "redis_connected": True,
            "operations_working": True
        }
        
    except Exception as e:
        logger.error(f"Stats health check failed: {e}")
        return {
            "status": "unhealthy",
            "redis_connected": False,
            "operations_working": False,
            "error": str(e)
        }
