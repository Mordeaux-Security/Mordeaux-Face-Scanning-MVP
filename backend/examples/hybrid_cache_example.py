#!/usr/bin/env python3
"""
Hybrid Cache Service V2 - Usage Examples

This script demonstrates how to use the hybrid cache service that combines
Redis (fast, volatile) and PostgreSQL (persistent, reliable) caching.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.cache_v2 import get_hybrid_cache_service

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def example_face_embeddings_cache():
    """Example: Caching face embeddings."""
    print("\n" + "="*60)
    print("FACE EMBEDDINGS CACHE EXAMPLE")
    print("="*60)
    
    cache = get_hybrid_cache_service()
    
    # Simulate image content
    image_bytes = b"fake_image_data_for_testing"
    tenant_id = "test_tenant"
    embeddings = [
        {
            "bbox": [100, 150, 200, 250],
            "embedding": [0.1, 0.2, 0.3] * 512,  # Simulate 512-dim embedding
            "det_score": 0.95
        }
    ]
    
    # Cache the embeddings
    print("1. Caching face embeddings...")
    success = await cache.cache_face_embeddings(image_bytes, tenant_id, embeddings)
    print(f"   Cache result: {'Success' if success else 'Failed'}")
    
    # Retrieve the embeddings
    print("2. Retrieving cached face embeddings...")
    cached_embeddings = await cache.get_cached_face_embeddings(image_bytes, tenant_id)
    
    if cached_embeddings:
        print(f"   Retrieved {len(cached_embeddings)} embeddings")
        print(f"   First embedding bbox: {cached_embeddings[0]['bbox']}")
        print(f"   Detection score: {cached_embeddings[0]['det_score']}")
    else:
        print("   No cached embeddings found")


async def example_perceptual_hash_cache():
    """Example: Caching perceptual hashes."""
    print("\n" + "="*60)
    print("PERCEPTUAL HASH CACHE EXAMPLE")
    print("="*60)
    
    cache = get_hybrid_cache_service()
    
    # Simulate image content
    image_bytes = b"fake_image_data_for_testing_hash"
    tenant_id = "test_tenant"
    phash = "a1b2c3d4e5f6g7h8"
    
    # Cache the perceptual hash
    print("1. Caching perceptual hash...")
    success = await cache.cache_perceptual_hash(image_bytes, tenant_id, phash)
    print(f"   Cache result: {'Success' if success else 'Failed'}")
    
    # Retrieve the perceptual hash
    print("2. Retrieving cached perceptual hash...")
    cached_phash = await cache.get_cached_perceptual_hash(image_bytes, tenant_id)
    
    if cached_phash:
        print(f"   Retrieved hash: {cached_phash}")
        print(f"   Hash matches: {cached_phash == phash}")
    else:
        print("   No cached hash found")


async def example_crawl_cache():
    """Example: Crawl cache for duplicate prevention."""
    print("\n" + "="*60)
    print("CRAWL CACHE EXAMPLE (DUPLICATE PREVENTION)")
    print("="*60)
    
    cache = get_hybrid_cache_service()
    
    # Simulate crawling scenarios
    url1 = "https://example.com/image1.jpg"
    url2 = "https://example.com/image2.jpg"
    image_bytes1 = b"fake_image_1_data"
    image_bytes2 = b"fake_image_2_data"
    tenant_id = "crawler_tenant"
    
    # Check if images should be skipped (first time - should not skip)
    print("1. Checking if images should be skipped (first time)...")
    should_skip1, raw_key1 = await cache.should_skip_crawled_image(url1, image_bytes1, tenant_id)
    should_skip2, raw_key2 = await cache.should_skip_crawled_image(url2, image_bytes2, tenant_id)
    
    print(f"   Image 1 should skip: {should_skip1} (raw_key: {raw_key1})")
    print(f"   Image 2 should skip: {should_skip2} (raw_key: {raw_key2})")
    
    # Store the crawled images
    print("2. Storing crawled images...")
    success1 = await cache.store_crawled_image(url1, image_bytes1, "raw_key_1", "thumb_key_1", tenant_id)
    success2 = await cache.store_crawled_image(url2, image_bytes2, "raw_key_2", "thumb_key_2", tenant_id)
    
    print(f"   Store result 1: {'Success' if success1 else 'Failed'}")
    print(f"   Store result 2: {'Success' if success2 else 'Failed'}")
    
    # Check again if images should be skipped (should skip now)
    print("3. Checking if images should be skipped (second time)...")
    should_skip1_again, raw_key1_again = await cache.should_skip_crawled_image(url1, image_bytes1, tenant_id)
    should_skip2_again, raw_key2_again = await cache.should_skip_crawled_image(url2, image_bytes2, tenant_id)
    
    print(f"   Image 1 should skip: {should_skip1_again} (raw_key: {raw_key1_again})")
    print(f"   Image 2 should skip: {should_skip2_again} (raw_key: {raw_key2_again})")


async def example_cache_stats():
    """Example: Getting cache statistics."""
    print("\n" + "="*60)
    print("CACHE STATISTICS EXAMPLE")
    print("="*60)
    
    cache = get_hybrid_cache_service()
    
    # Get comprehensive cache statistics
    print("Getting cache statistics...")
    stats = await cache.get_cache_stats()
    
    print("Cache Status:")
    print(f"  Redis enabled: {stats['redis_enabled']}")
    print(f"  PostgreSQL enabled: {stats['postgres_enabled']}")
    
    if stats['redis_stats']:
        print("\nRedis Statistics:")
        redis_stats = stats['redis_stats']
        if 'error' in redis_stats:
            print(f"  Error: {redis_stats['error']}")
        else:
            print(f"  Memory used: {redis_stats.get('used_memory_human', 'N/A')}")
            print(f"  Connected clients: {redis_stats.get('connected_clients', 'N/A')}")
            print(f"  Hit rate: {redis_stats.get('hit_rate', 'N/A')}%")
    
    if stats['postgres_stats']:
        print("\nPostgreSQL Statistics:")
        postgres_stats = stats['postgres_stats']
        if 'error' in postgres_stats:
            print(f"  Error: {postgres_stats['error']}")
        else:
            print(f"  Face embeddings cache: {postgres_stats.get('face_embeddings_cache_count', 0)} entries")
            print(f"  Perceptual hash cache: {postgres_stats.get('perceptual_hash_cache_count', 0)} entries")
            print(f"  Crawl cache: {postgres_stats.get('crawl_cache_count', 0)} entries")


async def example_tenant_management():
    """Example: Tenant cache management."""
    print("\n" + "="*60)
    print("TENANT CACHE MANAGEMENT EXAMPLE")
    print("="*60)
    
    cache = get_hybrid_cache_service()
    
    # Cache some data for a tenant
    tenant_id = "demo_tenant"
    image_bytes = b"tenant_specific_image_data"
    
    print(f"1. Caching data for tenant: {tenant_id}")
    await cache.cache_face_embeddings(image_bytes, tenant_id, [{"test": "data"}])
    await cache.cache_perceptual_hash(image_bytes, tenant_id, "tenant_hash_123")
    
    # Get stats before invalidation
    print("2. Getting cache stats before invalidation...")
    stats_before = await cache.get_cache_stats()
    print(f"   PostgreSQL face embeddings: {stats_before['postgres_stats'].get('face_embeddings_cache_count', 0)}")
    print(f"   PostgreSQL perceptual hash: {stats_before['postgres_stats'].get('perceptual_hash_cache_count', 0)}")
    
    # Invalidate tenant cache
    print(f"3. Invalidating cache for tenant: {tenant_id}")
    invalidated_count = await cache.invalidate_tenant_cache(tenant_id)
    print(f"   Invalidated {invalidated_count} cache entries")
    
    # Get stats after invalidation
    print("4. Getting cache stats after invalidation...")
    stats_after = await cache.get_cache_stats()
    print(f"   PostgreSQL face embeddings: {stats_after['postgres_stats'].get('face_embeddings_cache_count', 0)}")
    print(f"   PostgreSQL perceptual hash: {stats_after['postgres_stats'].get('perceptual_hash_cache_count', 0)}")


async def main():
    """Run all examples."""
    print("HYBRID CACHE SERVICE V2 - USAGE EXAMPLES")
    print("="*60)
    print("This demonstrates the hybrid Redis + PostgreSQL caching system")
    print("that provides both speed (Redis) and persistence (PostgreSQL).")
    
    try:
        # Run examples
        await example_face_embeddings_cache()
        await example_perceptual_hash_cache()
        await example_crawl_cache()
        await example_cache_stats()
        await example_tenant_management()
        
        print("\n" + "="*60)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\nExample failed: {e}")
        print("Make sure Redis and PostgreSQL are running and accessible.")


if __name__ == "__main__":
    asyncio.run(main())
