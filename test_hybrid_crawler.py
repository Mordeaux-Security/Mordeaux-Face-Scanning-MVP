#!/usr/bin/env python3
"""
Test script for the crawler with hybrid cache integration.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the app directory to the path
sys.path.insert(0, str(Path(__file__).parent / "backend"))

from app.services.crawler import EnhancedImageCrawler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_crawler_with_hybrid_cache():
    """Test the crawler with hybrid cache."""
    print("Testing Enhanced Image Crawler with Hybrid Cache")
    print("=" * 60)
    
    # Test URL (using a simple example)
    test_url = "https://example.com"
    
    # Create crawler with hybrid cache
    crawler_config = {
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'timeout': 30,
        'min_face_quality': 0.5,
        'require_face': False,  # Don't require faces for testing
        'crop_faces': True,
        'face_margin': 0.2,
        'max_total_images': 5,  # Small number for testing
        'max_pages': 1,  # Just one page
        'same_domain_only': True,
        'similarity_threshold': 5,
        'max_concurrent_images': 3,
        'batch_size': 10,
    }
    
    try:
        async with EnhancedImageCrawler(**crawler_config) as crawler:
            print(f"Starting crawl of: {test_url}")
            print(f"Crawler config: {crawler_config}")
            print(f"Cache service type: {type(crawler.cache_service).__name__}")
            
            # Test the crawl
            result = await crawler.crawl_page(test_url, method="all")
            
            print("\n" + "=" * 60)
            print("CRAWL RESULTS")
            print("=" * 60)
            print(f"URL: {result.url}")
            print(f"Targeting method: {result.targeting_method}")
            print(f"Pages crawled: {result.pages_crawled}")
            print(f"Images found: {result.images_found}")
            print(f"Raw images saved: {result.raw_images_saved}")
            print(f"Thumbnails saved: {result.thumbnails_saved}")
            print(f"Cache hits: {result.cache_hits}")
            print(f"Cache misses: {result.cache_misses}")
            
            if result.errors:
                print(f"\nErrors encountered:")
                for error in result.errors[:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(result.errors) > 5:
                    print(f"  ... and {len(result.errors) - 5} more errors")
            
            print(f"\nSaved raw image keys:")
            for key in result.saved_raw_keys[:3]:  # Show first 3 keys
                print(f"  - {key}")
            if len(result.saved_raw_keys) > 3:
                print(f"  ... and {len(result.saved_raw_keys) - 3} more")
            
            print(f"\nSaved thumbnail keys:")
            for key in result.saved_thumbnail_keys[:3]:  # Show first 3 keys
                print(f"  - {key}")
            if len(result.saved_thumbnail_keys) > 3:
                print(f"  ... and {len(result.saved_thumbnail_keys) - 3} more")
            
            # Test cache statistics
            print("\n" + "=" * 60)
            print("CACHE STATISTICS")
            print("=" * 60)
            try:
                cache_stats = await crawler.cache_service.get_cache_stats()
                print(f"Redis enabled: {cache_stats['redis_enabled']}")
                print(f"PostgreSQL enabled: {cache_stats['postgres_enabled']}")
                
                if cache_stats['redis_stats']:
                    redis_stats = cache_stats['redis_stats']
                    if 'error' in redis_stats:
                        print(f"Redis error: {redis_stats['error']}")
                    else:
                        print(f"Redis memory: {redis_stats.get('used_memory_human', 'N/A')}")
                        print(f"Redis hit rate: {redis_stats.get('hit_rate', 'N/A')}%")
                
                if cache_stats['postgres_stats']:
                    postgres_stats = cache_stats['postgres_stats']
                    if 'error' in postgres_stats:
                        print(f"PostgreSQL error: {postgres_stats['error']}")
                    else:
                        print(f"PostgreSQL face embeddings: {postgres_stats.get('face_embeddings_cache_count', 0)}")
                        print(f"PostgreSQL perceptual hash: {postgres_stats.get('perceptual_hash_cache_count', 0)}")
                        print(f"PostgreSQL crawl cache: {postgres_stats.get('crawl_cache_count', 0)}")
                        
            except Exception as e:
                print(f"Error getting cache stats: {e}")
            
            print("\n" + "=" * 60)
            print("TEST COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            
            return True
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main test function."""
    print("HYBRID CACHE CRAWLER TEST")
    print("=" * 60)
    print("This test verifies that the crawler works with the new hybrid cache.")
    print("Make sure Redis and PostgreSQL are running.")
    
    success = await test_crawler_with_hybrid_cache()
    
    if success:
        print("\n✅ Test passed! The crawler is working with the hybrid cache.")
    else:
        print("\n❌ Test failed! Check the logs for details.")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
