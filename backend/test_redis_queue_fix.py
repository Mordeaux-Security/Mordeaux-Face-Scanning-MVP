#!/usr/bin/env python3
"""
Test script to verify Redis queue fix and 3-3-1-1 configuration
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.redis_queues import get_redis_client, setup_redis_queues, push_sites_to_queue, get_site_from_queue
from app.services.multiprocess_crawler import MultiprocessCrawler

def test_redis_queue():
    """Test Redis queue operations"""
    print("Testing Redis queue operations...")
    
    # Get Redis client
    redis_client = get_redis_client()
    
    # Test 1: Clear queues
    print("1. Clearing Redis queues...")
    setup_redis_queues(redis_client)
    
    # Test 2: Push sites
    test_sites = ["https://example.com", "https://test.com", "https://demo.com"]
    print(f"2. Pushing {len(test_sites)} sites to queue...")
    push_sites_to_queue(redis_client, test_sites)
    
    # Test 3: Verify queue length
    queue_length = redis_client.llen('sites_to_crawl')
    print(f"3. Queue length after pushing: {queue_length}")
    
    if queue_length != len(test_sites):
        print("❌ ERROR: Queue length doesn't match expected value!")
        return False
    
    # Test 4: Pop sites
    print("4. Popping sites from queue...")
    popped_sites = []
    for i in range(len(test_sites)):
        site = get_site_from_queue(redis_client, timeout=5)
        if site:
            popped_sites.append(site)
            print(f"   Popped: {site}")
    
    print(f"5. Popped {len(popped_sites)} sites")
    
    if len(popped_sites) != len(test_sites):
        print("❌ ERROR: Didn't pop all sites!")
        return False
    
    print("✅ Redis queue test passed!")
    return True

def test_multiprocess_crawler_config():
    """Test multiprocess crawler configuration"""
    print("\nTesting multiprocess crawler configuration...")
    
    # Test 3-3-1-1 configuration
    crawler = MultiprocessCrawler(
        num_crawlers=3,
        num_extractors=3,
        num_gpu_workers=1,
        num_batch_processors=1,
        batch_size=64
    )
    
    print(f"Crawlers: {crawler.num_crawlers}")
    print(f"Extractors: {crawler.num_extractors}")
    print(f"GPU Workers: {crawler.num_gpu_workers}")
    print(f"Batch Processors: {crawler.num_batch_processors}")
    print(f"Batch Size: {crawler.batch_size}")
    
    # Verify configuration
    expected_config = (3, 3, 1, 1)
    actual_config = (crawler.num_crawlers, crawler.num_extractors, 
                    crawler.num_gpu_workers, crawler.num_batch_processors)
    
    if actual_config != expected_config:
        print(f"❌ ERROR: Configuration mismatch! Expected {expected_config}, got {actual_config}")
        return False
    
    print("✅ Multiprocess crawler configuration test passed!")
    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("REDIS QUEUE FIX AND 3-3-1-1 CONFIGURATION TEST")
    print("=" * 60)
    
    # Test Redis queue
    redis_test_passed = test_redis_queue()
    
    # Test multiprocess crawler config
    config_test_passed = test_multiprocess_crawler_config()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Redis Queue Test: {'✅ PASSED' if redis_test_passed else '❌ FAILED'}")
    print(f"Configuration Test: {'✅ PASSED' if config_test_passed else '❌ FAILED'}")
    
    if redis_test_passed and config_test_passed:
        print("\n🎉 All tests passed! Redis queue fix and 3-3-1-1 configuration are working.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())


