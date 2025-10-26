#!/usr/bin/env python3
"""
Simple Redis queue test for Docker environment
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_redis_connection():
    """Test basic Redis connection"""
    try:
        from app.services.redis_queues import get_redis_client
        redis_client = get_redis_client()
        redis_client.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_queue_operations():
    """Test queue operations"""
    try:
        from app.services.redis_queues import get_redis_client, setup_redis_queues, push_sites_to_queue
        
        redis_client = get_redis_client()
        
        # Clear queues
        print("✅ About to clear Redis queues...")
        setup_redis_queues(redis_client)
        print("✅ Redis queues cleared")
        
        # Check if sites_to_crawl was deleted
        exists_after_clear = redis_client.exists('sites_to_crawl')
        print(f"✅ sites_to_crawl exists after clear: {exists_after_clear}")
        
        # Check queue is empty
        initial_length = redis_client.llen('sites_to_crawl')
        print(f"✅ Initial queue length: {initial_length}")
        
        # Check Redis info
        info = redis_client.info()
        print(f"✅ Redis connected to DB: {redis_client.connection_pool.connection_kwargs.get('db', 0)}")
        
        # Test with a simple key first
        redis_client.set('test_key', 'test_value')
        test_value = redis_client.get('test_key')
        print(f"✅ Simple key test: {test_value}")
        
        # Test with a list operation
        redis_client.delete('test_list')
        redis_client.rpush('test_list', 'item1')
        redis_client.rpush('test_list', 'item2')
        list_length = redis_client.llen('test_list')
        print(f"✅ List test: {list_length} items")
        
        # Check if the list persists
        redis_client2 = get_redis_client()
        list_length2 = redis_client2.llen('test_list')
        print(f"✅ List test with new client: {list_length2} items")
        
        # Push test sites one by one with detailed logging
        test_sites = ["https://example.com", "https://test.com"]
        for i, site in enumerate(test_sites):
            result = redis_client.rpush('sites_to_crawl', site)
            print(f"✅ Pushed site {i+1}: {site}, result={result}")
            # Check length after each push
            length = redis_client.llen('sites_to_crawl')
            print(f"✅ Queue length after site {i+1}: {length}")
            
            # Also check if the key exists
            exists = redis_client.exists('sites_to_crawl')
            print(f"✅ Key exists after site {i+1}: {exists}")
        
        print(f"✅ Pushed {len(test_sites)} sites to queue")
        
        # Check queue length immediately after pushing
        queue_length = redis_client.llen('sites_to_crawl')
        print(f"✅ Queue length after push: {queue_length}")
        
        # Also check with a different Redis client to see if it's a connection issue
        redis_client2 = get_redis_client()
        queue_length2 = redis_client2.llen('sites_to_crawl')
        print(f"✅ Queue length with new client: {queue_length2}")
        
        if queue_length == len(test_sites):
            print("✅ Queue operations working correctly")
            return True
        else:
            print(f"❌ Queue length mismatch: expected {len(test_sites)}, got {queue_length}")
            print(f"❌ This suggests the queue is being cleared somewhere else")
            return False
            
    except Exception as e:
        print(f"❌ Queue operations failed: {e}")
        return False

def main():
    """Run tests"""
    print("Testing Redis queue fix...")
    
    # Test connection
    if not test_redis_connection():
        return 1
    
    # Test queue operations
    if not test_queue_operations():
        return 1
    
    print("🎉 All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
