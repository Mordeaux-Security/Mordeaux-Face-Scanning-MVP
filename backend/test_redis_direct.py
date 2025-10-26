#!/usr/bin/env python3
"""
Direct Redis test without setup_redis_queues
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_direct_redis():
    """Test Redis directly without setup_redis_queues"""
    try:
        from app.services.redis_queues import get_redis_client
        
        redis_client = get_redis_client()
        
        # Clear the queue manually
        redis_client.delete('sites_to_crawl')
        print("✅ Manually cleared sites_to_crawl")
        
        # Check initial state
        initial_length = redis_client.llen('sites_to_crawl')
        print(f"✅ Initial queue length: {initial_length}")
        
        # Push sites directly
        test_sites = ["https://example.com", "https://test.com"]
        for i, site in enumerate(test_sites):
            result = redis_client.rpush('sites_to_crawl', site)
            print(f"✅ Pushed site {i+1}: {site}, result={result}")
            
            # Check length immediately
            length = redis_client.llen('sites_to_crawl')
            print(f"✅ Queue length after site {i+1}: {length}")
            
            # Check if key exists
            exists = redis_client.exists('sites_to_crawl')
            print(f"✅ Key exists after site {i+1}: {exists}")
        
        # Final check
        final_length = redis_client.llen('sites_to_crawl')
        print(f"✅ Final queue length: {final_length}")
        
        if final_length == len(test_sites):
            print("✅ Direct Redis test passed!")
            return True
        else:
            print(f"❌ Direct Redis test failed: expected {len(test_sites)}, got {final_length}")
            return False
            
    except Exception as e:
        print(f"❌ Direct Redis test failed: {e}")
        return False

def main():
    """Run direct Redis test"""
    print("Testing Redis directly...")
    
    if test_direct_redis():
        print("🎉 Direct Redis test passed!")
        return 0
    else:
        print("❌ Direct Redis test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())


