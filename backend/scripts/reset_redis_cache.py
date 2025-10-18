#!/usr/bin/env python3
"""
Redis Cache Reset Script

This script provides various methods to reset Redis cache during development and testing.
It can be run standalone or imported as a module.

Usage:
    python scripts/reset_redis_cache.py --help
    python scripts/reset_redis_cache.py --all
    python scripts/reset_redis_cache.py --tenant test-tenant-123
    python scripts/reset_redis_cache.py --db 15
"""

import argparse
import asyncio
import os
import sys
from typing import Optional, List

# Try to import redis, but handle missing dependency gracefully
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: redis module not available. Install with: pip install redis")

# Add the backend directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from app.crawler.cache import HybridCacheService
    from app.core.config import get_settings
    APP_MODULES_AVAILABLE = True
except ImportError as e:
    APP_MODULES_AVAILABLE = False
    print(f"Warning: App modules not available: {e}")
    print("Make sure you're running from the project root and dependencies are installed")


class RedisCacheReset:
    """Redis cache reset utility."""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis cache reset utility.
        
        Args:
            redis_url: Redis URL (defaults to settings)
        """
        if not REDIS_AVAILABLE:
            raise ImportError("Redis module not available. Install with: pip install redis")
        
        if not APP_MODULES_AVAILABLE:
            raise ImportError("App modules not available. Check your environment setup.")
        
        self.settings = get_settings()
        self.redis_url = redis_url or self.settings.redis_url
        self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
        self.cache_service = None
    
    async def setup_cache_service(self):
        """Set up cache service."""
        self.cache_service = HybridCacheService(redis_url=self.redis_url)
    
    def test_redis_connection(self) -> bool:
        """Test Redis connection."""
        try:
            self.redis_client.ping()
            print(f"✅ Redis connection successful: {self.redis_url}")
            return True
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            return False
    
    def get_redis_info(self) -> dict:
        """Get Redis server information."""
        try:
            info = self.redis_client.info()
            db_size = self.redis_client.dbsize()
            
            return {
                "redis_version": info.get("redis_version", "unknown"),
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "db_size": db_size,
                "uptime": info.get("uptime_in_seconds", 0)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def list_cache_keys(self, pattern: str = "*", limit: int = 100) -> List[str]:
        """List cache keys matching pattern."""
        try:
            keys = self.redis_client.keys(pattern)
            return keys[:limit] if len(keys) > limit else keys
        except Exception as e:
            print(f"Failed to list keys: {e}")
            return []
    
    async def clear_all_cache(self) -> bool:
        """Clear all cache data."""
        try:
            print("🧹 Clearing all cache data...")
            
            # Clear Redis
            self.redis_client.flushdb()
            print("  ✅ Redis cache cleared")
            
            # Clear cache service
            if not self.cache_service:
                await self.setup_cache_service()
            
            success = await self.cache_service.clear_all_cache()
            if success:
                print("  ✅ Cache service cleared")
            else:
                print("  ⚠️  Cache service clear had issues")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Failed to clear cache: {e}")
            return False
    
    async def clear_tenant_cache(self, tenant_id: str) -> int:
        """Clear cache for specific tenant."""
        try:
            print(f"🧹 Clearing cache for tenant: {tenant_id}")
            
            if not self.cache_service:
                await self.setup_cache_service()
            
            deleted_count = await self.cache_service.invalidate_tenant_cache(tenant_id)
            print(f"  ✅ Cleared {deleted_count} cache entries for tenant {tenant_id}")
            
            return deleted_count
            
        except Exception as e:
            print(f"  ❌ Failed to clear tenant cache: {e}")
            return 0
    
    async def clear_cache_pattern(self, pattern: str) -> int:
        """Clear cache keys matching pattern."""
        try:
            print(f"🧹 Clearing cache keys matching pattern: {pattern}")
            
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                print(f"  ✅ Deleted {deleted_count} keys matching pattern")
                return deleted_count
            else:
                print(f"  ℹ️  No keys found matching pattern")
                return 0
                
        except Exception as e:
            print(f"  ❌ Failed to clear pattern cache: {e}")
            return 0
    
    def reset_cache_stats(self):
        """Reset cache statistics."""
        try:
            if not self.cache_service:
                asyncio.run(self.setup_cache_service())
            
            self.cache_service.reset_cache_stats()
            print("✅ Cache statistics reset")
            
        except Exception as e:
            print(f"❌ Failed to reset cache stats: {e}")
    
    def close(self):
        """Close Redis connection."""
        if self.redis_client:
            self.redis_client.close()


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Reset Redis cache for development and testing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/reset_redis_cache.py --all                    # Clear all cache
  python scripts/reset_redis_cache.py --tenant test-tenant-123 # Clear tenant cache
  python scripts/reset_redis_cache.py --db 15                  # Use specific DB
  python scripts/reset_redis_cache.py --pattern "embedding:*"  # Clear pattern
  python scripts/reset_redis_cache.py --info                   # Show Redis info
  python scripts/reset_redis_cache.py --list-keys              # List cache keys
  
Alternative methods if dependencies are missing:
  docker compose exec redis redis-cli FLUSHDB                  # Clear Redis directly
  curl -X DELETE http://localhost:8000/cache/all               # Clear via API
        """
    )
    
    parser.add_argument("--all", action="store_true", help="Clear all cache data")
    parser.add_argument("--tenant", type=str, help="Clear cache for specific tenant")
    parser.add_argument("--pattern", type=str, help="Clear cache keys matching pattern")
    parser.add_argument("--db", type=int, help="Use specific Redis database")
    parser.add_argument("--info", action="store_true", help="Show Redis server information")
    parser.add_argument("--list-keys", action="store_true", help="List cache keys")
    parser.add_argument("--stats", action="store_true", help="Reset cache statistics")
    parser.add_argument("--redis-url", type=str, help="Custom Redis URL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    
    args = parser.parse_args()
    
    # Check if dependencies are available
    if not REDIS_AVAILABLE or not APP_MODULES_AVAILABLE:
        print("\n❌ Missing dependencies!")
        print("\nTo fix this, run one of the following:")
        print("1. Install dependencies:")
        print("   pip install redis")
        print("   # Or install all backend dependencies:")
        print("   cd backend && pip install -r requirements.txt")
        print("\n2. Use alternative methods:")
        print("   # Clear Redis directly via Docker:")
        print("   docker compose exec redis redis-cli FLUSHDB")
        print("   # Clear via API:")
        print("   curl -X DELETE http://localhost:8000/cache/all")
        print("   # Clear via Makefile (if available):")
        print("   make reset-redis-docker")
        sys.exit(1)
    
    # Determine Redis URL
    redis_url = args.redis_url
    if args.db is not None:
        settings = get_settings()
        redis_url = f"redis://redis:6379/{args.db}"
    
    # Initialize reset utility
    try:
        reset_util = RedisCacheReset(redis_url)
    except Exception as e:
        print(f"❌ Failed to initialize Redis reset utility: {e}")
        print("\nAlternative methods:")
        print("  docker compose exec redis redis-cli FLUSHDB")
        print("  curl -X DELETE http://localhost:8000/cache/all")
        sys.exit(1)
    
    # Test connection
    if not reset_util.test_redis_connection():
        sys.exit(1)
    
    # Show Redis info
    if args.info:
        info = reset_util.get_redis_info()
        print("\n📊 Redis Server Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        print()
    
    # List cache keys
    if args.list_keys:
        keys = reset_util.list_cache_keys(limit=50)
        print(f"\n🔑 Cache Keys ({len(keys)} found):")
        for key in keys:
            print(f"  {key}")
        print()
    
    # Dry run mode
    if args.dry_run:
        print("🔍 Dry run mode - no changes will be made")
        print(f"Redis URL: {reset_util.redis_url}")
        if args.all:
            print("Would clear all cache data")
        if args.tenant:
            print(f"Would clear cache for tenant: {args.tenant}")
        if args.pattern:
            print(f"Would clear keys matching pattern: {args.pattern}")
        if args.stats:
            print("Would reset cache statistics")
        return
    
    # Execute operations
    try:
        if args.all:
            await reset_util.clear_all_cache()
        
        if args.tenant:
            await reset_util.clear_tenant_cache(args.tenant)
        
        if args.pattern:
            await reset_util.clear_cache_pattern(args.pattern)
        
        if args.stats:
            reset_util.reset_cache_stats()
        
        # If no specific operation, show help
        if not any([args.all, args.tenant, args.pattern, args.stats]):
            parser.print_help()
    
    finally:
        reset_util.close()


if __name__ == "__main__":
    asyncio.run(main())
