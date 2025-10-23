"""
Caching facade module for the crawler service.

This module provides a clean interface to the hybrid cache service, abstracting away
the complexity of Redis and PostgreSQL caching operations.
"""

import logging
from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


class CachingFacade:
    """
    Caching facade providing a clean interface to cache operations.
    
    Abstracts away the complexity of hybrid caching (Redis + PostgreSQL),
    providing simple methods for cache checking and statistics.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._cache_service = None
    
    def _get_cache_service(self):
        """Get the hybrid cache service instance."""
        if self._cache_service is None:
            from ..cache import get_hybrid_cache_service
            self._cache_service = get_hybrid_cache_service()
        return self._cache_service
    
    async def should_skip_image(self, image_url: str, image_bytes: bytes, tenant_id: str = "default") -> Tuple[bool, Optional[str]]:
        """
        Check if an image should be skipped based on crawl cache.
        
        This is the main duplicate prevention mechanism for crawlers.
        
        Args:
            image_url: URL of the image to check
            image_bytes: Image data bytes
            tenant_id: Tenant identifier for multi-tenancy
            
        Returns:
            Tuple of (should_skip, cached_key) where:
            - should_skip: True if image should be skipped
            - cached_key: Cache key if found in cache, None otherwise
        """
        try:
            cache_service = self._get_cache_service()
            return await cache_service.should_skip_crawled_image(image_url, image_bytes, tenant_id)
        except Exception as e:
            self.logger.error(f"Error checking cache for image {self._truncate_log_string(image_url)}: {e}")
            return False, None
    
    async def get_cache_statistics(self) -> Dict[str, int]:
        """
        Get cache hit/miss statistics.
        
        Returns:
            Dictionary with cache statistics including:
            - redis_hits: Number of Redis cache hits
            - postgres_hits: Number of PostgreSQL cache hits  
            - cache_misses: Number of cache misses
        """
        try:
            cache_service = self._get_cache_service()
            return await cache_service.get_cache_stats()
        except Exception as e:
            self.logger.error(f"Error getting cache statistics: {e}")
            return {
                'redis_hits': 0,
                'postgres_hits': 0,
                'cache_misses': 0
            }
    
    def reset_cache_statistics(self):
        """Reset cache hit/miss counters."""
        try:
            cache_service = self._get_cache_service()
            cache_service.reset_cache_stats()
        except Exception as e:
            self.logger.error(f"Error resetting cache statistics: {e}")
    
    async def close(self):
        """Close cache service connections."""
        try:
            if self._cache_service:
                # Note: The hybrid cache service doesn't have a close method,
                # but we could add cleanup here if needed
                pass
        except Exception as e:
            self.logger.error(f"Error closing cache service: {e}")
    
    def _truncate_log_string(self, text: str, max_length: int = 120) -> str:
        """
        Truncate long strings for logging with a hash suffix for identification.
        
        Args:
            text: String to truncate
            max_length: Maximum length before truncation
            
        Returns:
            Truncated string with hash suffix if truncated
        """
        if len(text) <= max_length:
            return text
        
        # Create a short hash of the original string for identification
        import hashlib
        hash_suffix = hashlib.md5(text.encode()).hexdigest()[:8]
        truncated = text[:max_length - len(hash_suffix) - 3]  # Reserve space for "..." and hash
        return f"{truncated}...{hash_suffix}"
