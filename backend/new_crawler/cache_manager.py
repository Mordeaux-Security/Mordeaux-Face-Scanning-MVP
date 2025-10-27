"""
Cache Manager for New Crawler System

Handles Redis caching with perceptual hash (phash) computation for deduplication.
Provides efficient image deduplication and caching of processing results.
"""

import hashlib
import logging
import time
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import imagehash
from PIL import Image
import io

from .config import get_config
from .redis_manager import get_redis_manager
from .data_structures import CandidateImage, ImageTask, FaceResult

logger = logging.getLogger(__name__)


class CacheManager:
    """Cache manager for image deduplication and result caching."""
    
    def __init__(self, config=None, redis_manager=None):
        self.config = config or get_config()
        self.redis = redis_manager or get_redis_manager()
        
    def compute_phash(self, image_path: str) -> Optional[str]:
        """Compute perceptual hash for an image file."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Compute perceptual hash
                phash = imagehash.phash(img)
                return str(phash)
        except Exception as e:
            logger.error(f"Failed to compute phash for {image_path}: {e}")
            return None
    
    def compute_phash_from_bytes(self, image_bytes: bytes) -> Optional[str]:
        """Compute perceptual hash from image bytes."""
        try:
            img = Image.open(io.BytesIO(image_bytes))
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Compute perceptual hash
            phash = imagehash.phash(img)
            return str(phash)
        except Exception as e:
            logger.error(f"Failed to compute phash from bytes: {e}")
            return None
    
    def compute_file_hash(self, file_path: str) -> Optional[str]:
        """Compute SHA256 hash of file content."""
        try:
            hash_sha256 = hashlib.sha256()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.error(f"Failed to compute file hash for {file_path}: {e}")
            return None
    
    def get_phash_cache_key(self, phash: str) -> str:
        """Get Redis cache key for phash."""
        return self.config.get_cache_key('phash', phash=phash)
    
    def is_image_cached(self, phash: str) -> bool:
        """Check if image with given phash is already cached."""
        cache_key = self.get_phash_cache_key(phash)
        return self.redis.exists_cache(cache_key)
    
    def get_cached_image_info(self, phash: str) -> Optional[Dict[str, Any]]:
        """Get cached image information."""
        cache_key = self.get_phash_cache_key(phash)
        return self.redis.get_cache(cache_key)
    
    def cache_image_info(self, phash: str, image_info: Dict[str, Any]) -> bool:
        """Cache image information."""
        cache_key = self.get_phash_cache_key(phash)
        return self.redis.set_cache(cache_key, image_info)
    
    def cache_face_result(self, phash: str, face_result: FaceResult) -> bool:
        """Cache face processing result."""
        cache_key = self.get_phash_cache_key(phash)
        cache_data = {
            'phash': phash,
            'faces_count': len(face_result.faces),
            'raw_image_key': face_result.raw_image_key,
            'thumbnail_keys': face_result.thumbnail_keys,
            'processing_time_ms': face_result.processing_time_ms,
            'gpu_used': face_result.gpu_used,
            'cached_at': time.time()
        }
        return self.redis.set_cache(cache_key, cache_data)
    
    def get_cached_face_result(self, phash: str) -> Optional[Dict[str, Any]]:
        """Get cached face processing result."""
        cache_key = self.get_phash_cache_key(phash)
        return self.redis.get_cache(cache_key)
    
    def should_skip_image(self, phash: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Check if image should be skipped due to caching."""
        cached_info = self.get_cached_image_info(phash)
        if cached_info:
            logger.debug(f"Image with phash {phash[:8]}... already cached, skipping")
            return True, cached_info
        return False, None
    
    def process_image_task(self, image_task: ImageTask) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Process image task and check cache."""
        # Check if image is already cached
        should_skip, cached_info = self.should_skip_image(image_task.phash)
        if should_skip:
            return True, cached_info
        
        # Image not cached, needs processing
        return False, None
    
    def store_processing_result(self, image_task: ImageTask, face_result: FaceResult) -> bool:
        """Store processing result in cache."""
        try:
            # Cache the face result
            success = self.cache_face_result(image_task.phash, face_result)
            
            if success:
                logger.debug(f"Cached processing result for phash {image_task.phash[:8]}...")
            
            return success
        except Exception as e:
            logger.error(f"Failed to store processing result: {e}")
            return False
    
    def get_site_stats_cache_key(self, site_id: str) -> str:
        """Get cache key for site statistics."""
        return self.config.get_cache_key('site_stats', site_id=site_id)
    
    def cache_site_stats(self, site_id: str, stats: Dict[str, Any]) -> bool:
        """Cache site processing statistics."""
        cache_key = self.get_site_stats_cache_key(site_id)
        return self.redis.set_cache(cache_key, stats)
    
    def get_cached_site_stats(self, site_id: str) -> Optional[Dict[str, Any]]:
        """Get cached site statistics."""
        cache_key = self.get_site_stats_cache_key(site_id)
        return self.redis.get_cache(cache_key)
    
    def update_site_stats(self, site_id: str, updates: Dict[str, Any]) -> bool:
        """Update cached site statistics."""
        try:
            # Get existing stats
            existing_stats = self.get_cached_site_stats(site_id) or {}
            
            # Update with new values
            existing_stats.update(updates)
            existing_stats['last_updated'] = time.time()
            
            # Cache updated stats
            return self.cache_site_stats(site_id, existing_stats)
        except Exception as e:
            logger.error(f"Failed to update site stats for {site_id}: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            # Get all phash cache keys
            client = self.redis._get_client()
            phash_keys = client.keys(self.config.get_cache_key('phash', phash='*'))
            site_stats_keys = client.keys(self.config.get_cache_key('site_stats', site_id='*'))
            
            return {
                'phash_cache_count': len(phash_keys),
                'site_stats_cache_count': len(site_stats_keys),
                'total_cache_keys': len(phash_keys) + len(site_stats_keys),
                'cache_ttl_days': self.config.nc_cache_ttl_days,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {
                'phash_cache_count': 0,
                'site_stats_cache_count': 0,
                'total_cache_keys': 0,
                'cache_ttl_days': self.config.nc_cache_ttl_days,
                'timestamp': time.time()
            }
    
    def clear_cache(self, pattern: str = None) -> bool:
        """Clear cache entries."""
        try:
            client = self.redis._get_client()
            
            if pattern:
                # Clear specific pattern
                keys = client.keys(pattern)
                if keys:
                    result = client.delete(*keys)
                    logger.info(f"Cleared {result} cache keys matching pattern: {pattern}")
                    return True
            else:
                # Clear all crawler cache
                phash_keys = client.keys(self.config.get_cache_key('phash', phash='*'))
                site_stats_keys = client.keys(self.config.get_cache_key('site_stats', site_id='*'))
                all_keys = phash_keys + site_stats_keys
                
                if all_keys:
                    result = client.delete(*all_keys)
                    logger.info(f"Cleared {result} cache keys")
                    return True
            
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False
    
    def cleanup_expired_cache(self) -> int:
        """Cleanup expired cache entries (Redis handles TTL automatically)."""
        # Redis automatically handles TTL, so this is mainly for logging
        try:
            stats = self.get_cache_stats()
            logger.info(f"Cache cleanup completed. Current cache size: {stats['total_cache_keys']}")
            return stats['total_cache_keys']
        except Exception as e:
            logger.error(f"Failed to cleanup cache: {e}")
            return 0
    
    def health_check(self) -> Dict[str, Any]:
        """Perform cache health check."""
        try:
            # Test basic operations
            test_phash = "test_hash_12345"
            test_data = {'test': True, 'timestamp': time.time()}
            
            # Test set/get
            cache_key = self.get_phash_cache_key(test_phash)
            set_success = self.redis.set_cache(cache_key, test_data, ttl_seconds=60)
            get_data = self.redis.get_cache(cache_key)
            delete_success = self.redis.delete_cache(cache_key)
            
            # Get cache stats
            stats = self.get_cache_stats()
            
            return {
                'status': 'healthy' if set_success and get_data and delete_success else 'unhealthy',
                'set_test': set_success,
                'get_test': get_data is not None,
                'delete_test': delete_success,
                'cache_stats': stats,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Cache health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': time.time()
            }


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager() -> CacheManager:
    """Get global cache manager instance."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager


def close_cache_manager():
    """Close global cache manager."""
    global _cache_manager
    if _cache_manager:
        _cache_manager = None



