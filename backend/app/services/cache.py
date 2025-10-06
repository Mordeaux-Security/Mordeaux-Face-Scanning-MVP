import json
import hashlib
import time
from typing import Any, Optional, Dict, List
import logging
import redis
from ..core.config import get_settings

logger = logging.getLogger(__name__)

class CacheService:
    """Redis-based caching service for face embeddings and search results."""
    
    def __init__(self):
        self.settings = get_settings()
        self.redis_client = redis.from_url(self.settings.redis_url, decode_responses=True)
        
        # Cache TTL settings (in seconds)
        self.embedding_cache_ttl = 3600  # 1 hour for embeddings
        self.search_cache_ttl = 300      # 5 minutes for search results
        self.phash_cache_ttl = 7200      # 2 hours for perceptual hashes
        
        # Cache key prefixes
        self.embedding_prefix = "embedding:"
        self.search_prefix = "search:"
        self.phash_prefix = "phash:"
        self.face_detection_prefix = "face_detection:"
    
    def _generate_cache_key(self, prefix: str, content: bytes, tenant_id: str, **kwargs) -> str:
        """Generate a cache key based on content hash and tenant."""
        # Create a hash of the content
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        
        # Include additional parameters in the key
        key_parts = [prefix, tenant_id, content_hash]
        if kwargs:
            for key, value in sorted(kwargs.items()):
                key_parts.append(f"{key}:{value}")
        
        return ":".join(key_parts)
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for Redis storage."""
        try:
            return json.dumps(data, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize data for caching: {e}")
            return None
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from Redis storage."""
        try:
            return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize cached data: {e}")
            return None
    
    async def cache_face_embeddings(self, content: bytes, tenant_id: str, embeddings: List[Dict]) -> bool:
        """Cache face embeddings for an image."""
        try:
            cache_key = self._generate_cache_key(self.embedding_prefix, content, tenant_id)
            serialized_data = self._serialize_data(embeddings)
            
            if serialized_data:
                self.redis_client.setex(cache_key, self.embedding_cache_ttl, serialized_data)
                logger.debug(f"Cached face embeddings for tenant {tenant_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to cache face embeddings: {e}")
        return False
    
    async def get_cached_face_embeddings(self, content: bytes, tenant_id: str) -> Optional[List[Dict]]:
        """Get cached face embeddings for an image."""
        try:
            cache_key = self._generate_cache_key(self.embedding_prefix, content, tenant_id)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                embeddings = self._deserialize_data(cached_data)
                logger.debug(f"Retrieved cached face embeddings for tenant {tenant_id}")
                return embeddings
        except Exception as e:
            logger.error(f"Failed to retrieve cached face embeddings: {e}")
        return None
    
    async def cache_perceptual_hash(self, content: bytes, tenant_id: str, phash: str) -> bool:
        """Cache perceptual hash for an image."""
        try:
            cache_key = self._generate_cache_key(self.phash_prefix, content, tenant_id)
            self.redis_client.setex(cache_key, self.phash_cache_ttl, phash)
            logger.debug(f"Cached perceptual hash for tenant {tenant_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache perceptual hash: {e}")
        return False
    
    async def get_cached_perceptual_hash(self, content: bytes, tenant_id: str) -> Optional[str]:
        """Get cached perceptual hash for an image."""
        try:
            cache_key = self._generate_cache_key(self.phash_prefix, content, tenant_id)
            cached_phash = self.redis_client.get(cache_key)
            
            if cached_phash:
                logger.debug(f"Retrieved cached perceptual hash for tenant {tenant_id}")
                return cached_phash
        except Exception as e:
            logger.error(f"Failed to retrieve cached perceptual hash: {e}")
        return None
    
    async def cache_search_results(self, content: bytes, tenant_id: str, topk: int, results: List[Dict]) -> bool:
        """Cache search results for an image query."""
        try:
            cache_key = self._generate_cache_key(self.search_prefix, content, tenant_id, topk=topk)
            serialized_data = self._serialize_data(results)
            
            if serialized_data:
                self.redis_client.setex(cache_key, self.search_cache_ttl, serialized_data)
                logger.debug(f"Cached search results for tenant {tenant_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to cache search results: {e}")
        return False
    
    async def get_cached_search_results(self, content: bytes, tenant_id: str, topk: int) -> Optional[List[Dict]]:
        """Get cached search results for an image query."""
        try:
            cache_key = self._generate_cache_key(self.search_prefix, content, tenant_id, topk=topk)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                results = self._deserialize_data(cached_data)
                logger.debug(f"Retrieved cached search results for tenant {tenant_id}")
                return results
        except Exception as e:
            logger.error(f"Failed to retrieve cached search results: {e}")
        return None
    
    async def cache_face_detection(self, content: bytes, tenant_id: str, faces: List[Dict]) -> bool:
        """Cache face detection results."""
        try:
            cache_key = self._generate_cache_key(self.face_detection_prefix, content, tenant_id)
            serialized_data = self._serialize_data(faces)
            
            if serialized_data:
                self.redis_client.setex(cache_key, self.embedding_cache_ttl, serialized_data)
                logger.debug(f"Cached face detection results for tenant {tenant_id}")
                return True
        except Exception as e:
            logger.error(f"Failed to cache face detection results: {e}")
        return False
    
    async def get_cached_face_detection(self, content: bytes, tenant_id: str) -> Optional[List[Dict]]:
        """Get cached face detection results."""
        try:
            cache_key = self._generate_cache_key(self.face_detection_prefix, content, tenant_id)
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                faces = self._deserialize_data(cached_data)
                logger.debug(f"Retrieved cached face detection results for tenant {tenant_id}")
                return faces
        except Exception as e:
            logger.error(f"Failed to retrieve cached face detection results: {e}")
        return None
    
    async def invalidate_tenant_cache(self, tenant_id: str) -> int:
        """Invalidate all cached data for a specific tenant."""
        try:
            # Get all keys matching the tenant pattern
            pattern = f"*:{tenant_id}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                deleted_count = self.redis_client.delete(*keys)
                logger.info(f"Invalidated {deleted_count} cache entries for tenant {tenant_id}")
                return deleted_count
        except Exception as e:
            logger.error(f"Failed to invalidate cache for tenant {tenant_id}: {e}")
        return 0
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            info = self.redis_client.info()
            
            # Count keys by prefix
            embedding_keys = len(self.redis_client.keys(f"{self.embedding_prefix}*"))
            search_keys = len(self.redis_client.keys(f"{self.search_prefix}*"))
            phash_keys = len(self.redis_client.keys(f"{self.phash_prefix}*"))
            face_detection_keys = len(self.redis_client.keys(f"{self.face_detection_prefix}*"))
            
            return {
                "redis_info": {
                    "used_memory_human": info.get("used_memory_human"),
                    "connected_clients": info.get("connected_clients"),
                    "total_commands_processed": info.get("total_commands_processed"),
                    "keyspace_hits": info.get("keyspace_hits"),
                    "keyspace_misses": info.get("keyspace_misses")
                },
                "cache_counts": {
                    "embedding_cache": embedding_keys,
                    "search_cache": search_keys,
                    "phash_cache": phash_keys,
                    "face_detection_cache": face_detection_keys,
                    "total_cached_items": embedding_keys + search_keys + phash_keys + face_detection_keys
                },
                "cache_ttl_settings": {
                    "embedding_cache_ttl": self.embedding_cache_ttl,
                    "search_cache_ttl": self.search_cache_ttl,
                    "phash_cache_ttl": self.phash_cache_ttl
                }
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}
    
    async def clear_all_cache(self) -> bool:
        """Clear all cache data (use with caution)."""
        try:
            self.redis_client.flushdb()
            logger.warning("Cleared all cache data")
            return True
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
            return False

# Global cache service instance
_cache_service = None

def get_cache_service() -> CacheService:
    """Get cache service instance."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
