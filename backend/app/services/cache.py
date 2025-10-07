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

"""
Crawl cache service for avoiding reprocessing of already-stored images.
Phase 1: Basic URL caching with minimal fields.
"""

import hashlib
import logging
import os
from typing import Optional, Tuple, List, Dict, Any
from psycopg import connect, Connection
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class CrawlCacheDB:
    """Simple cache service for crawler state persistence."""
    
    def __init__(self):
        self.connection_string = self._build_connection_string()
        self.cache_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        
    def _build_connection_string(self) -> str:
        """Build PostgreSQL connection string from environment variables."""
        host = os.getenv('POSTGRES_HOST', 'localhost')
        port = os.getenv('POSTGRES_PORT', '5432')
        db = os.getenv('POSTGRES_DB', 'mordeaux')
        user = os.getenv('POSTGRES_USER', 'mordeaux')
        password = os.getenv('POSTGRES_PASSWORD', '')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
    
    def _get_connection(self) -> Connection:
        """Get database connection."""
        try:
            return connect(self.connection_string, row_factory=dict_row)
        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}")
            raise
    
    def should_skip_image(self, url: str, image_bytes: bytes, similarity_threshold: int = 5) -> Tuple[bool, Optional[str]]:
        """
        Check if image should be skipped based on cache with tolerant duplicate detection.
        
        Args:
            url: Image URL
            image_bytes: Image content bytes
            similarity_threshold: Hamming distance threshold for similarity (default: 5)
            
        Returns:
            Tuple of (should_skip, existing_raw_key)
        """
        if not self.cache_enabled:
            return False, None
            
        try:
            # Stage 1: URL hash lookup (fastest - single query)
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT raw_image_key FROM crawl_cache WHERE url_hash = %s",
                        (url_hash,)
                    )
                    result = cur.fetchone()
                    
                    if result:
                        logger.info(f"Cache hit (URL): {url[:100]}...")
                        return True, result['raw_image_key']
                    
                    # Stage 2: Enhanced content hash lookup with tolerance (only if Stage 1 fails)
                    from .face import compute_tolerant_phash, compute_phash_similarity
                    new_hashes = compute_tolerant_phash(image_bytes)
                    
                    if any(new_hashes):  # If we have at least one hash
                        # Optimized query with indexed lookup for each hash type
                        phash, dhash, whash, ahash = new_hashes
                        
                        # Check phash first (most reliable)
                        if phash:
                            cur.execute("""
                                SELECT raw_image_key, phash, dhash, whash, ahash 
                                FROM crawl_cache 
                                WHERE phash = %s
                            """, (phash,))
                            row = cur.fetchone()
                            if row:
                                logger.info(f"Cache hit (exact phash): {url[:100]}...")
                                return True, row['raw_image_key']
                        
                        # Check dhash if phash didn't match
                        if dhash:
                            cur.execute("""
                                SELECT raw_image_key, phash, dhash, whash, ahash 
                                FROM crawl_cache 
                                WHERE dhash = %s
                            """, (dhash,))
                            row = cur.fetchone()
                            if row:
                                logger.info(f"Cache hit (exact dhash): {url[:100]}...")
                                return True, row['raw_image_key']
                        
                        # Fallback to similarity check for whash and ahash (less reliable)
                        if whash or ahash:
                            cur.execute("""
                                SELECT raw_image_key, phash, dhash, whash, ahash 
                                FROM crawl_cache 
                                WHERE (whash IS NOT NULL AND whash != '') OR (ahash IS NOT NULL AND ahash != '')
                                LIMIT 1000
                            """)
                            
                            for row in cur.fetchall():
                                existing_hashes = (row['phash'] or "", row['dhash'] or "", row['whash'] or "", row['ahash'] or "")
                                
                                if compute_phash_similarity(new_hashes, existing_hashes, similarity_threshold):
                                    logger.info(f"Cache hit (content similarity): {url[:100]}... (threshold: {similarity_threshold})")
                                    return True, row['raw_image_key']
                    
                    return False, None
                    
        except Exception as e:
            logger.warning(f"Cache lookup failed for {url[:100]}...: {str(e)}")
            return False, None
    
    def store_processed_image(self, url: str, image_bytes: bytes, 
                            raw_key: str, thumbnail_key: Optional[str] = None,
                            face_data: Optional[Dict] = None) -> bool:
        """
        Store processed image in cache.
        
        Args:
            url: Image URL
            image_bytes: Image content bytes
            raw_key: MinIO key for raw image
            thumbnail_key: MinIO key for thumbnail (if exists)
            face_data: Face detection results (if any)
            
        Returns:
            True if stored successfully, False otherwise
        """
        if not self.cache_enabled:
            return False
            
        try:
            url_hash = hashlib.sha256(url.encode()).hexdigest()
            
            # Compute multiple hash types for tolerant duplicate detection
            from .face import compute_tolerant_phash
            phash, dhash, whash, ahash = compute_tolerant_phash(image_bytes)
            face_detected = thumbnail_key is not None
            
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO crawl_cache 
                        (url_hash, phash, dhash, whash, ahash, raw_image_key, thumbnail_key, face_detected)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url_hash) DO UPDATE SET
                            phash = EXCLUDED.phash,
                            dhash = EXCLUDED.dhash,
                            whash = EXCLUDED.whash,
                            ahash = EXCLUDED.ahash,
                            raw_image_key = EXCLUDED.raw_image_key,
                            thumbnail_key = EXCLUDED.thumbnail_key,
                            face_detected = EXCLUDED.face_detected,
                            processed_at = CURRENT_TIMESTAMP
                    """, (url_hash, phash, dhash, whash, ahash, raw_key, thumbnail_key, face_detected))
                    
                    conn.commit()
                    logger.debug(f"Cached image: {url[:100]}... -> {raw_key}")
                    return True
                    
        except Exception as e:
            logger.warning(f"Failed to cache image {url[:100]}...: {str(e)}")
            return False
    
    def batch_store_processed_images(self, entries: List[Dict[str, Any]]) -> int:
        """
        Batch store multiple processed images in cache.
        
        Args:
            entries: List of cache entry dictionaries
            
        Returns:
            Number of successfully stored entries
        """
        if not self.cache_enabled or not entries:
            return 0
            
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    # Prepare batch data
                    batch_data = []
                    for entry in entries:
                        batch_data.append((
                            entry['url_hash'],
                            entry['phash'],
                            entry.get('dhash', ''),
                            entry.get('whash', ''),
                            entry.get('ahash', ''),
                            entry['raw_image_key'],
                            entry['thumbnail_key'],
                            entry['face_detected']
                        ))
                    
                    # Execute batch insert
                    cur.executemany("""
                        INSERT INTO crawl_cache 
                        (url_hash, phash, dhash, whash, ahash, raw_image_key, thumbnail_key, face_detected)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (url_hash) DO UPDATE SET
                            phash = EXCLUDED.phash,
                            dhash = EXCLUDED.dhash,
                            whash = EXCLUDED.whash,
                            ahash = EXCLUDED.ahash,
                            raw_image_key = EXCLUDED.raw_image_key,
                            thumbnail_key = EXCLUDED.thumbnail_key,
                            face_detected = EXCLUDED.face_detected,
                            processed_at = CURRENT_TIMESTAMP
                    """, batch_data)
                    
                    conn.commit()
                    logger.info(f"Batch cached {len(entries)} images")
                    return len(entries)
                    
        except Exception as e:
            logger.warning(f"Failed to batch cache {len(entries)} images: {str(e)}")
            return 0
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get basic cache statistics."""
        if not self.cache_enabled:
            return {'enabled': False}
            
        try:
            with self._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) as total FROM crawl_cache")
                    total = cur.fetchone()['total']
                    
                    cur.execute("SELECT COUNT(*) as faces FROM crawl_cache WHERE face_detected = true")
                    faces = cur.fetchone()['faces']
                    
                    return {
                        'enabled': True,
                        'total_cached': total,
                        'faces_found': faces,
                        'cache_hit_rate': 'N/A'  # Will be calculated by crawler
                    }
                    
        except Exception as e:
            logger.warning(f"Failed to get cache stats: {str(e)}")
            return {'enabled': False, 'error': str(e)}
    
    def _compute_phash(self, image_bytes: bytes) -> str:
        """Compute perceptual hash for image content."""
        try:
            from .face import compute_phash
            return compute_phash(image_bytes)
        except Exception as e:
            logger.warning(f"Failed to compute phash: {str(e)}")
            return ""


def get_crawl_cache_service() -> CrawlCacheDB:
    """Get crawl cache service instance."""
    return CrawlCacheDB()
