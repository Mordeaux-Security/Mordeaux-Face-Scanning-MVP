"""
Hybrid Cache Service V2

Combines Redis (fast, volatile) and PostgreSQL (persistent, reliable) caching.
- Redis: Primary cache for ultra-fast lookups and hot data
- PostgreSQL: Secondary cache for persistent storage and cold data

This provides the best of both worlds:
- Speed of Redis for frequently accessed data
- Persistence of PostgreSQL for reliability across restarts
- Cost efficiency for large datasets
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import redis
from psycopg import connect, Connection
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class HybridCacheService:
    """
    Hybrid caching service that combines Redis and PostgreSQL.
    
    Architecture:
    1. Redis (Primary): Fast, volatile cache for hot data
    2. PostgreSQL (Secondary): Persistent cache for reliable storage
    3. Automatic backfill: PostgreSQL data → Redis when accessed
    4. Graceful degradation: Works even if Redis is unavailable
    """
    
    def __init__(self, redis_url: Optional[str] = None, postgres_config: Optional[Dict] = None):
        """
        Initialize hybrid cache service.
        
        Args:
            redis_url: Redis connection URL (optional, will use env vars if not provided)
            postgres_config: PostgreSQL configuration dict (optional, will use env vars if not provided)
        """
        self.redis_enabled = True
        self.postgres_enabled = True
        
        # Cache hit tracking
        self.redis_hits = 0
        self.postgres_hits = 0
        self.cache_misses = 0
        
        # Initialize Redis
        try:
            if redis_url:
                self.redis_client = redis.from_url(redis_url, decode_responses=True)
            else:
                # Try to get Redis URL from environment
                redis_url_env = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
                self.redis_client = redis.from_url(redis_url_env, decode_responses=True)
            
            # Test Redis connection
            self.redis_client.ping()
            logger.info("Redis cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"Redis cache unavailable: {e}. Falling back to PostgreSQL only.")
            self.redis_enabled = False
            self.redis_client = None
        
        # Initialize PostgreSQL
        try:
            if postgres_config:
                self.postgres_config = postgres_config
            else:
                # Build from environment variables
                self.postgres_config = {
                    'host': os.getenv('POSTGRES_HOST', 'localhost'),
                    'port': os.getenv('POSTGRES_PORT', '5432'),
                    'db': os.getenv('POSTGRES_DB', 'mordeaux'),
                    'user': os.getenv('POSTGRES_USER', 'mordeaux'),
                    'password': os.getenv('POSTGRES_PASSWORD', '')
                }
            
            # Test PostgreSQL connection
            with self._get_postgres_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
            
            logger.info("PostgreSQL cache initialized successfully")
            
        except Exception as e:
            logger.warning(f"PostgreSQL cache unavailable: {e}. Using Redis only.")
            self.postgres_enabled = False
        
        # Cache TTL settings (Redis only)
        self.redis_ttl_settings = {
            'embedding_cache': 3600,      # 1 hour
            'search_cache': 300,          # 5 minutes  
            'phash_cache': 7200,          # 2 hours
            'face_detection_cache': 3600, # 1 hour
            'crawl_cache': 86400,         # 24 hours
        }
        
        # Thread pool for blocking operations
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="hybrid_cache")
    
    def _get_postgres_connection(self) -> Connection:
        """Get PostgreSQL connection."""
        if not self.postgres_enabled:
            raise RuntimeError("PostgreSQL cache is disabled")
        
        connection_string = (
            f"postgresql://{self.postgres_config['user']}:{self.postgres_config['password']}"
            f"@{self.postgres_config['host']}:{self.postgres_config['port']}"
            f"/{self.postgres_config['db']}"
        )
        
        try:
            return connect(connection_string, row_factory=dict_row)
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise
    
    def _generate_cache_key(self, prefix: str, content: bytes, tenant_id: str = "default", **kwargs) -> str:
        """Generate a cache key based on content hash and parameters."""
        content_hash = hashlib.sha256(content).hexdigest()[:16]
        key_parts = [prefix, tenant_id, content_hash]
        
        if kwargs:
            for key, value in sorted(kwargs.items()):
                key_parts.append(f"{key}:{value}")
        
        return ":".join(key_parts)
    
    def _serialize_data(self, data: Any) -> str:
        """Serialize data for storage."""
        try:
            return json.dumps(data, default=str)
        except Exception as e:
            logger.error(f"Failed to serialize data: {e}")
            return None
    
    def _deserialize_data(self, data: str) -> Any:
        """Deserialize data from storage."""
        try:
            return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to deserialize data: {e}")
            return None
    
    # ==================== FACE EMBEDDINGS CACHE ====================
    
    async def cache_face_embeddings(self, content: bytes, tenant_id: str, embeddings: List[Dict]) -> bool:
        """Cache face embeddings in both Redis and PostgreSQL."""
        cache_key = self._generate_cache_key("embedding", content, tenant_id)
        serialized_data = self._serialize_data(embeddings)
        
        if not serialized_data:
            return False
        
        success = True
        
        # Store in Redis (primary cache)
        if self.redis_enabled:
            try:
                self.redis_client.setex(
                    cache_key, 
                    self.redis_ttl_settings['embedding_cache'], 
                    serialized_data
                )
                logger.debug(f"Cached face embeddings in Redis for tenant {tenant_id}")
            except Exception as e:
                logger.warning(f"Failed to cache face embeddings in Redis: {e}")
                success = False
        
        # Store in PostgreSQL (secondary cache)
        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._thread_pool,
                    self._store_embeddings_postgres,
                    cache_key, content, tenant_id, embeddings
                )
                logger.debug(f"Cached face embeddings in PostgreSQL for tenant {tenant_id}")
            except Exception as e:
                logger.warning(f"Failed to cache face embeddings in PostgreSQL: {e}")
                success = False
        
        return success
    
    def _store_embeddings_postgres(self, cache_key: str, content: bytes, tenant_id: str, embeddings: List[Dict]):
        """Store face embeddings in PostgreSQL."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO face_embeddings_cache 
                    (cache_key, tenant_id, content_hash, embeddings_data, created_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        embeddings_data = EXCLUDED.embeddings_data,
                        created_at = CURRENT_TIMESTAMP
                """, (
                    cache_key,
                    tenant_id,
                    hashlib.sha256(content).hexdigest(),
                    json.dumps(embeddings)
                ))
                conn.commit()
    
    async def get_cached_face_embeddings(self, content: bytes, tenant_id: str) -> Optional[List[Dict]]:
        """Get cached face embeddings from Redis (primary) or PostgreSQL (secondary)."""
        cache_key = self._generate_cache_key("embedding", content, tenant_id)
        
        # Try Redis first (primary cache)
        if self.redis_enabled:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    embeddings = self._deserialize_data(cached_data)
                    if embeddings:
                        logger.debug(f"Retrieved face embeddings from Redis for tenant {tenant_id}")
                        return embeddings
            except Exception as e:
                logger.warning(f"Redis lookup failed for face embeddings: {e}")
        
        # Fall back to PostgreSQL (secondary cache)
        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool,
                    self._get_embeddings_postgres,
                    cache_key
                )
                
                if result:
                    # Backfill Redis for future fast access
                    if self.redis_enabled:
                        try:
                            serialized_data = self._serialize_data(result)
                            if serialized_data:
                                self.redis_client.setex(
                                    cache_key,
                                    self.redis_ttl_settings['embedding_cache'],
                                    serialized_data
                                )
                        except Exception as e:
                            logger.warning(f"Failed to backfill Redis with embeddings: {e}")
                    
                    logger.debug(f"Retrieved face embeddings from PostgreSQL for tenant {tenant_id}")
                    return result
                    
            except Exception as e:
                logger.warning(f"PostgreSQL lookup failed for face embeddings: {e}")
        
        return None

    # ==================== FACE DETECTION CACHE ====================
    async def cache_face_detection(self, content: bytes, tenant_id: str, faces: List[Dict]) -> bool:
        """Cache face detection results (bboxes, embeddings, scores)."""
        cache_key = self._generate_cache_key("face_detect", content, tenant_id)
        serialized_data = self._serialize_data(faces)
        if not serialized_data:
            return False

        success = True
        if self.redis_enabled:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.redis_ttl_settings['face_detection_cache'],
                    serialized_data
                )
            except Exception as e:
                logger.warning(f"Failed to cache face detection in Redis: {e}")
                success = False

        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._thread_pool,
                    self._store_face_detection_postgres,
                    cache_key, content, tenant_id, faces
                )
            except Exception as e:
                logger.warning(f"Failed to cache face detection in PostgreSQL: {e}")
                success = False
        return success

    def _store_face_detection_postgres(self, cache_key: str, content: bytes, tenant_id: str, faces: List[Dict]):
        """Store face detection results in PostgreSQL."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO face_embeddings_cache 
                    (cache_key, tenant_id, content_hash, embeddings_data, created_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        embeddings_data = EXCLUDED.embeddings_data,
                        created_at = CURRENT_TIMESTAMP
                """, (
                    cache_key,
                    tenant_id,
                    hashlib.sha256(content).hexdigest(),
                    json.dumps(faces)
                ))
                conn.commit()

    async def get_cached_face_detection(self, content: bytes, tenant_id: str) -> Optional[List[Dict]]:
        """Get cached face detection results from Redis or PostgreSQL."""
        cache_key = self._generate_cache_key("face_detect", content, tenant_id)

        if self.redis_enabled:
            try:
                cached = self.redis_client.get(cache_key)
                if cached:
                    data = self._deserialize_data(cached)
                    if data is not None:
                        return data
            except Exception as e:
                logger.warning(f"Redis lookup failed for face detection: {e}")

        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool,
                    self._get_face_detection_postgres,
                    cache_key
                )
                if result is not None:
                    if self.redis_enabled:
                        try:
                            serialized = self._serialize_data(result)
                            if serialized:
                                self.redis_client.setex(
                                    cache_key,
                                    self.redis_ttl_settings['face_detection_cache'],
                                    serialized
                                )
                        except Exception as e:
                            logger.warning(f"Failed to backfill Redis face detection: {e}")
                    return result
            except Exception as e:
                logger.warning(f"PostgreSQL lookup failed for face detection: {e}")
        return None

    def _get_face_detection_postgres(self, cache_key: str) -> Optional[List[Dict]]:
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT embeddings_data FROM face_embeddings_cache 
                    WHERE cache_key = %s
                """, (cache_key,))
                row = cur.fetchone()
                if row:
                    try:
                        return json.loads(row['embeddings_data'])
                    except Exception:
                        return None
        return None
    
    def _get_embeddings_postgres(self, cache_key: str) -> Optional[List[Dict]]:
        """Get face embeddings from PostgreSQL."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT embeddings_data FROM face_embeddings_cache 
                    WHERE cache_key = %s
                """, (cache_key,))
                
                result = cur.fetchone()
                if result:
                    return json.loads(result['embeddings_data'])
        
        return None
    
    # ==================== PERCEPTUAL HASH CACHE ====================
    
    async def cache_perceptual_hash(self, content: bytes, tenant_id: str, phash: str) -> bool:
        """Cache perceptual hash in both Redis and PostgreSQL."""
        cache_key = self._generate_cache_key("phash", content, tenant_id)
        
        success = True
        
        # Store in Redis (primary cache)
        if self.redis_enabled:
            try:
                self.redis_client.setex(
                    cache_key,
                    self.redis_ttl_settings['phash_cache'],
                    phash
                )
                logger.debug(f"Cached perceptual hash in Redis for tenant {tenant_id}")
            except Exception as e:
                logger.warning(f"Failed to cache perceptual hash in Redis: {e}")
                success = False
        
        # Store in PostgreSQL (secondary cache)
        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._thread_pool,
                    self._store_phash_postgres,
                    cache_key, content, tenant_id, phash
                )
                logger.debug(f"Cached perceptual hash in PostgreSQL for tenant {tenant_id}")
            except Exception as e:
                logger.warning(f"Failed to cache perceptual hash in PostgreSQL: {e}")
                success = False
        
        return success
    
    def _store_phash_postgres(self, cache_key: str, content: bytes, tenant_id: str, phash: str):
        """Store perceptual hash in PostgreSQL."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO perceptual_hash_cache 
                    (cache_key, tenant_id, content_hash, phash, created_at)
                    VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        phash = EXCLUDED.phash,
                        created_at = CURRENT_TIMESTAMP
                """, (
                    cache_key,
                    tenant_id,
                    hashlib.sha256(content).hexdigest(),
                    phash
                ))
                conn.commit()
    
    async def get_cached_perceptual_hash(self, content: bytes, tenant_id: str) -> Optional[str]:
        """Get cached perceptual hash from Redis (primary) or PostgreSQL (secondary)."""
        cache_key = self._generate_cache_key("phash", content, tenant_id)
        
        # Try Redis first (primary cache)
        if self.redis_enabled:
            try:
                cached_phash = self.redis_client.get(cache_key)
                if cached_phash:
                    logger.debug(f"Retrieved perceptual hash from Redis for tenant {tenant_id}")
                    return cached_phash
            except Exception as e:
                logger.warning(f"Redis lookup failed for perceptual hash: {e}")
        
        # Fall back to PostgreSQL (secondary cache)
        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool,
                    self._get_phash_postgres,
                    cache_key
                )
                
                if result:
                    # Backfill Redis for future fast access
                    if self.redis_enabled:
                        try:
                            self.redis_client.setex(
                                cache_key,
                                self.redis_ttl_settings['phash_cache'],
                                result
                            )
                        except Exception as e:
                            logger.warning(f"Failed to backfill Redis with phash: {e}")
                    
                    logger.debug(f"Retrieved perceptual hash from PostgreSQL for tenant {tenant_id}")
                    return result
                    
            except Exception as e:
                logger.warning(f"PostgreSQL lookup failed for perceptual hash: {e}")
        
        return None
    
    def _get_phash_postgres(self, cache_key: str) -> Optional[str]:
        """Get perceptual hash from PostgreSQL."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT phash FROM perceptual_hash_cache 
                    WHERE cache_key = %s
                """, (cache_key,))
                
                result = cur.fetchone()
                if result:
                    return result['phash']
        
        return None
    
    async def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache hit/miss statistics.
        
        Returns:
            Dict with cache statistics
        """
        total_hits = self.redis_hits + self.postgres_hits
        total_requests = total_hits + self.cache_misses
        
        return {
            'redis_hits': self.redis_hits,
            'postgres_hits': self.postgres_hits,
            'cache_misses': self.cache_misses,
            'total_hits': total_hits,
            'total_requests': total_requests,
            'hit_rate': (total_hits / total_requests * 100) if total_requests > 0 else 0,
            'redis_hit_rate': (self.redis_hits / total_requests * 100) if total_requests > 0 else 0,
            'postgres_hit_rate': (self.postgres_hits / total_requests * 100) if total_requests > 0 else 0
        }
    
    def reset_cache_stats(self):
        """Reset cache hit/miss counters."""
        self.redis_hits = 0
        self.postgres_hits = 0
        self.cache_misses = 0
    
    # ==================== CRAWL CACHE (DUPLICATE PREVENTION) ====================
    
    async def should_skip_crawled_image(self, url: str, image_bytes: bytes, tenant_id: str = "default") -> Tuple[bool, Optional[str]]:
        """
        Check if image should be skipped based on crawl cache.
        This is the main duplicate prevention for crawlers.
        """
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        
        # Try Redis first (primary cache)
        if self.redis_enabled:
            try:
                redis_key = f"crawl:{tenant_id}:{url_hash}"
                cached_result = self.redis_client.get(redis_key)
                if cached_result:
                    result = json.loads(cached_result)
                    self.redis_hits += 1
                    logger.debug(f"Cache hit (Redis): {url[:100]}...")
                    return result.get('should_skip', False), result.get('raw_key')
            except Exception as e:
                logger.warning(f"Redis crawl lookup failed: {e}")
        
        # Fall back to PostgreSQL (secondary cache)
        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self._thread_pool,
                    self._check_crawl_cache_postgres,
                    url_hash, image_bytes
                )
                
                if result:
                    should_skip, raw_key = result
                    self.postgres_hits += 1
                    
                    # Backfill Redis for future fast access
                    if self.redis_enabled:
                        try:
                            redis_key = f"crawl:{tenant_id}:{url_hash}"
                            cache_data = json.dumps({
                                'should_skip': should_skip,
                                'raw_key': raw_key,
                                'cached_at': time.time()
                            })
                            self.redis_client.setex(redis_key, self.redis_ttl_settings['crawl_cache'], cache_data)
                        except Exception as e:
                            logger.warning(f"Failed to backfill Redis with crawl data: {e}")
                    
                    logger.debug(f"Cache hit (PostgreSQL): {url[:100]}...")
                    return should_skip, raw_key
                    
            except Exception as e:
                logger.warning(f"PostgreSQL crawl lookup failed: {e}")
        
        # Cache miss - increment counter
        self.cache_misses += 1
        return False, None
    
    def _check_crawl_cache_postgres(self, url_hash: str, image_bytes: bytes) -> Optional[Tuple[bool, Optional[str]]]:
        """Check crawl cache in PostgreSQL."""
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                # Check by URL hash first
                cur.execute("""
                    SELECT raw_image_key FROM crawl_cache 
                    WHERE url_hash = %s
                """, (url_hash,))
                
                result = cur.fetchone()
                if result:
                    return True, result['raw_image_key']
                
                # Check by content hash for exact duplicates
                cur.execute("""
                    SELECT raw_image_key FROM crawl_cache 
                    WHERE content_hash = %s
                """, (content_hash,))
                
                result = cur.fetchone()
                if result:
                    return True, result['raw_image_key']
        
        return None
    
    async def should_skip_crawled_image_by_phash(self, image_bytes: bytes, phash: str, tenant_id: str = "default", similarity_threshold: int = 5) -> Tuple[bool, Optional[str]]:
        """
        Check if image should be skipped based on perceptual hash similarity.
        This catches visually identical images with different metadata/compression.
        """
        if not self.postgres_enabled:
            return False, None
            
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self._thread_pool,
                self._check_phash_similarity_postgres,
                phash, similarity_threshold
            )
            
            if result:
                should_skip, raw_key = result
                logger.debug(f"Perceptual hash similarity match found (threshold: {similarity_threshold})")
                return should_skip, raw_key
                
        except Exception as e:
            logger.warning(f"PostgreSQL perceptual hash lookup failed: {e}")
        
        return False, None
    
    def _check_phash_similarity_postgres(self, phash: str, similarity_threshold: int) -> Optional[Tuple[bool, Optional[str]]]:
        """Check for perceptual hash similarity in PostgreSQL."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                # Get all stored perceptual hashes and compare using Hamming distance
                cur.execute("""
                    SELECT cc.raw_image_key, ph.phash 
                    FROM crawl_cache cc
                    JOIN perceptual_hash_cache ph ON cc.content_hash = ph.content_hash
                    WHERE ph.phash IS NOT NULL
                """)
                
                results = cur.fetchall()
                
                # Convert input phash to integer for comparison
                try:
                    input_phash_int = int(phash, 16)
                except ValueError:
                    logger.warning(f"Invalid perceptual hash format: {phash}")
                    return None
                
                for result in results:
                    raw_key, stored_phash = result
                    
                    try:
                        stored_phash_int = int(stored_phash, 16)
                        # Calculate Hamming distance (XOR and count bits)
                        hamming_distance = bin(input_phash_int ^ stored_phash_int).count('1')
                        
                        if hamming_distance <= similarity_threshold:
                            logger.debug(f"Perceptual hash similarity found: distance={hamming_distance} <= threshold={similarity_threshold}")
                            return True, raw_key
                            
                    except ValueError:
                        continue  # Skip invalid hash formats
        
        return None
    
    async def store_crawled_image(self, url: str, image_bytes: bytes, raw_key: str, 
                                thumbnail_key: Optional[str] = None, tenant_id: str = "default", source_url: Optional[str] = None) -> bool:
        """Store crawled image metadata (not image bytes) in both Redis and PostgreSQL."""
        url_hash = hashlib.sha256(url.encode()).hexdigest()
        content_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Extract metadata only (no image bytes)
        metadata = {
            'hash': content_hash,
            'length': len(image_bytes),
            'mime': 'image/jpeg',
            'raw_key': raw_key,
            'thumb_key': thumbnail_key
        }
        
        # Add source URL if provided
        if source_url:
            metadata['source_url'] = source_url
        
        success = True
        
        # Store in Redis (primary cache)
        if self.redis_enabled:
            try:
                redis_key = f"crawl:{tenant_id}:{url_hash}"
                cache_data = json.dumps({
                    'should_skip': True,
                    'raw_key': raw_key,
                    'thumbnail_key': thumbnail_key,
                    'metadata': metadata,
                    'cached_at': time.time()
                })
                self.redis_client.setex(redis_key, self.redis_ttl_settings['crawl_cache'], cache_data)
                logger.debug(f"Cached crawled image metadata in Redis: {url[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to cache crawled image metadata in Redis: {e}")
                success = False
        
        # Store in PostgreSQL (secondary cache)
        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._thread_pool,
                    self._store_crawl_cache_postgres,
                    url_hash, content_hash, raw_key, thumbnail_key, metadata
                )
                logger.debug(f"Cached crawled image metadata in PostgreSQL: {url[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to cache crawled image metadata in PostgreSQL: {e}")
                success = False
        
        return success
    
    def _store_crawl_cache_postgres(self, url_hash: str, content_hash: str, raw_key: str, thumbnail_key: Optional[str], metadata: Dict):
        """Store crawl cache with metadata in PostgreSQL."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO crawl_cache 
                    (url_hash, content_hash, raw_image_key, thumbnail_key, metadata, processed_at)
                    VALUES (%s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (url_hash) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        raw_image_key = EXCLUDED.raw_image_key,
                        thumbnail_key = EXCLUDED.thumbnail_key,
                        metadata = EXCLUDED.metadata,
                        processed_at = CURRENT_TIMESTAMP
                """, (url_hash, content_hash, raw_key, thumbnail_key, json.dumps(metadata)))
                conn.commit()
    
    # ==================== CACHE MANAGEMENT ====================
    
    async def invalidate_tenant_cache(self, tenant_id: str) -> int:
        """Invalidate all cached data for a specific tenant."""
        total_invalidated = 0
        
        # Invalidate Redis cache
        if self.redis_enabled:
            try:
                pattern = f"*:{tenant_id}:*"
                keys = self.redis_client.keys(pattern)
                if keys:
                    deleted_count = self.redis_client.delete(*keys)
                    total_invalidated += deleted_count
                    logger.info(f"Invalidated {deleted_count} Redis cache entries for tenant {tenant_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate Redis cache for tenant {tenant_id}: {e}")
        
        # Invalidate PostgreSQL cache
        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                deleted_count = await loop.run_in_executor(
                    self._thread_pool,
                    self._invalidate_postgres_tenant,
                    tenant_id
                )
                total_invalidated += deleted_count
                logger.info(f"Invalidated {deleted_count} PostgreSQL cache entries for tenant {tenant_id}")
            except Exception as e:
                logger.warning(f"Failed to invalidate PostgreSQL cache for tenant {tenant_id}: {e}")
        
        return total_invalidated
    
    def _invalidate_postgres_tenant(self, tenant_id: str) -> int:
        """Invalidate PostgreSQL cache for tenant."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                # Delete from all cache tables
                tables = ['face_embeddings_cache', 'perceptual_hash_cache', 'crawl_cache']
                total_deleted = 0
                
                for table in tables:
                    cur.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (tenant_id,))
                    total_deleted += cur.rowcount
                
                conn.commit()
                return total_deleted
    
    def _calculate_redis_hit_rate(self, redis_info: Dict) -> Optional[float]:
        """Calculate Redis cache hit rate."""
        hits = redis_info.get('keyspace_hits', 0)
        misses = redis_info.get('keyspace_misses', 0)
        
        if hits + misses > 0:
            return (hits / (hits + misses)) * 100
        return None
    
    def _get_postgres_stats(self) -> Dict[str, Any]:
        """Get PostgreSQL cache statistics."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                stats = {}
                
                # Count entries in each cache table
                tables = ['face_embeddings_cache', 'perceptual_hash_cache', 'crawl_cache']
                for table in tables:
                    cur.execute(f"SELECT COUNT(*) as count FROM {table}")
                    result = cur.fetchone()
                    stats[f'{table}_count'] = result['count'] if result else 0
                
                # Get total size (approximate)
                cur.execute("""
                    SELECT pg_size_pretty(pg_total_relation_size('face_embeddings_cache')) as embeddings_size,
                           pg_size_pretty(pg_total_relation_size('perceptual_hash_cache')) as phash_size,
                           pg_size_pretty(pg_total_relation_size('crawl_cache')) as crawl_size
                """)
                size_result = cur.fetchone()
                if size_result:
                    stats.update(size_result)
                
                return stats
    
    async def clear_all_cache(self) -> bool:
        """Clear all cache data from both Redis and PostgreSQL."""
        success = True
        
        # Clear Redis
        if self.redis_enabled:
            try:
                self.redis_client.flushdb()
                logger.warning("Cleared all Redis cache data")
            except Exception as e:
                logger.error(f"Failed to clear Redis cache: {e}")
                success = False
        
        # Clear PostgreSQL
        if self.postgres_enabled:
            try:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    self._thread_pool,
                    self._clear_postgres_cache
                )
                logger.warning("Cleared all PostgreSQL cache data")
            except Exception as e:
                logger.error(f"Failed to clear PostgreSQL cache: {e}")
                success = False
        
        return success
    
    def _clear_postgres_cache(self):
        """Clear all PostgreSQL cache data."""
        with self._get_postgres_connection() as conn:
            with conn.cursor() as cur:
                tables = ['face_embeddings_cache', 'perceptual_hash_cache', 'crawl_cache']
                for table in tables:
                    cur.execute(f"TRUNCATE TABLE {table}")
                conn.commit()
    
    def close_cache_resources(self):
        """
        Clean shutdown of cache service resources.
        
        This function:
        1. Shuts down thread pools with wait=True
        2. Closes Redis connections if needed
        3. Resets statistics
        4. Forces garbage collection
        """
        logger.info("Closing cache service resources...")
        
        try:
            # Shutdown thread pool if it exists
            if hasattr(self, '_thread_pool') and self._thread_pool is not None:
                logger.info("Shutting down cache service thread pool...")
                self._thread_pool.shutdown(wait=True)
                logger.info("Cache service thread pool shutdown complete")
        except Exception as e:
            logger.warning(f"Error shutting down cache service thread pool: {e}")
        
        try:
            # Close Redis connection if it exists
            if self.redis_enabled and hasattr(self, 'redis_client') and self.redis_client is not None:
                logger.info("Closing Redis connection...")
                try:
                    # Try to close the connection pool
                    if hasattr(self.redis_client, 'connection_pool'):
                        self.redis_client.connection_pool.disconnect()
                    # Close the client
                    self.redis_client.close()
                except Exception as close_error:
                    logger.warning(f"Error during Redis close: {close_error}")
                logger.info("Redis connection closed")
        except Exception as e:
            logger.warning(f"Error closing Redis connection: {e}")
        
        try:
            # Reset cache statistics
            self.redis_hits = 0
            self.postgres_hits = 0
            self.cache_misses = 0
            logger.info("Cache service statistics reset")
        except Exception as e:
            logger.warning(f"Error resetting cache statistics: {e}")
        
        try:
            # Force garbage collection
            import gc
            gc.collect()
            logger.info("Cache service cleanup complete - garbage collection triggered")
        except Exception as e:
            logger.warning(f"Error during cache service garbage collection: {e}")


# Global cache service instance
_hybrid_cache_service = None


def get_hybrid_cache_service() -> HybridCacheService:
    """Get hybrid cache service instance."""
    global _hybrid_cache_service
    if _hybrid_cache_service is None:
        _hybrid_cache_service = HybridCacheService()
    return _hybrid_cache_service


# Convenience functions for easy migration from existing cache services
def get_cache_service():
    """Compatibility function - returns hybrid cache service."""
    return get_hybrid_cache_service()


def close_cache_service():
    """Close the global cache service instance."""
    global _hybrid_cache_service
    if _hybrid_cache_service is not None:
        _hybrid_cache_service.close_cache_resources()
        # Clear the client references to ensure proper cleanup
        if hasattr(_hybrid_cache_service, 'redis_client'):
            _hybrid_cache_service.redis_client = None
        _hybrid_cache_service = None
        logger.info("Global cache service instance closed")
