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


def get_cache_service() -> CrawlCacheDB:
    """Get cache service instance."""
    return CrawlCacheDB()
