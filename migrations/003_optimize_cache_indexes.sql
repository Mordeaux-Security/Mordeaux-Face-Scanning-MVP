-- Migration 003: Optimize cache indexes for better performance
-- This migration adds proper indexes to the crawl_cache table for faster lookups

-- Add index on url_hash (primary lookup)
CREATE INDEX IF NOT EXISTS idx_crawl_cache_url_hash ON crawl_cache(url_hash);

-- Add index on phash (most reliable content hash)
CREATE INDEX IF NOT EXISTS idx_crawl_cache_phash ON crawl_cache(phash) WHERE phash IS NOT NULL AND phash != '';

-- Add index on dhash (secondary content hash)
CREATE INDEX IF NOT EXISTS idx_crawl_cache_dhash ON crawl_cache(dhash) WHERE dhash IS NOT NULL AND dhash != '';

-- Add index on whash (wavelet hash)
CREATE INDEX IF NOT EXISTS idx_crawl_cache_whash ON crawl_cache(whash) WHERE whash IS NOT NULL AND whash != '';

-- Add index on ahash (average hash)
CREATE INDEX IF NOT EXISTS idx_crawl_cache_ahash ON crawl_cache(ahash) WHERE ahash IS NOT NULL AND ahash != '';

-- Add index on face_detected for filtering
CREATE INDEX IF NOT EXISTS idx_crawl_cache_face_detected ON crawl_cache(face_detected);

-- Add index on processed_at for cleanup operations
CREATE INDEX IF NOT EXISTS idx_crawl_cache_processed_at ON crawl_cache(processed_at);

-- Add composite index for common queries
CREATE INDEX IF NOT EXISTS idx_crawl_cache_face_processed ON crawl_cache(face_detected, processed_at);

-- Add partial index for non-empty hashes (optimizes similarity searches)
CREATE INDEX IF NOT EXISTS idx_crawl_cache_has_content_hash ON crawl_cache(processed_at) 
WHERE (phash IS NOT NULL AND phash != '') OR (dhash IS NOT NULL AND dhash != '') OR (whash IS NOT NULL AND whash != '') OR (ahash IS NOT NULL AND ahash != '');

-- Update table statistics for better query planning
ANALYZE crawl_cache;
