-- Enhance crawl cache table for tolerant content deduplication
-- Add multiple hash types for better duplicate detection

-- Add new columns for multiple hash types
ALTER TABLE crawl_cache ADD COLUMN IF NOT EXISTS dhash VARCHAR(32);
ALTER TABLE crawl_cache ADD COLUMN IF NOT EXISTS whash VARCHAR(32);
ALTER TABLE crawl_cache ADD COLUMN IF NOT EXISTS ahash VARCHAR(32);

-- Add indexes for new hash columns
CREATE INDEX IF NOT EXISTS idx_crawl_cache_dhash ON crawl_cache(dhash);
CREATE INDEX IF NOT EXISTS idx_crawl_cache_whash ON crawl_cache(whash);
CREATE INDEX IF NOT EXISTS idx_crawl_cache_ahash ON crawl_cache(ahash);

-- Add similarity threshold configuration
ALTER TABLE crawl_cache ADD COLUMN IF NOT EXISTS similarity_threshold INTEGER DEFAULT 5;
