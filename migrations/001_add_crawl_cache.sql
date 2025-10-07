-- Add crawl cache table for Phase 1: Basic URL caching
CREATE TABLE IF NOT EXISTS crawl_cache (
    id SERIAL PRIMARY KEY,
    url_hash VARCHAR(64) NOT NULL UNIQUE,         -- SHA-256 hash of URL (primary lookup)
    phash VARCHAR(32),                            -- Perceptual hash for content dedup
    raw_image_key VARCHAR(255),                   -- MinIO key for raw image
    thumbnail_key VARCHAR(255),                   -- MinIO key for thumbnail (nullable)
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    face_detected BOOLEAN DEFAULT FALSE           -- Simple face detection flag
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_crawl_cache_url_hash ON crawl_cache(url_hash);
CREATE INDEX IF NOT EXISTS idx_crawl_cache_phash ON crawl_cache(phash);
