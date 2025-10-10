-- Migration 005: Add metadata column to crawl_cache for content-addressed storage
-- This migration adds support for storing image metadata instead of image bytes

-- Add metadata column to crawl_cache table
ALTER TABLE crawl_cache ADD COLUMN IF NOT EXISTS metadata JSONB;

-- Create index on metadata column for efficient queries
CREATE INDEX IF NOT EXISTS idx_crawl_cache_metadata ON crawl_cache USING GIN (metadata);

-- Update the cache_stats view to include metadata information
CREATE OR REPLACE VIEW cache_stats AS
SELECT 
    'face_embeddings_cache' as cache_type,
    COUNT(*) as total_entries,
    COUNT(DISTINCT tenant_id) as unique_tenants,
    MIN(created_at) as oldest_entry,
    MAX(created_at) as newest_entry,
    pg_size_pretty(pg_total_relation_size('face_embeddings_cache')) as table_size
FROM face_embeddings_cache
UNION ALL
SELECT 
    'perceptual_hash_cache' as cache_type,
    COUNT(*) as total_entries,
    COUNT(DISTINCT tenant_id) as unique_tenants,
    MIN(created_at) as oldest_entry,
    MAX(created_at) as newest_entry,
    pg_size_pretty(pg_total_relation_size('perceptual_hash_cache')) as table_size
FROM perceptual_hash_cache
UNION ALL
SELECT 
    'crawl_cache' as cache_type,
    COUNT(*) as total_entries,
    COUNT(DISTINCT tenant_id) as unique_tenants,
    MIN(processed_at) as oldest_entry,
    MAX(processed_at) as newest_entry,
    pg_size_pretty(pg_total_relation_size('crawl_cache')) as table_size
FROM crawl_cache;

-- Update table statistics
ANALYZE crawl_cache;
