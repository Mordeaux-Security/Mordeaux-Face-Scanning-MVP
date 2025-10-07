-- Migration 004: Create tables for Hybrid Cache Service V2
-- This migration creates PostgreSQL tables to support the hybrid Redis + PostgreSQL caching system

-- Face embeddings cache table
CREATE TABLE IF NOT EXISTS face_embeddings_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    content_hash VARCHAR(64) NOT NULL,
    embeddings_data JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Perceptual hash cache table
CREATE TABLE IF NOT EXISTS perceptual_hash_cache (
    cache_key VARCHAR(255) PRIMARY KEY,
    tenant_id VARCHAR(100) NOT NULL DEFAULT 'default',
    content_hash VARCHAR(64) NOT NULL,
    phash VARCHAR(32) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Enhanced crawl cache table (extends existing crawl_cache)
-- This adds tenant support and content_hash for better duplicate detection
ALTER TABLE crawl_cache ADD COLUMN IF NOT EXISTS tenant_id VARCHAR(100) NOT NULL DEFAULT 'default';
ALTER TABLE crawl_cache ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64);

-- Add indexes for performance
CREATE INDEX IF NOT EXISTS idx_face_embeddings_cache_tenant_id ON face_embeddings_cache(tenant_id);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_cache_content_hash ON face_embeddings_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_face_embeddings_cache_created_at ON face_embeddings_cache(created_at);

CREATE INDEX IF NOT EXISTS idx_perceptual_hash_cache_tenant_id ON perceptual_hash_cache(tenant_id);
CREATE INDEX IF NOT EXISTS idx_perceptual_hash_cache_content_hash ON perceptual_hash_cache(content_hash);
CREATE INDEX IF NOT EXISTS idx_perceptual_hash_cache_phash ON perceptual_hash_cache(phash);
CREATE INDEX IF NOT EXISTS idx_perceptual_hash_cache_created_at ON perceptual_hash_cache(created_at);

-- Enhanced indexes for crawl_cache
CREATE INDEX IF NOT EXISTS idx_crawl_cache_tenant_id ON crawl_cache(tenant_id);
CREATE INDEX IF NOT EXISTS idx_crawl_cache_content_hash ON crawl_cache(content_hash) WHERE content_hash IS NOT NULL;

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_face_embeddings_tenant_created ON face_embeddings_cache(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_perceptual_hash_tenant_created ON perceptual_hash_cache(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_crawl_cache_tenant_processed ON crawl_cache(tenant_id, processed_at);

-- Partial indexes for cleanup operations (old entries)
CREATE INDEX IF NOT EXISTS idx_face_embeddings_old_entries ON face_embeddings_cache(created_at) WHERE created_at < now() - interval '7 days';
CREATE INDEX IF NOT EXISTS idx_perceptual_hash_old_entries ON perceptual_hash_cache(created_at) WHERE created_at < now() - interval '7 days';

-- Update table statistics for better query planning
ANALYZE face_embeddings_cache;
ANALYZE perceptual_hash_cache;
ANALYZE crawl_cache;

-- Create a view for cache statistics
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

-- Create cleanup function for old cache entries
CREATE OR REPLACE FUNCTION cleanup_old_cache_entries(retention_days INTEGER DEFAULT 7)
RETURNS TABLE(
    table_name TEXT,
    deleted_count BIGINT
) AS $$
BEGIN
    -- Clean up face embeddings cache
    DELETE FROM face_embeddings_cache 
    WHERE created_at < now() - (retention_days || ' days')::interval;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    table_name := 'face_embeddings_cache';
    RETURN NEXT;
    
    -- Clean up perceptual hash cache
    DELETE FROM perceptual_hash_cache 
    WHERE created_at < now() - (retention_days || ' days')::interval;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    table_name := 'perceptual_hash_cache';
    RETURN NEXT;
    
    -- Clean up crawl cache (keep longer - 30 days default)
    DELETE FROM crawl_cache 
    WHERE processed_at < now() - (GREATEST(retention_days * 4, 30) || ' days')::interval;
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    table_name := 'crawl_cache';
    RETURN NEXT;
    
    -- Update statistics after cleanup
    ANALYZE face_embeddings_cache;
    ANALYZE perceptual_hash_cache;
    ANALYZE crawl_cache;
END;
$$ LANGUAGE plpgsql;

-- Create index monitoring function
CREATE OR REPLACE FUNCTION get_cache_index_usage()
RETURNS TABLE(
    table_name TEXT,
    index_name TEXT,
    index_size TEXT,
    index_usage_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        schemaname||'.'||tablename as table_name,
        indexname as index_name,
        pg_size_pretty(pg_relation_size(indexrelid)) as index_size,
        idx_tup_read as index_usage_count
    FROM pg_stat_user_indexes 
    WHERE schemaname = 'public' 
    AND tablename IN ('face_embeddings_cache', 'perceptual_hash_cache', 'crawl_cache')
    ORDER BY idx_tup_read DESC;
END;
$$ LANGUAGE plpgsql;
