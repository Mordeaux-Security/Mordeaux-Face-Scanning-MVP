-- Create content table
CREATE TABLE IF NOT EXISTS content (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    source_id UUID NOT NULL,
    external_id VARCHAR(255), -- External system ID
    title VARCHAR(500),
    description TEXT,
    url VARCHAR(1000),
    s3_key_raw VARCHAR(500) NOT NULL,
    s3_key_processed VARCHAR(500),
    content_type VARCHAR(100),
    file_size BIGINT,
    mime_type VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'processing', 'completed', 'failed', 'skipped'
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,
    error_message TEXT,
    fetch_ts TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for content
CREATE INDEX IF NOT EXISTS idx_content_tenant_id ON content(tenant_id);
CREATE INDEX IF NOT EXISTS idx_content_source_id ON content(source_id);
CREATE INDEX IF NOT EXISTS idx_content_external_id ON content(external_id);
CREATE INDEX IF NOT EXISTS idx_content_status ON content(status);
CREATE INDEX IF NOT EXISTS idx_content_fetch_ts ON content(fetch_ts);
CREATE INDEX IF NOT EXISTS idx_content_processing_started_at ON content(processing_started_at);
CREATE INDEX IF NOT EXISTS idx_content_processing_completed_at ON content(processing_completed_at);
CREATE INDEX IF NOT EXISTS idx_content_created_at ON content(created_at);
CREATE INDEX IF NOT EXISTS idx_content_tenant_status ON content(tenant_id, status);
