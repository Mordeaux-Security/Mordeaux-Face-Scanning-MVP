-- Create sources table
CREATE TABLE IF NOT EXISTS sources (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL, -- 'crawler', 'upload', 'api', 'webhook'
    url VARCHAR(500),
    config JSONB DEFAULT '{}',
    status VARCHAR(20) DEFAULT 'active', -- 'active', 'inactive', 'error', 'paused'
    last_run_at TIMESTAMPTZ,
    next_run_at TIMESTAMPTZ,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for sources
CREATE INDEX IF NOT EXISTS idx_sources_tenant_id ON sources(tenant_id);
CREATE INDEX IF NOT EXISTS idx_sources_type ON sources(type);
CREATE INDEX IF NOT EXISTS idx_sources_status ON sources(status);
CREATE INDEX IF NOT EXISTS idx_sources_last_run_at ON sources(last_run_at);
CREATE INDEX IF NOT EXISTS idx_sources_next_run_at ON sources(next_run_at);
CREATE INDEX IF NOT EXISTS idx_sources_created_at ON sources(created_at);
