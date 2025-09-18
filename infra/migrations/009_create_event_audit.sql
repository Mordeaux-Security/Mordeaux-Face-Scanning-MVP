-- Create event_audit table
CREATE TABLE IF NOT EXISTS event_audit (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID,
    event_type VARCHAR(100) NOT NULL, -- 'user_login', 'content_processed', 'face_detected', etc.
    event_category VARCHAR(50) NOT NULL, -- 'authentication', 'processing', 'search', 'admin'
    event_data JSONB NOT NULL DEFAULT '{}',
    user_id VARCHAR(255),
    session_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    request_id VARCHAR(255), -- For tracing requests
    source_service VARCHAR(100), -- Which service generated the event
    severity VARCHAR(20) DEFAULT 'info', -- 'debug', 'info', 'warn', 'error', 'critical'
    tags JSONB DEFAULT '[]', -- Additional tags for filtering
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for event_audit
CREATE INDEX IF NOT EXISTS idx_event_audit_tenant_id ON event_audit(tenant_id);
CREATE INDEX IF NOT EXISTS idx_event_audit_event_type ON event_audit(event_type);
CREATE INDEX IF NOT EXISTS idx_event_audit_event_category ON event_audit(event_category);
CREATE INDEX IF NOT EXISTS idx_event_audit_user_id ON event_audit(user_id);
CREATE INDEX IF NOT EXISTS idx_event_audit_session_id ON event_audit(session_id);
CREATE INDEX IF NOT EXISTS idx_event_audit_ip_address ON event_audit(ip_address);
CREATE INDEX IF NOT EXISTS idx_event_audit_request_id ON event_audit(request_id);
CREATE INDEX IF NOT EXISTS idx_event_audit_source_service ON event_audit(source_service);
CREATE INDEX IF NOT EXISTS idx_event_audit_severity ON event_audit(severity);
CREATE INDEX IF NOT EXISTS idx_event_audit_created_at ON event_audit(created_at);
CREATE INDEX IF NOT EXISTS idx_event_audit_tenant_created_at ON event_audit(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_event_audit_type_created_at ON event_audit(event_type, created_at);

-- Create partial index for recent events (last 30 days)
CREATE INDEX IF NOT EXISTS idx_event_audit_recent ON event_audit(created_at) 
WHERE created_at >= NOW() - INTERVAL '30 days';
