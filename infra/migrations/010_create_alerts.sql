-- Create alerts table
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    alert_type VARCHAR(100) NOT NULL, -- 'security_threat', 'policy_violation', 'system_error', etc.
    severity VARCHAR(20) NOT NULL, -- 'low', 'medium', 'high', 'critical'
    title VARCHAR(255) NOT NULL,
    description TEXT,
    message TEXT, -- Detailed message
    alert_data JSONB DEFAULT '{}', -- Additional alert data
    source_service VARCHAR(100), -- Which service generated the alert
    source_id VARCHAR(255), -- ID of the source (content, face, user, etc.)
    status VARCHAR(20) DEFAULT 'open', -- 'open', 'acknowledged', 'investigating', 'resolved', 'dismissed'
    priority INTEGER DEFAULT 0, -- Higher number = higher priority
    assigned_to VARCHAR(255), -- User assigned to handle the alert
    resolution_notes TEXT,
    resolved_at TIMESTAMPTZ,
    resolved_by VARCHAR(255),
    acknowledged_at TIMESTAMPTZ,
    acknowledged_by VARCHAR(255),
    tags JSONB DEFAULT '[]', -- Tags for categorization
    metadata JSONB DEFAULT '{}', -- Additional metadata
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for alerts
CREATE INDEX IF NOT EXISTS idx_alerts_tenant_id ON alerts(tenant_id);
CREATE INDEX IF NOT EXISTS idx_alerts_alert_type ON alerts(alert_type);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority);
CREATE INDEX IF NOT EXISTS idx_alerts_assigned_to ON alerts(assigned_to);
CREATE INDEX IF NOT EXISTS idx_alerts_resolved_at ON alerts(resolved_at);
CREATE INDEX IF NOT EXISTS idx_alerts_acknowledged_at ON alerts(acknowledged_at);
CREATE INDEX IF NOT EXISTS idx_alerts_source_service ON alerts(source_service);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);
CREATE INDEX IF NOT EXISTS idx_alerts_updated_at ON alerts(updated_at);
CREATE INDEX IF NOT EXISTS idx_alerts_tenant_status ON alerts(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_alerts_tenant_severity ON alerts(tenant_id, severity);
CREATE INDEX IF NOT EXISTS idx_alerts_status_created_at ON alerts(status, created_at);

-- Create partial index for open alerts
CREATE INDEX IF NOT EXISTS idx_alerts_open ON alerts(created_at) 
WHERE status IN ('open', 'acknowledged', 'investigating');
