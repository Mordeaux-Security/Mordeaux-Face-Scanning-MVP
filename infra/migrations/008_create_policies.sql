-- Create policies table
CREATE TABLE IF NOT EXISTS policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    policy_type VARCHAR(50) DEFAULT 'access_control', -- 'access_control', 'data_retention', 'privacy'
    rules JSONB NOT NULL DEFAULT '[]', -- Array of policy rules
    conditions JSONB DEFAULT '{}', -- Global conditions
    actions JSONB DEFAULT '[]', -- Default actions
    priority INTEGER DEFAULT 0, -- Higher number = higher priority
    is_active BOOLEAN DEFAULT true,
    is_default BOOLEAN DEFAULT false, -- Default policy for tenant
    effective_from TIMESTAMPTZ DEFAULT NOW(),
    effective_until TIMESTAMPTZ,
    created_by VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for policies
CREATE INDEX IF NOT EXISTS idx_policies_tenant_id ON policies(tenant_id);
CREATE INDEX IF NOT EXISTS idx_policies_name ON policies(name);
CREATE INDEX IF NOT EXISTS idx_policies_policy_type ON policies(policy_type);
CREATE INDEX IF NOT EXISTS idx_policies_priority ON policies(priority);
CREATE INDEX IF NOT EXISTS idx_policies_is_active ON policies(is_active);
CREATE INDEX IF NOT EXISTS idx_policies_is_default ON policies(is_default);
CREATE INDEX IF NOT EXISTS idx_policies_effective_from ON policies(effective_from);
CREATE INDEX IF NOT EXISTS idx_policies_effective_until ON policies(effective_until);
CREATE INDEX IF NOT EXISTS idx_policies_created_at ON policies(created_at);
CREATE INDEX IF NOT EXISTS idx_policies_tenant_active ON policies(tenant_id, is_active);
CREATE INDEX IF NOT EXISTS idx_policies_tenant_default ON policies(tenant_id, is_default);
