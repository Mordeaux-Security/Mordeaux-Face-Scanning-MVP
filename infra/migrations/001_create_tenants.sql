-- Create tenants table
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    slug VARCHAR(100) UNIQUE,
    description TEXT,
    settings JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index on tenant name for faster lookups
CREATE INDEX IF NOT EXISTS idx_tenants_name ON tenants(name);
CREATE INDEX IF NOT EXISTS idx_tenants_slug ON tenants(slug);
CREATE INDEX IF NOT EXISTS idx_tenants_active ON tenants(is_active);
CREATE INDEX IF NOT EXISTS idx_tenants_created_at ON tenants(created_at);
