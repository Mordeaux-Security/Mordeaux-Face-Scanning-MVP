-- Create clusters table
CREATE TABLE IF NOT EXISTS clusters (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255),
    description TEXT,
    cluster_type VARCHAR(50) DEFAULT 'face', -- 'face', 'person', 'group'
    algorithm VARCHAR(50) NOT NULL, -- 'dbscan', 'kmeans', 'hierarchical', 'manual'
    parameters JSONB DEFAULT '{}', -- Algorithm parameters
    centroid FLOAT[], -- Centroid vector for the cluster
    centroid_norm FLOAT, -- Precomputed L2 norm
    size INTEGER DEFAULT 0, -- Number of faces in cluster
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for clusters
CREATE INDEX IF NOT EXISTS idx_clusters_tenant_id ON clusters(tenant_id);
CREATE INDEX IF NOT EXISTS idx_clusters_name ON clusters(name);
CREATE INDEX IF NOT EXISTS idx_clusters_cluster_type ON clusters(cluster_type);
CREATE INDEX IF NOT EXISTS idx_clusters_algorithm ON clusters(algorithm);
CREATE INDEX IF NOT EXISTS idx_clusters_size ON clusters(size);
CREATE INDEX IF NOT EXISTS idx_clusters_quality_score ON clusters(quality_score);
CREATE INDEX IF NOT EXISTS idx_clusters_is_active ON clusters(is_active);
CREATE INDEX IF NOT EXISTS idx_clusters_created_at ON clusters(created_at);
CREATE INDEX IF NOT EXISTS idx_clusters_tenant_active ON clusters(tenant_id, is_active);
