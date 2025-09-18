-- Create cluster_members table
CREATE TABLE IF NOT EXISTS cluster_members (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    cluster_id UUID NOT NULL,
    face_id UUID NOT NULL,
    similarity_score FLOAT NOT NULL CHECK (similarity_score >= 0 AND similarity_score <= 1),
    distance_score FLOAT, -- Distance from centroid
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    is_representative BOOLEAN DEFAULT false, -- Is this face representative of the cluster
    added_by VARCHAR(50) DEFAULT 'algorithm', -- 'algorithm', 'manual', 'user'
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(cluster_id, face_id)
);

-- Create indexes for cluster_members
CREATE INDEX IF NOT EXISTS idx_cluster_members_cluster_id ON cluster_members(cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_members_face_id ON cluster_members(face_id);
CREATE INDEX IF NOT EXISTS idx_cluster_members_similarity_score ON cluster_members(similarity_score);
CREATE INDEX IF NOT EXISTS idx_cluster_members_distance_score ON cluster_members(distance_score);
CREATE INDEX IF NOT EXISTS idx_cluster_members_confidence_score ON cluster_members(confidence_score);
CREATE INDEX IF NOT EXISTS idx_cluster_members_is_representative ON cluster_members(is_representative);
CREATE INDEX IF NOT EXISTS idx_cluster_members_added_by ON cluster_members(added_by);
CREATE INDEX IF NOT EXISTS idx_cluster_members_created_at ON cluster_members(created_at);
