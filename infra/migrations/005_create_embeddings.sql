-- Create embeddings table
CREATE TABLE IF NOT EXISTS embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    face_id UUID NOT NULL,
    model_name VARCHAR(100) NOT NULL, -- 'facenet', 'arcface', 'insightface', etc.
    model_version VARCHAR(50) NOT NULL,
    vector FLOAT[] NOT NULL, -- 512-dimensional vector
    vector_norm FLOAT, -- Precomputed L2 norm for faster similarity
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for embeddings
CREATE INDEX IF NOT EXISTS idx_embeddings_face_id ON embeddings(face_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_name ON embeddings(model_name);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_version ON embeddings(model_version);
CREATE INDEX IF NOT EXISTS idx_embeddings_vector_norm ON embeddings(vector_norm);
CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);
CREATE INDEX IF NOT EXISTS idx_embeddings_model_face ON embeddings(model_name, face_id);

-- Create GIN index for vector similarity search (if pgvector extension is available)
-- CREATE INDEX IF NOT EXISTS idx_embeddings_vector_gin ON embeddings USING gin(vector);
