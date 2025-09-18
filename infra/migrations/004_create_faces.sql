-- Create faces table
CREATE TABLE IF NOT EXISTS faces (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content_id UUID NOT NULL,
    face_index INTEGER NOT NULL, -- Order of face in the content
    bbox JSONB NOT NULL, -- {x, y, width, height}
    landmarks JSONB, -- Face landmarks if available
    quality_score FLOAT CHECK (quality_score >= 0 AND quality_score <= 1),
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    s3_key_aligned VARCHAR(500), -- Aligned face image
    s3_key_thumbnail VARCHAR(500), -- Thumbnail image
    phash VARCHAR(64), -- Perceptual hash
    age_estimate INTEGER,
    gender_estimate VARCHAR(10),
    emotion JSONB, -- Detected emotions
    attributes JSONB DEFAULT '{}', -- Additional face attributes
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for faces
CREATE INDEX IF NOT EXISTS idx_faces_content_id ON faces(content_id);
CREATE INDEX IF NOT EXISTS idx_faces_face_index ON faces(face_index);
CREATE INDEX IF NOT EXISTS idx_faces_quality_score ON faces(quality_score);
CREATE INDEX IF NOT EXISTS idx_faces_confidence_score ON faces(confidence_score);
CREATE INDEX IF NOT EXISTS idx_faces_phash ON faces(phash);
CREATE INDEX IF NOT EXISTS idx_faces_age_estimate ON faces(age_estimate);
CREATE INDEX IF NOT EXISTS idx_faces_gender_estimate ON faces(gender_estimate);
CREATE INDEX IF NOT EXISTS idx_faces_created_at ON faces(created_at);
CREATE INDEX IF NOT EXISTS idx_faces_content_face_index ON faces(content_id, face_index);
