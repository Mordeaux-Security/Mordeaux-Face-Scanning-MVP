CREATE TABLE IF NOT EXISTS images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id TEXT NOT NULL,
  site TEXT,
  object_key TEXT NOT NULL,
  phash TEXT,
  width INT,
  height INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS faces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  image_id UUID REFERENCES images(id) ON DELETE CASCADE,
  bbox FLOAT8[] NOT NULL,
  quality FLOAT8,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
