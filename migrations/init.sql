CREATE TABLE IF NOT EXISTS images (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id TEXT NOT NULL,
  site TEXT,
  object_key TEXT NOT NULL,
  bucket_name TEXT NOT NULL DEFAULT 'raw-images',
  phash TEXT,
  width INT,
  height INT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS faces (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  image_id UUID REFERENCES images(id) ON DELETE CASCADE,
  tenant_id TEXT NOT NULL,
  vector_id TEXT,
  collection_name TEXT NOT NULL DEFAULT 'faces_v1',
  bbox FLOAT8[] NOT NULL,
  quality FLOAT8,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Audit tables
CREATE TABLE IF NOT EXISTS audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id TEXT,
  tenant_id TEXT NOT NULL,
  method TEXT NOT NULL,
  path TEXT NOT NULL,
  status_code INTEGER NOT NULL,
  process_time FLOAT NOT NULL,
  user_agent TEXT,
  ip_address INET,
  response_size INTEGER DEFAULT 0,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS search_audit_logs (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  request_id TEXT,
  tenant_id TEXT NOT NULL,
  operation_type TEXT NOT NULL,
  face_count INTEGER DEFAULT 0,
  result_count INTEGER DEFAULT 0,
  vector_backend TEXT NOT NULL,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_images_tenant_id ON images(tenant_id);
CREATE INDEX IF NOT EXISTS idx_images_created_at ON images(created_at);
CREATE INDEX IF NOT EXISTS idx_images_phash ON images(phash);
CREATE INDEX IF NOT EXISTS idx_images_bucket_name ON images(bucket_name);

CREATE INDEX IF NOT EXISTS idx_faces_tenant_id ON faces(tenant_id);
CREATE INDEX IF NOT EXISTS idx_faces_image_id ON faces(image_id);
CREATE INDEX IF NOT EXISTS idx_faces_created_at ON faces(created_at);
CREATE INDEX IF NOT EXISTS idx_faces_vector_id ON faces(vector_id);
CREATE INDEX IF NOT EXISTS idx_faces_collection_name ON faces(collection_name);

-- Audit logs indexes
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_id ON audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_status_code ON audit_logs(status_code);
CREATE INDEX IF NOT EXISTS idx_audit_logs_method ON audit_logs(method);
CREATE INDEX IF NOT EXISTS idx_audit_logs_path ON audit_logs(path);
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_created ON audit_logs(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_status_created ON audit_logs(status_code, created_at);

-- Search audit logs indexes
CREATE INDEX IF NOT EXISTS idx_search_audit_logs_tenant_id ON search_audit_logs(tenant_id);
CREATE INDEX IF NOT EXISTS idx_search_audit_logs_created_at ON search_audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_search_audit_logs_operation_type ON search_audit_logs(operation_type);
CREATE INDEX IF NOT EXISTS idx_search_audit_logs_vector_backend ON search_audit_logs(vector_backend);
CREATE INDEX IF NOT EXISTS idx_search_audit_logs_tenant_created ON search_audit_logs(tenant_id, created_at);
CREATE INDEX IF NOT EXISTS idx_search_audit_logs_operation_created ON search_audit_logs(operation_type, created_at);

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_audit_logs_tenant_status_created ON audit_logs(tenant_id, status_code, created_at);
CREATE INDEX IF NOT EXISTS idx_search_audit_logs_tenant_operation_created ON search_audit_logs(tenant_id, operation_type, created_at);

-- Partial indexes for error analysis
CREATE INDEX IF NOT EXISTS idx_audit_logs_errors ON audit_logs(tenant_id, created_at) WHERE status_code >= 400;
CREATE INDEX IF NOT EXISTS idx_search_audit_logs_failures ON search_audit_logs(tenant_id, created_at) WHERE result_count = 0;
