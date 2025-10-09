# Configuration Guide

## Environment Variable Consolidation

To prevent code bloating and configuration duplication, all environment variables have been **consolidated into a single root-level `.env.example` file**.

### 📁 File Locations

- **Main Config**: `.env.example` (root level) ✅ **USE THIS**
- **Deprecated**: `face-pipeline/env.example` ⚠️ (kept for reference only)

### 🔄 Variable Name Mapping

Due to legacy naming in `docker-compose.yml`, we support **both naming conventions**:

| Purpose | Legacy (docker-compose) | New (face-pipeline) | Value |
|---------|------------------------|---------------------|-------|
| Storage Endpoint | `S3_ENDPOINT` | `MINIO_ENDPOINT` | `localhost:9000` |
| Access Key | `S3_ACCESS_KEY` | `MINIO_ACCESS_KEY` | `changeme` |
| Secret Key | `S3_SECRET_KEY` | `MINIO_SECRET_KEY` | `changeme` |
| Use SSL | `S3_USE_SSL` | `MINIO_SECURE` | `false` |
| Raw Images | `S3_BUCKET_RAW` | `MINIO_BUCKET_RAW` | `raw-images` |
| Thumbnails | `S3_BUCKET_THUMBS` | `MINIO_BUCKET_THUMBS` | `thumbnails` |
| Face Crops | N/A | `MINIO_BUCKET_CROPS` | `face-crops` |
| Metadata | `S3_BUCKET_AUDIT` | `MINIO_BUCKET_METADATA` | `face-metadata` |

### 🎯 Requested Face Pipeline Variables

All requested variables are now in the root `.env.example`:

```bash
# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=changeme
MINIO_SECRET_KEY=changeme
MINIO_SECURE=false
MINIO_BUCKET_RAW=raw-images
MINIO_BUCKET_CROPS=face-crops
MINIO_BUCKET_THUMBS=thumbnails
MINIO_BUCKET_METADATA=face-metadata

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=faces_v1

# Face Pipeline Settings
MAX_CONCURRENT=4
JOB_TIMEOUT_SEC=300
FACE_MIN_SIZE=80
BLUR_MIN_VARIANCE=120.0
PRESIGN_TTL_SEC=600
```

### 🔧 How Settings Work

The `face-pipeline/config/settings.py` module:

1. **Reads from root `.env` file** (prevents duplication)
2. **Supports both naming conventions** (MINIO_* and S3_*)
3. **Prefers MINIO_* prefix** but falls back to S3_* if not set
4. **Provides helper properties** for compatibility:
   - `settings.storage_endpoint` → Uses MINIO_ENDPOINT or S3_ENDPOINT
   - `settings.storage_access_key` → Uses MINIO_ACCESS_KEY or S3_ACCESS_KEY
   - etc.

### 📝 Usage Example

```python
from face_pipeline.config.settings import settings

# Access configuration
print(settings.minio_endpoint)          # localhost:9000
print(settings.qdrant_url)              # http://localhost:6333
print(settings.max_concurrent)          # 4
print(settings.face_min_size)           # 80
print(settings.blur_min_variance)       # 120.0

# Legacy compatibility
print(settings.storage_endpoint)        # Works with both MINIO_ and S3_ prefixes
```

### 🚀 Quick Start

1. **Copy the example file**:
   ```bash
   cp .env.example .env
   ```

2. **Update values** in `.env`:
   ```bash
   # Change from placeholders to real values
   MINIO_ACCESS_KEY=your_actual_key
   MINIO_SECRET_KEY=your_actual_secret
   POSTGRES_PASSWORD=your_db_password
   ```

3. **Run services**:
   ```bash
   docker-compose up -d
   ```

### ✅ Anti-Bloat Measures Taken

1. ✅ **Single source of truth**: All env vars in root `.env.example`
2. ✅ **Deprecated duplicate**: Marked `face-pipeline/env.example` as deprecated
3. ✅ **Backward compatibility**: Settings module supports both naming conventions
4. ✅ **Clear documentation**: This guide explains the consolidation
5. ✅ **Variable mapping table**: Shows legacy vs new naming

### ⚠️ Important Notes

- **Docker Compose** expects `S3_*` variables (legacy naming)
- **Face Pipeline** prefers `MINIO_*` variables (clearer naming)
- **Both work**: The settings module handles the translation
- **No duplication**: Same values, just different variable names for compatibility

### 🔍 Duplication Check

Before adding **any** new environment variable:

1. Check if it exists in `.env.example`
2. Check if there's a legacy equivalent (S3_* vs MINIO_*)
3. Add to root `.env.example` only (not to face-pipeline/env.example)
4. Update this documentation if adding new variables

---

**Last Updated**: After consolidation to prevent config bloat

