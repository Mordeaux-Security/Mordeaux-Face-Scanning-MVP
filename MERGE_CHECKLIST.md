# 🔒 Critical Items to Preserve During Merges

## ⚠️ DO NOT CHANGE THESE - They Will Break Your System

### 1. **Docker Service Names** (docker-compose.yml)
```yaml
# These service names are hardcoded in multiple places:
- api              # Referenced as http://api:8000
- face-pipeline    # Referenced as http://face-pipeline:8001
- redis            # Referenced as redis://redis:6379
- minio             # Referenced as minio:9000
- qdrant           # Referenced as http://qdrant:6333
- nginx             # Main entry point
```
**Why:** Service discovery depends on these exact names.

---

### 2. **Network Configuration**
```yaml
networks:
  mordeaux: {}  # All services must use this network
```
**Why:** Services communicate via this network name.

---

### 3. **Volume Names**
```yaml
volumes:
  minio-data: {}    # MinIO storage - contains your images!
  qdrant-data: {}   # Qdrant vectors - contains face embeddings!
```
**Why:** Changing these creates NEW empty volumes, losing all data!

---

### 4. **Port Mappings** (Critical for External Access)
```yaml
nginx:    "80:80"           # Main web interface
minio:    "9001:9001"       # MinIO console
qdrant:   "6333:6333"       # Qdrant dashboard
```
**Why:** Frontend/API clients expect these ports.

---

### 5. **Internal Service URLs** (Environment Variables)
```yaml
PIPELINE_URL: http://face-pipeline:8001
REDIS_URL: redis://redis:6379/0
MINIO_ENDPOINT: minio:9000
QDRANT_URL: http://qdrant:6333
```
**Why:** Services communicate using these exact URLs.

---

### 6. **Bucket Names** (MinIO Storage)
```yaml
MINIO_BUCKET_RAW: raw-images
MINIO_BUCKET_CROPS: face-crops
MINIO_BUCKET_THUMBS: thumbnails
MINIO_BUCKET_METADATA: face-metadata
```
**Why:** Code references these exact bucket names. Changing = data loss!

---

### 7. **Qdrant Collection Names**
```yaml
QDRANT_COLLECTION: faces_v1
IDENTITY_COLLECTION: identities_v1
```
**Why:** These are your vector database tables. Changing = can't find existing faces!

---

### 8. **Vector Dimension**
```yaml
VECTOR_DIM: "512"
```
**Why:** Face embeddings are 512-dimensional. Changing breaks all existing vectors!

---

### 9. **Critical Thresholds** (Face Recognition)
```yaml
VERIFY_HI_THRESHOLD: "0.78"    # Identity verification threshold
VERIFY_TOP_K: "50"             # Search result limit
VERIFY_HNSW_EF: "128"          # Vector search precision
```
**Why:** These affect search accuracy and security. Changing = false positives/negatives!

---

### 10. **Redis Stream/Group Names**
```yaml
REDIS_STREAM_NAME: face:ingest
REDIS_GROUP_NAME: pipeline
REDIS_CONSUMER_NAME: pipe-1
```
**Why:** Queue workers depend on these exact names.

---

### 11. **Health Check Endpoints**
```yaml
# API
GET /healthz
GET /api/v1/health

# Face Pipeline
GET /api/v1/health
GET /healthz
```
**Why:** Docker health checks and monitoring depend on these.

---

### 12. **API Endpoint Paths**
```yaml
POST /api/v1/search
POST /api/v1/enroll_identity
POST /api/v1/verify
POST /api/v1/identity_safe_search
```
**Why:** Frontend and external clients call these exact paths.

---

### 13. **Nginx Routing Configuration**
```nginx
location /api/        → http://api:8000
location /pipeline/   → http://face-pipeline:8001/
location /             → Frontend static files
```
**Why:** All HTTP routing depends on this configuration.

---

### 14. **Model Configuration** (ONNX)
```yaml
DET_SIZE: "640,640"              # Detection image size
IMAGE_SIZE: 112                   # Face embedding size
ONNX_PROVIDERS_CSV: CPUExecutionProvider
```
**Why:** Model inputs/outputs are fixed. Changing = crashes!

---

### 15. **MinIO Credentials** (Development)
```yaml
MINIO_ROOT_USER: minioadmin
MINIO_ROOT_PASSWORD: minioadmin
MINIO_ACCESS_KEY: minioadmin
MINIO_SECRET_KEY: minioadmin
```
**Why:** If changed, existing containers can't access MinIO!

---

## ✅ Safe to Change (During Merges)

- Log levels (`LOG_LEVEL`)
- Timeout values (can be tuned)
- Rate limiting values (can be adjusted)
- Worker concurrency (scaling)
- Feature flags (`ENABLE_QUEUE_WORKER`, etc.)
- Comments and documentation
- Code improvements (as long as APIs stay the same)

---

## 🔍 Pre-Merge Checklist

Before merging, verify:
- [ ] Service names unchanged
- [ ] Volume names unchanged
- [ ] Port mappings unchanged
- [ ] Bucket names unchanged
- [ ] Collection names unchanged
- [ ] Vector dimension unchanged
- [ ] Environment variable names unchanged
- [ ] API endpoint paths unchanged
- [ ] Network name unchanged

---

## 🚨 If You Must Change Something Critical

1. **Create a migration script** for data
2. **Update ALL references** across codebase
3. **Test in development** first
4. **Document the change** in CHANGELOG.md
5. **Coordinate with team** before merging

---

## 📝 Example: What Happens If You Change Bucket Names

```yaml
# ❌ WRONG - Changed bucket name
MINIO_BUCKET_RAW: new-raw-images

# Result:
# - Code looks for "new-raw-images"
# - All existing images are in "raw-images"
# - System can't find any images!
# - Data appears "lost" (it's not, just inaccessible)
```

---

## 💡 Best Practice

**When in doubt, don't change it.** If you need to change something critical:
1. Discuss with team first
2. Create a migration plan
3. Test thoroughly
4. Update documentation


