# Verification-First Flow: Quick Start

## What's Implemented

A complete verification-first identity system that prevents "Person A shows up for Person B" by requiring explicit enrollment and strict 1:1 verification.

### New Components

1. **Identity Collection** (`identities_v1`)
   - Stores identity centroids (average of 3-5 enrollment photos)
   - Indexed by `tenant_id` and `identity_id`

2. **New API Endpoints**
   - `POST /api/v1/enroll_identity` - Enroll user with 3-5 photos
   - `POST /api/v1/verify` - Verify probe photo and return only verified identity's faces

3. **Identity Tagging**
   - Faces can be tagged with `identity_id` during ingestion
   - Enables instant filtering in verification flow

---

## Getting Started

### 1. Ensure Collections Exist

The `identities_v1` collection is automatically created on startup via `ensure_all()`.

To manually verify:
```python
from pipeline.ensure import ensure_identities
ensure_identities()  # Creates identities_v1 with COSINE distance
```

### 2. Enroll a User Identity

```bash
curl -X POST http://localhost:8001/api/v1/enroll_identity \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo-tenant",
    "identity_id": "user-alice",
    "images_b64": [
      "data:image/jpeg;base64,/9j/4AAQ...",  # frontal
      "data:image/jpeg;base64,/9j/4AAQ...",  # slight left
      "data:image/jpeg;base64,/9j/4AAQ..."   # slight right
    ]
  }'
```

**Response:**
```json
{
  "ok": true,
  "identity": {
    "tenant_id": "demo-tenant",
    "identity_id": "user-alice"
  },
  "vector_dim": 512
}
```

### 3. Verify and Search

```bash
curl -X POST http://localhost:8001/api/v1/verify \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo-tenant",
    "identity_id": "user-alice",
    "image_b64": "data:image/jpeg;base64,/9j/4AAQ...",
    "hi_threshold": 0.78,
    "top_k": 50
  }'
```

**Response (verified):**
```json
{
  "verified": true,
  "similarity": 0.85,
  "threshold": 0.78,
  "tenant_id": "demo-tenant",
  "identity_id": "user-alice",
  "results": [
    {
      "id": "face-uuid-123",
      "score": 0.92,
      "payload": {
        "site": "uploads.example.com",
        "url": "https://...",
        "ts": "2025-11-14T10:30:00Z",
        "identity_id": "user-alice"
      }
    }
  ],
  "count": 12
}
```

**Response (rejected):**
```json
{
  "verified": false,
  "similarity": 0.65,
  "threshold": 0.78,
  "tenant_id": "demo-tenant",
  "identity_id": "user-alice",
  "results": [],
  "count": 0
}
```

---

## Tag Faces During Ingestion (Optional but Recommended)

To enable instant filtering by `identity_id`, include it in the pipeline input:

```python
message = {
    "tenant_id": "demo-tenant",
    "site": "user-uploads",
    "url": "https://example.com/upload.jpg",
    "image_sha256": "abc123...",
    "bucket": "raw-images",
    "key": "demo-tenant/abc123.jpg",
    "image_phash": "fedcba...",
    "face_hints": [],
    "identity_id": "user-alice"  # <-- Tag faces with identity
}

# Process through pipeline
from pipeline.processor import process_image
result = process_image(message)
```

Now `/verify` can instantly filter faces by `identity_id` in Qdrant.

---

## Configuration

### Environment Variables

```bash
# Collections
QDRANT_COLLECTION=faces_v1
IDENTITY_COLLECTION=identities_v1

# Verification defaults (can be overridden per request)
VERIFY_HI_THRESHOLD=0.78
VERIFY_TOP_K=50
VERIFY_HNSW_EF=128

# Quality thresholds
MIN_FACE_SIZE=80
BLUR_MIN_VARIANCE=120.0
```

See `env.example` for full configuration options.

---

## Recommended Operating Points

| Parameter | Recommended Value | Purpose |
|-----------|------------------|---------|
| Enrollment images | 3-5 | Build robust centroid |
| `hi_threshold` | 0.78 | Accuracy-first (lower to 0.76 if too strict) |
| `top_k` | 50-100 | Number of faces to return |
| `hnsw_ef` | 128-256 | Search quality |

**Quality Requirements:**
- Good lighting, no harsh shadows
- Near-frontal pose (|yaw| < 20°)
- Sharp focus (blur variance ≥ 120)
- Face size ≥ 80×80 pixels

See `VERIFICATION_FLOW_GUIDE.md` for detailed recommendations.

---

## Usage Examples

### Enroll Identity (One-Time Setup)

```bash
curl -s -X POST http://localhost/pipeline/api/v1/enroll_identity \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "demo",
    "identity_id": "user_123",
    "images_b64": [
      "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
      "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
    ]
  }'
```

**Response:**
```json
{
  "ok": true,
  "identity": {
    "tenant_id": "demo",
    "identity_id": "user_123"
  },
  "vector_dim": 512
}
```

### Verify Identity & Search Faces

```bash
curl -s -X POST http://localhost/pipeline/api/v1/verify \
  -H 'Content-Type: application/json' \
  -d '{
    "tenant_id": "demo",
    "identity_id": "user_123",
    "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "hi_threshold": 0.78,
    "top_k": 50
  }'
```

**Response (Verified = True):**
```json
{
  "verified": true,
  "similarity": 0.85,
  "threshold": 0.78,
  "tenant_id": "demo",
  "identity_id": "user_123",
  "results": [
    {
      "id": "face-uuid-123",
      "score": 0.92,
      "payload": {
        "site": "uploads.example.com",
        "url": "https://example.com/image.jpg",
        "ts": "2025-11-13T22:30:00Z",
        "identity_id": "user_123"
      }
    }
  ],
  "count": 1
}
```

**Response (Verified = False):**
```json
{
  "verified": false,
  "similarity": 0.65,
  "threshold": 0.78,
  "tenant_id": "demo",
  "identity_id": "user_123",
  "results": [],
  "count": 0
}
```

**Key Points:**
- `verified`: `true` if similarity >= threshold, `false` otherwise
- `results`: **Only populated if `verified=true`**, contains faces belonging to that identity
- `results`: **Empty array if `verified=false`**, preventing "Person A shows up for Person B"

## Example Workflow

### User Registration Flow

1. **User uploads 3-5 selfies**
   ```
   Frontend → Backend → enroll_identity()
   ```

2. **System creates identity centroid**
   ```
   identities_v1[tenant:user] = centroid(photos)
   ```

3. **User's photos are tagged during ingestion**
   ```
   faces_v1[*].identity_id = "user-alice"
   ```

### Search/Verification Flow

1. **User uploads probe photo**
   ```
   Frontend → Backend → verify()
   ```

2. **System verifies identity**
   ```
   similarity = cosine(probe, centroid)
   if similarity >= 0.78: verified = true
   ```

3. **System returns only verified faces**
   ```
   if verified:
     return faces WHERE identity_id = "user-alice"
   else:
     return []
   ```

---

## Testing

### Test Enrollment

```python
import requests
import base64

# Read test images
images_b64 = []
for path in ["frontal.jpg", "left.jpg", "right.jpg"]:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        images_b64.append(f"data:image/jpeg;base64,{b64}")

# Enroll
response = requests.post("http://localhost:8001/api/v1/enroll_identity", json={
    "tenant_id": "test-tenant",
    "identity_id": "test-user-1",
    "images_b64": images_b64
})

print(response.json())
```

### Test Verification

```python
# Verify with same person
with open("probe_same_person.jpg", "rb") as f:
    probe_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:8001/api/v1/verify", json={
    "tenant_id": "test-tenant",
    "identity_id": "test-user-1",
    "image_b64": f"data:image/jpeg;base64,{probe_b64}",
    "hi_threshold": 0.78
})

result = response.json()
print(f"Verified: {result['verified']}, Similarity: {result['similarity']}")
```

---

## Migration from Open Search

### Step 1: Add Enrollment to User Onboarding

```python
# During user registration
if user_photos:
    enroll_identity(
        tenant_id=user.tenant_id,
        identity_id=user.id,
        images_b64=user_photos
    )
```

### Step 2: Tag Existing User Content

```python
# When processing user-uploaded images
message = {
    ...
    "identity_id": user.id  # Add this
}
```

### Step 3: Replace Open Search with Verify

```python
# Old (open search)
results = search(tenant_id=tenant_id, image_b64=probe)

# New (verification-first)
results = verify(
    tenant_id=tenant_id,
    identity_id=user.id,
    image_b64=probe,
    hi_threshold=0.78
)
if not results["verified"]:
    return {"error": "verification_failed", "message": "Photo doesn't match enrolled identity"}
```

---

## Error Handling

### Common Errors

**404: `identity_not_enrolled`**
```json
{"detail": "identity_not_enrolled"}
```
Solution: User must enroll with `/enroll_identity` first.

**422: `no_face_detected`**
```json
{"detail": "no_face_detected"}
```
Solution: Photo quality issue. Ask user to upload clearer photo.

**422: `provide at least 2 images for a stable centroid`**
```json
{"detail": "provide at least 2 images for a stable centroid"}
```
Solution: Enrollment requires minimum 2 images (3-5 recommended).

### Quality Checks (Recommended)

```python
def verify_with_quality_check(tenant_id, identity_id, image_b64):
    try:
        result = verify(tenant_id, identity_id, image_b64)
        
        if result["verified"]:
            return {"status": "success", "results": result["results"]}
        else:
            # Guide user based on similarity
            sim = result["similarity"]
            if sim < 0.60:
                message = "Photo doesn't match. Please upload a photo of the enrolled person."
            elif sim < 0.78:
                message = "Photo quality may be poor. Please upload a clearer, front-facing photo."
            
            return {"status": "rejected", "message": message}
    
    except HTTPException as e:
        if e.status_code == 422 and "no_face" in str(e.detail):
            return {"status": "error", "message": "No face detected. Please upload a clear photo."}
        raise
```

---

## Performance

### Expected Latency

- **Enrollment** (3 photos): 300-500ms (CPU), 100-200ms (GPU)
- **Verification**: 150-300ms (CPU), 50-100ms (GPU)

### Optimization

1. **Cache identity centroids** (optional):
   ```python
   # Redis cache for hot identities
   centroid = cache.get(f"identity:{tenant_id}:{identity_id}")
   if not centroid:
       centroid = fetch_from_qdrant()
       cache.set(f"identity:{tenant_id}:{identity_id}", centroid, ttl=3600)
   ```

2. **Reduce `top_k`** for faster response:
   ```python
   top_k = 20  # Instead of 50
   ```

3. **Use GPU** for production:
   ```bash
   ONNX_PROVIDERS_CSV=CUDAExecutionProvider,CPUExecutionProvider
   ```

---

## Security Best Practices

1. **Rate limit enrollment**: Max 5 enrollments per hour per tenant
2. **Require authentication**: Don't allow public enrollment
3. **Audit trail**: Log all enrollment and verification events
4. **Tenant isolation**: Always filter by `tenant_id`
5. **HTTPS only**: Never send biometric data over HTTP

---

## Next Steps

1. ✅ Collections created (`identities_v1`, `faces_v1`)
2. ✅ API endpoints available (`/enroll_identity`, `/verify`)
3. ✅ Face tagging enabled (`identity_id` in pipeline)
4. 📋 Integrate enrollment into user onboarding
5. 📋 Replace open search with verification in user-facing features
6. 📋 Add quality checks and user guidance
7. 📋 Monitor verification pass rates and adjust thresholds

See `VERIFICATION_FLOW_GUIDE.md` for detailed operating recommendations.

