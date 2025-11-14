# Verification-First Flow: Usage Guide

Quick reference for using the enrollment and verification endpoints.

---

## 🚀 Quick Start

### 1. Enroll a User Identity (One-Time Setup)

Enroll a user with 3-5 photos to create their identity centroid.

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

**Response (Success):**
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

**Requirements:**
- Minimum 2 images (3-5 recommended for better accuracy)
- Maximum 10 images
- Each image must contain a detectable face
- Images should be: frontal, slight left turn, slight right turn

---

### 2. Verify Identity & Search Faces

Verify a probe photo belongs to the enrolled identity. Only returns faces if verification passes.

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
        "bbox": [100, 150, 200, 250],
        "quality": 0.90,
        "identity_id": "user_123"
      }
    },
    {
      "id": "face-uuid-456",
      "score": 0.88,
      "payload": {
        "site": "uploads.example.com",
        "url": "https://example.com/image2.jpg",
        "ts": "2025-11-13T22:31:00Z",
        "bbox": [150, 200, 250, 300],
        "quality": 0.85,
        "identity_id": "user_123"
      }
    }
  ],
  "count": 2
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
- `similarity`: Cosine similarity score (0-1) between probe and enrolled centroid
- `results`: **Only populated if `verified=true`**, contains faces belonging to that identity
- `results`: **Empty array if `verified=false`**, preventing "Person A shows up for Person B"

---

## 📋 Request Parameters

### Enroll Identity

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tenant_id` | string | Yes | Tenant identifier for multi-tenant isolation |
| `identity_id` | string | Yes | Unique identifier for this user/identity |
| `images_b64` | array[string] | Yes | Array of base64-encoded images (2-10 images) |
| `overwrite` | boolean | No | Whether to overwrite existing enrollment (default: `true`) |

### Verify Identity

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `tenant_id` | string | Yes | Tenant identifier (must match enrollment) |
| `identity_id` | string | Yes | Identity to verify against (must exist) |
| `image_b64` | string | Yes | Base64-encoded probe image |
| `hi_threshold` | float | No | Verification threshold (default: `0.78`, range: 0.0-1.0) |
| `top_k` | integer | No | Max faces to return if verified (default: `50`, range: 1-200) |

---

## 🔧 Example: Python Usage

### Enroll Identity

```python
import requests
import base64

# Read images and encode to base64
images_b64 = []
for image_path in ["frontal.jpg", "left.jpg", "right.jpg"]:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        images_b64.append(f"data:image/jpeg;base64,{b64}")

# Enroll identity
response = requests.post(
    "http://localhost/pipeline/api/v1/enroll_identity",
    json={
        "tenant_id": "demo",
        "identity_id": "user_123",
        "images_b64": images_b64
    }
)

result = response.json()
if result.get("ok"):
    print(f"✓ Enrolled: {result['identity']}")
else:
    print(f"✗ Enrollment failed: {response.status_code}")
```

### Verify Identity

```python
import requests
import base64

# Read probe image
with open("probe_photo.jpg", "rb") as f:
    probe_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"

# Verify
response = requests.post(
    "http://localhost/pipeline/api/v1/verify",
    json={
        "tenant_id": "demo",
        "identity_id": "user_123",
        "image_b64": probe_b64,
        "hi_threshold": 0.78,
        "top_k": 50
    }
)

result = response.json()
if result.get("verified"):
    print(f"✓ Verified! Similarity: {result['similarity']:.2f}")
    print(f"  Found {result['count']} faces:")
    for face in result["results"]:
        print(f"    - {face['id']} (score: {face['score']:.2f})")
else:
    print(f"✗ Verification failed. Similarity: {result['similarity']:.2f} < {result['threshold']}")
```

---

## 🔧 Example: JavaScript/TypeScript Usage

### Enroll Identity

```typescript
async function enrollIdentity(tenantId: string, identityId: string, imageFiles: File[]) {
  // Convert images to base64
  const imagesB64 = await Promise.all(
    imageFiles.map(async (file) => {
      const buffer = await file.arrayBuffer();
      const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
      return `data:${file.type};base64,${base64}`;
    })
  );

  // Enroll
  const response = await fetch('http://localhost/pipeline/api/v1/enroll_identity', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tenant_id: tenantId,
      identity_id: identityId,
      images_b64: imagesB64
    })
  });

  const result = await response.json();
  return result;
}
```

### Verify Identity

```typescript
async function verifyIdentity(
  tenantId: string,
  identityId: string,
  probeImage: File,
  threshold: number = 0.78
) {
  // Convert image to base64
  const buffer = await probeImage.arrayBuffer();
  const base64 = btoa(String.fromCharCode(...new Uint8Array(buffer)));
  const imageB64 = `data:${probeImage.type};base64,${base64}`;

  // Verify
  const response = await fetch('http://localhost/pipeline/api/v1/verify', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      tenant_id: tenantId,
      identity_id: identityId,
      image_b64: imageB64,
      hi_threshold: threshold,
      top_k: 50
    })
  });

  const result = await response.json();
  
  if (result.verified) {
    console.log(`✓ Verified! Similarity: ${result.similarity.toFixed(2)}`);
    console.log(`  Found ${result.count} faces`);
    return result.results;
  } else {
    console.log(`✗ Verification failed. Similarity: ${result.similarity.toFixed(2)}`);
    return [];
  }
}
```

---

## 🎯 Common Workflows

### User Onboarding Flow

```python
def onboard_user(tenant_id: str, user_id: str, photo_paths: list[str]):
    """Onboard a new user with enrollment photos."""
    
    # 1. Encode photos to base64
    images_b64 = []
    for path in photo_paths:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
            images_b64.append(f"data:image/jpeg;base64,{b64}")
    
    # 2. Enroll identity
    response = requests.post(
        "http://localhost/pipeline/api/v1/enroll_identity",
        json={
            "tenant_id": tenant_id,
            "identity_id": user_id,
            "images_b64": images_b64
        }
    )
    
    if response.status_code == 200 and response.json().get("ok"):
        print(f"✓ User {user_id} enrolled successfully")
        return True
    else:
        print(f"✗ Enrollment failed: {response.text}")
        return False
```

### Search Flow (Verification-First)

```python
def search_user_faces(tenant_id: str, user_id: str, probe_image_path: str):
    """Search for faces belonging to a specific user."""
    
    # 1. Encode probe image
    with open(probe_image_path, "rb") as f:
        probe_b64 = f"data:image/jpeg;base64,{base64.b64encode(f.read()).decode()}"
    
    # 2. Verify and get results
    response = requests.post(
        "http://localhost/pipeline/api/v1/verify",
        json={
            "tenant_id": tenant_id,
            "identity_id": user_id,
            "image_b64": probe_b64,
            "hi_threshold": 0.78,
            "top_k": 50
        }
    )
    
    result = response.json()
    
    if result.get("verified"):
        print(f"✓ Verified! Found {result['count']} faces")
        return result["results"]
    else:
        print(f"✗ Verification failed. Similarity: {result['similarity']:.2f}")
        return []
```

### Re-enrollment Flow

```python
def re_enroll_user(tenant_id: str, user_id: str, new_photos: list[str]):
    """Re-enroll a user with new photos (overwrites existing)."""
    
    # Same as enrollment, overwrite is True by default
    return onboard_user(tenant_id, user_id, new_photos)
```

---

## ⚠️ Error Handling

### Common Errors

**404: Identity Not Enrolled**
```json
{
  "detail": "identity_not_enrolled"
}
```
**Solution**: User must enroll first using `/api/v1/enroll_identity`

**422: No Face Detected**
```json
{
  "detail": "no_face_detected"
}
```
**Solution**: Upload a clearer photo with a visible face

**422: Too Few Images**
```json
{
  "detail": "provide at least 2 images for a stable centroid"
}
```
**Solution**: Provide at least 2 images for enrollment (3-5 recommended)

**400: Invalid Image Data**
```json
{
  "detail": "invalid_image_data"
}
```
**Solution**: Check image format and base64 encoding

### Error Handling Example

```python
def safe_verify(tenant_id: str, identity_id: str, image_b64: str):
    try:
        response = requests.post(
            "http://localhost/pipeline/api/v1/verify",
            json={
                "tenant_id": tenant_id,
                "identity_id": identity_id,
                "image_b64": image_b64
            },
            timeout=30
        )
        
        if response.status_code == 404:
            return {"error": "identity_not_enrolled", "message": "User must enroll first"}
        elif response.status_code == 422:
            error_detail = response.json().get("detail", "validation_error")
            return {"error": error_detail, "message": "Photo quality issue"}
        elif response.status_code == 200:
            return response.json()
        else:
            return {"error": "unknown", "message": f"HTTP {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"error": "timeout", "message": "Request timed out"}
    except requests.exceptions.ConnectionError:
        return {"error": "connection_error", "message": "Could not connect to API"}
    except Exception as e:
        return {"error": "unknown", "message": str(e)}
```

---

## 📊 Response Fields Reference

### Enroll Response

| Field | Type | Description |
|-------|------|-------------|
| `ok` | boolean | Whether enrollment succeeded |
| `identity` | object | Enrolled identity info (`tenant_id`, `identity_id`) |
| `vector_dim` | integer | Embedding dimension (always 512) |

### Verify Response

| Field | Type | Description |
|-------|------|-------------|
| `verified` | boolean | Whether verification passed (similarity >= threshold) |
| `similarity` | float | Cosine similarity score (0-1) |
| `threshold` | float | Threshold used for verification |
| `tenant_id` | string | Tenant identifier |
| `identity_id` | string | Identity identifier |
| `results` | array | Faces belonging to identity (only if `verified=true`) |
| `count` | integer | Number of faces returned (0 if `verified=false`) |

### Result Item (in `results` array)

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Face ID (Qdrant point ID) |
| `score` | float | Similarity score (0-1) |
| `payload` | object | Face metadata (site, url, ts, bbox, quality, identity_id) |

---

## 🔍 Best Practices

### Enrollment

1. **Use 3-5 diverse photos**:
   - 1 frontal face (straight ahead)
   - 1 slight left turn (~15-20°)
   - 1 slight right turn (~15-20°)
   - Optional: additional angles or expressions

2. **Ensure good image quality**:
   - Good lighting (even, no harsh shadows)
   - Sharp focus (no blur)
   - Near-frontal pose (|yaw| < 20°)
   - Face size ≥ 80×80 pixels

3. **Handle errors gracefully**:
   - Check for `no_face_detected` errors
   - Guide users to upload better photos
   - Retry with different images if needed

### Verification

1. **Choose appropriate threshold**:
   - **0.78** (default): Accuracy-first, strict
   - **0.76**: Balanced (use if 0.78 too strict)
   - **< 0.76**: Not recommended (higher false accept rate)

2. **Handle verification failures**:
   - If similarity < threshold: Ask user to upload clearer photo
   - If similarity < 0.60: Likely different person
   - Provide helpful feedback to users

3. **Monitor metrics**:
   - Track verification pass rates
   - Adjust threshold based on user feedback
   - Monitor false accept/reject rates

---

## 📚 Related Documentation

- **Deployment Guide**: `DEPLOYMENT_PLAN.md`
- **Operating Guide**: `VERIFICATION_FLOW_GUIDE.md`
- **Quick Start**: `VERIFICATION_QUICKSTART.md`
- **Implementation Details**: `VERIFICATION_IMPLEMENTATION_SUMMARY.md`
- **API Docs**: `http://localhost/pipeline/docs` (Swagger UI)

---

**Ready to use?** Start by enrolling a test user, then verify with a probe photo!

