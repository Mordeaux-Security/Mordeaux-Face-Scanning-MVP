# Verification-First Flow: Operating Guide

## Overview

The verification-first flow prevents "Person A shows up for Person B" by requiring explicit identity enrollment and strict 1:1 verification before returning any search results.

## Architecture

1. **Enrollment** (`POST /api/v1/enroll_identity`): Build a robust identity centroid from 3-5 photos
2. **Verification** (`POST /api/v1/verify`): 1:1 match against enrolled identity with strict threshold
3. **Results**: Only return faces belonging to verified identity

---

## Enrollment Best Practices

### Minimum Requirements
- **Number of images**: ≥3 (API enforces minimum of 2, but **3-5 recommended**)
- **Image variety**: 
  - 1 frontal face (straight ahead)
  - 1 slight left turn (~15-20°)
  - 1 slight right turn (~15-20°)
  - Optional: 2 additional angles or expressions

### Image Quality Requirements
- **Lighting**: Good, even lighting (avoid harsh shadows or backlit)
- **Resolution**: Face should be at least 112×112 pixels (larger is better)
- **Blur**: Sharp, clear focus (no motion blur)
- **Pose**: Near-frontal (|yaw| < 20°)
- **Expression**: Neutral to slight smile
- **Occlusion**: No sunglasses, masks, or hand covering face

### Example Enrollment Request

```json
POST /api/v1/enroll_identity
{
  "tenant_id": "user-123",
  "identity_id": "john-doe",
  "images_b64": [
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",  // frontal
    "data:image/jpeg;base64,/9j/4AAQSkZJRg...",  // slight left
    "data:image/jpeg;base64,/9j/4AAQSkZJRg..."   // slight right
  ],
  "overwrite": true
}
```

---

## Verification Threshold Recommendations

### Accuracy-First Settings

| Threshold | Use Case | False Accept Rate | False Reject Rate |
|-----------|----------|-------------------|-------------------|
| **0.78** (default) | High security, accuracy-first | Very Low (~0.01%) | Low-Medium (~5-8%) |
| 0.76 | Balanced (use if 0.78 too strict) | Low (~0.1%) | Low (~3-5%) |
| 0.74 | Convenience-first (not recommended) | Medium (~1%) | Very Low (~1-2%) |

### Recommended Strategy

1. **Start at 0.78** (default `hi_threshold`)
   - Optimized for accuracy and security
   - Minimizes false accepts (wrong person verified)

2. **If users complain about rejections**:
   - Lower to **0.76** BUT add a "try another photo" prompt
   - Guide users to upload clearer, more frontal photos
   - Do NOT go below 0.76 for security-sensitive applications

3. **Monitor metrics**:
   - Track `verified=false` responses
   - If rejection rate > 10%, investigate photo quality or enrollment issues

### Example Verification Request

```json
POST /api/v1/verify
{
  "tenant_id": "user-123",
  "identity_id": "john-doe",
  "image_b64": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
  "hi_threshold": 0.78,
  "top_k": 50
}
```

### Response Interpretation

```json
{
  "verified": true,
  "similarity": 0.85,  // High confidence match
  "threshold": 0.78,
  "results": [...],     // Faces belonging to john-doe
  "count": 12
}
```

- **similarity ≥ 0.85**: High confidence match
- **0.78 ≤ similarity < 0.85**: Valid match, moderate confidence
- **similarity < 0.78**: Rejected, different person or poor quality

---

## Search Recall Parameters

### Recommended Settings

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `top_k` | 50-100 | Number of faces to return if verified |
| `hnsw_ef` | 128-256 | HNSW search quality (higher = more accurate, slower) |

### Configuration

**For accuracy-first applications**:
```python
top_k = 50          # Sufficient for most use cases
hnsw_ef = 128       # Good balance of speed and accuracy
```

**For high-recall applications** (e.g., forensics):
```python
top_k = 100         # More comprehensive results
hnsw_ef = 256       # Maximum accuracy (2x slower)
```

**For real-time/interactive applications**:
```python
top_k = 20          # Faster response
hnsw_ef = 128       # Minimum acceptable quality
```

---

## Quality Rejection Guidelines

### Reject Low-Quality Probes

Implement pre-verification quality checks to provide better UX:

#### Detection Failures
- **No face detected** → "No face found. Please upload a photo with a clear view of your face."
- **Multiple faces** → "Multiple faces detected. Please upload a photo with only your face."

#### Quality Issues
- **Blur** (variance < 120) → "Image is too blurry. Please upload a clearer photo."
- **Extreme pose** (|yaw| > 20°) → "Please face the camera more directly."
- **Low resolution** (face < 80×80px) → "Face is too small. Please move closer or use a higher resolution photo."

#### Example Quality Check Flow

```python
# Before verification
if no_face_detected:
    return {"error": "no_face", "message": "Please upload a clearer, front-facing photo."}

if blur_score < 120:
    return {"error": "too_blurry", "message": "Image is blurry. Please ensure good lighting and focus."}

if abs(yaw_angle) > 20:
    return {"error": "pose_issue", "message": "Please face the camera more directly."}

# Proceed with verification
result = verify(...)
```

---

## Environment Configuration

### Threshold Settings

```bash
# Default verification threshold (can be overridden per request)
VERIFY_HI_THRESHOLD=0.78

# Search parameters
VERIFY_TOP_K=50
VERIFY_HNSW_EF=128
```

### Quality Thresholds

```bash
# Face detection
MIN_FACE_SIZE=80
DET_SCORE_THRESH=0.45

# Quality assessment
BLUR_MIN_VARIANCE=120.0
MIN_OVERALL_QUALITY=0.7
```

---

## Operational Monitoring

### Key Metrics to Track

1. **Enrollment Success Rate**
   - Target: >95% of enrollment attempts succeed
   - If low: Check image quality guidelines

2. **Verification Pass Rate**
   - Target: 85-92% for legitimate users
   - If too low: Consider lowering threshold to 0.76
   - If too high (>98%): Risk of false accepts, check threshold

3. **False Accept Rate** (if ground truth available)
   - Target: <0.1% with threshold=0.78
   - If high: Increase threshold or improve enrollment quality

4. **User Retry Rate**
   - Target: <15% of users retry verification
   - If high: Provide better photo guidance

### Example Dashboard Queries

```sql
-- Verification pass rate (last 7 days)
SELECT 
  COUNT(*) FILTER (WHERE verified = true) * 100.0 / COUNT(*) as pass_rate
FROM verification_logs 
WHERE created_at > NOW() - INTERVAL '7 days';

-- Average similarity score distribution
SELECT 
  CASE 
    WHEN similarity >= 0.85 THEN 'high_confidence'
    WHEN similarity >= 0.78 THEN 'moderate_confidence'
    ELSE 'rejected'
  END as confidence_bucket,
  COUNT(*)
FROM verification_logs
GROUP BY confidence_bucket;
```

---

## Troubleshooting

### "Too many rejections (verified=false)"

1. Check enrollment quality:
   - Are users enrolling with 3+ diverse, high-quality photos?
   - Review enrollment image quality metrics

2. Check probe quality:
   - Add pre-verification quality checks
   - Guide users to upload better photos

3. Consider lowering threshold:
   - Try 0.76 if 0.78 is too strict
   - Monitor false accept rate

### "False accepts (wrong person verified)"

1. **Increase threshold** to 0.80-0.82
2. Improve enrollment:
   - Require 5 images instead of 3
   - Enforce stricter quality requirements
3. Check for duplicate enrollments

### "Slow verification response"

1. Reduce `top_k` (e.g., 20 instead of 50)
2. Reduce `hnsw_ef` (e.g., 64 instead of 128)
3. Add Redis caching for identity centroids

---

## Best Practices Summary

| Component | Recommendation |
|-----------|---------------|
| Enrollment images | 3-5 diverse angles, frontal + slight turns |
| Image quality | Good lighting, sharp focus, near-frontal pose |
| Verification threshold | Start at 0.78, lower to 0.76 if needed |
| Search recall | top_k=50, hnsw_ef=128 |
| Quality rejection | Reject blur < 120, \|yaw\| > 20° |
| User feedback | Clear error messages with photo guidance |
| Monitoring | Track pass rate (target 85-92%) |

---

## API Reference

### Enroll Identity
```
POST /api/v1/enroll_identity
```
- Minimum 2 images (3-5 recommended)
- Returns centroid stored in identities_v1
- Overwrites existing enrollment by default

### Verify Identity
```
POST /api/v1/verify
```
- 1:1 verification against enrolled centroid
- Default threshold: 0.78
- Returns faces only if verified

### Search (legacy, no verification)
```
POST /api/v1/search
```
- Open search without identity verification
- Returns all matching faces (not recommended for user-facing apps)

---

## Security Considerations

1. **Prevent Enrollment Abuse**:
   - Rate limit enrollment attempts (e.g., 5 per hour per tenant)
   - Require authentication for enrollment
   - Log all enrollment events

2. **Protect Identity Data**:
   - Store centroids (vectors) only, not raw photos
   - Implement tenant isolation (filter by tenant_id)
   - Use HTTPS for all API calls

3. **Audit Trail**:
   - Log all verification attempts
   - Track similarity scores for forensic analysis
   - Monitor unusual patterns (e.g., many rejections from same tenant)

---

## Migration from Open Search

If migrating from open face search to verification-first:

1. **Enable enrollment for existing users**:
   - Prompt users to enroll with 3-5 photos
   - Offer incentive (e.g., "improved accuracy")

2. **Tag existing faces with identity_id**:
   - Add identity_id to pipeline input during ingestion
   - Re-process user-uploaded content with identity tags

3. **Deprecate open search**:
   - Phase out `/api/v1/search` for user-facing features
   - Keep for admin/analytics use cases

---

## Support

For questions or issues:
- Check logs for detailed error messages
- Review quality metrics (blur, yaw, det_score)
- Test with high-quality reference photos first
- Adjust thresholds based on your use case (accuracy vs. convenience)

