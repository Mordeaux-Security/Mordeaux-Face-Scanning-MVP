# Presigned Thumbnail Policy

## Overview

This document outlines the presigned URL policy for safely returning images to the UI without exposing raw MinIO paths or internal storage keys.

## Policy Requirements

### 1. Presigned URL Generation
- **TTL**: Maximum 10 minutes (600 seconds)
- **Type**: GET requests only for thumbnail access
- **Security**: Never expose raw object URLs or internal storage keys

### 2. Allowed Metadata Fields
Only the following metadata fields are allowed in API responses:
- `site`: Source website domain
- `url`: Original image URL
- `ts`: Timestamp when image was processed
- `bbox`: Face bounding box coordinates [x, y, width, height]
- `p_hash`: Perceptual hash of the image
- `quality`: Image quality score

### 3. Forbidden Fields
The following fields must never be returned in API responses:
- `raw_url`: Raw image URLs
- `raw_key`: Internal storage keys
- `det_score`: Face detection scores
- `embedding`: Face embedding vectors
- Any other internal system fields

### 4. Thumbnail Specifications
- **Size**: 256px longest side
- **Format**: JPEG
- **Quality**: 88% compression
- **Storage**: MinIO thumbnails bucket

## Implementation Details

### Backend API Endpoints

#### Search Face (`/api/search_face`)
```json
{
  "faces_found": 1,
  "phash": "a1b2c3d4e5f6g7h8",
  "thumb_url": "https://minio.example.com/thumbnails/tenant123/abc123_thumb.jpg?X-Amz-Algorithm=...",
  "results": [
    {
      "id": "face-123",
      "score": 0.95,
      "metadata": {
        "thumb_url": "https://minio.example.com/thumbnails/tenant123/def456_thumb.jpg?X-Amz-Algorithm=...",
        "bbox": [100, 150, 200, 250],
        "site": "example.com",
        "url": "https://example.com/image.jpg",
        "ts": "2024-01-01T12:00:00Z",
        "p_hash": "b2c3d4e5f6g7h8i9",
        "quality": 0.92
      }
    }
  ],
  "vector_backend": "qdrant"
}
```

#### Compare Face (`/api/compare_face`)
Same response format as search_face, but without storing the uploaded image.

#### Face Details (`/faces/{face_id}`)
```json
{
  "face_id": "face-123",
  "payload": {
    "site": "example.com",
    "url": "https://example.com/image.jpg",
    "ts": "2024-01-01T12:00:00Z",
    "bbox": [100, 150, 200, 250],
    "p_hash": "b2c3d4e5f6g7h8i9",
    "quality": 0.92
  },
  "thumb_url": "https://minio.example.com/thumbnails/tenant123/def456_thumb.jpg?X-Amz-Algorithm=..."
}
```

### Face Pipeline API

#### Search Faces (`/search`)
```json
{
  "query": {
    "tenant_id": "tenant-123",
    "search_mode": "image",
    "top_k": 10,
    "threshold": 0.25
  },
  "hits": [
    {
      "face_id": "face-123",
      "score": 0.95,
      "payload": {
        "site": "example.com",
        "url": "https://example.com/image.jpg",
        "ts": "2024-01-01T12:00:00Z",
        "bbox": [100, 150, 200, 250],
        "p_hash": "b2c3d4e5f6g7h8i9",
        "quality": 0.92
      },
      "thumb_url": "https://minio.example.com/thumbnails/tenant123/def456_thumb.jpg?X-Amz-Algorithm=..."
    }
  ],
  "count": 1
}
```

## Configuration

### Environment Variables
```bash
# Presigned URL TTL (seconds)
PRESIGNED_URL_TTL=600  # 10 minutes
PRESIGN_TTL_SEC=600    # Face pipeline

# Thumbnail configuration
THUMBNAIL_MAX_SIZE=256  # pixels
THUMBNAIL_QUALITY=88    # JPEG quality
```

### Storage Configuration
```bash
# MinIO buckets
MINIO_BUCKET_THUMBS=thumbnails
MINIO_BUCKET_RAW=raw-images
MINIO_BUCKET_CROPS=face-crops
MINIO_BUCKET_METADATA=face-metadata
```

## Security Considerations

### 1. URL Expiry
- Presigned URLs automatically expire after the configured TTL
- No manual cleanup required
- URLs become invalid after expiry (403/404 responses)

### 2. Access Control
- URLs are tenant-scoped through storage key structure
- No cross-tenant access possible
- URLs cannot be extended or renewed

### 3. Data Exposure
- Only thumbnails are accessible via presigned URLs
- Raw images remain private
- Internal storage keys are never exposed

## Testing

### Expiry Verification
Run the test script to verify URL expiry:
```bash
python test_presigned_url_expiry.py
```

### Expected Behavior
1. URLs are accessible immediately after generation
2. URLs return 403/404 after TTL + 60 seconds
3. No raw object URLs are exposed in API responses
4. Only allowed metadata fields are returned

## Monitoring

### Logging
- All presigned URL generation is logged
- Expiry times are tracked
- Access attempts are monitored

### Metrics
- URL generation rate
- Expiry events
- Access patterns
- Error rates

## Compliance

This policy ensures:
- ✅ No raw MinIO paths are exposed
- ✅ TTL never exceeds 10 minutes
- ✅ Only allowed metadata is returned
- ✅ Thumbnails are properly sized (256px)
- ✅ URLs expire automatically
- ✅ Security best practices are followed
