# API Documentation

## Overview

This document provides comprehensive documentation for the Mordeaux Face Scanning API, including endpoint specifications, tenant rules, authentication, and presigned URL policies.

## Authentication & Tenant Rules

### Tenant Validation
All API endpoints (except health checks and documentation) require tenant authentication via the `X-Tenant-ID` header.

#### Tenant Rules:
1. **Required Header**: `X-Tenant-ID` must be present in all requests
2. **Minimum Length**: Tenant ID must be at least 3 characters long
3. **Allow-list**: If configured, tenant must be in the allowed tenant list
4. **Status Validation**: Tenant must have "active" status
5. **Exempt Endpoints**: Health checks, docs, and admin endpoints are exempt from tenant validation

#### Tenant Status Types:
- `active`: Normal operation allowed
- `suspended`: Access denied with "Tenant account is suspended" error
- `deleted`: Access denied with "Tenant account has been deleted" error
- Other statuses: Access denied with "Tenant account is in [status] status" error

#### Error Responses:
- **400 Bad Request**: Missing or invalid tenant ID format
- **403 Forbidden**: Tenant not in allow-list, not found, or inactive status

## Presigned URL Policy

### Security Requirements
- **TTL**: Maximum 10 minutes (600 seconds)
- **Type**: GET requests only for thumbnail access
- **Security**: Never expose raw object URLs or internal storage keys
- **Tenant Scoping**: URLs are tenant-scoped through storage key structure

### Allowed Metadata Fields
Only these fields are returned in API responses:
- `site`: Source website domain
- `url`: Original image URL
- `ts`: Timestamp when image was processed
- `bbox`: Face bounding box coordinates [x, y, width, height]
- `p_hash`: Perceptual hash of the image
- `quality`: Image quality score
- `thumb_url`: Presigned URL for thumbnail access

### Forbidden Fields
These fields are never returned:
- `raw_url`: Raw image URLs
- `raw_key`: Internal storage keys
- `det_score`: Face detection scores
- `embedding`: Face embedding vectors
- Any other internal system fields

## Health & Monitoring Endpoints

### GET /ready

Comprehensive readiness endpoint that checks if MinIO and Qdrant are reachable and healthy. Includes timestamp, dependency names, detailed error messages, and thorough validation of storage (bucket listing, presigned URLs) and vector database (faces_v1 collection with dim=512).

**Purpose**: Verify that all critical dependencies are operational before accepting traffic.

**HTTP Status Codes**:
- `200 OK`: All dependencies are healthy and ready
- `503 Service Unavailable`: One or more dependencies are unhealthy

---

## API Endpoints

### Face Operations

#### POST /index_face
Upload an image, extract face embeddings, and store them in the vector database.

**Success Response (200 OK)**:
```json
{
  "indexed": 2,
  "phash": "a1b2c3d4e5f6g7h8",
  "thumb_url": "https://minio.example.com/thumbnails/tenant123/abc123_thumb.jpg?X-Amz-Algorithm=...",
  "vector_backend": "qdrant"
}
```

**Error Responses**:
- **400 Bad Request**: Invalid image format or size
- **413 Payload Too Large**: Image exceeds 10MB limit
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Processing failure

#### POST /search_face
Upload an image, extract face embeddings, and find similar faces in the database.

**Success Response (200 OK)**:
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

**Error Responses**:
- **400 Bad Request**: Invalid image format, size, or no faces detected
- **413 Payload Too Large**: Image exceeds 10MB limit
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Processing failure

#### POST /compare_face
Compare an uploaded image against existing faces without storing the image.

**Success Response (200 OK)**:
```json
{
  "phash": "a1b2c3d4e5f6g7h8",
  "faces_found": 1,
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
  "vector_backend": "qdrant",
  "message": "Found 1 face(s) and 1 similar matches"
}
```

**Error Responses**:
- **400 Bad Request**: Invalid image format, size, or no faces detected
- **413 Payload Too Large**: Image exceeds 10MB limit
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Processing failure

### Batch Processing

#### POST /batch/index
Create a new batch job to process multiple images for face indexing.

**Success Response (200 OK)**:
```json
{
  "batch_id": "batch-123e4567-e89b-12d3-a456-426614174000",
  "total_images": 10,
  "status": "created",
  "message": "Batch job created with 10 images"
}
```

**Error Responses**:
- **400 Bad Request**: No URLs provided or batch too large (>100 images)
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Batch creation failure

#### GET /batch/{batch_id}/status
Get the status of a batch job.

**Success Response (200 OK)**:
```json
{
  "batch_id": "batch-123e4567-e89b-12d3-a456-426614174000",
  "status": "processing",
  "total_images": 10,
  "processed_images": 7,
  "successful_images": 6,
  "failed_images": 1,
  "errors": [
    {
      "image_url": "https://example.com/broken.jpg",
      "error": "Failed to download image"
    }
  ],
  "created_at": 1640995200.0,
  "updated_at": 1640995300.0
}
```

**Error Responses**:
- **404 Not Found**: Batch job not found
- **403 Forbidden**: Access denied to batch job

#### GET /batch/list
List batch jobs for the current tenant.

**Success Response (200 OK)**:
```json
{
  "batches": [
    {
      "batch_id": "batch-123e4567-e89b-12d3-a456-426614174000",
      "status": "completed",
      "total_images": 10,
      "processed_images": 10,
      "successful_images": 9,
      "failed_images": 1,
      "created_at": 1640995200.0,
      "updated_at": 1640995400.0
    }
  ],
  "total_count": 1,
  "limit": 20,
  "offset": 0,
  "has_more": false
}
```

#### DELETE /batch/{batch_id}
Cancel a batch job.

**Success Response (200 OK)**:
```json
{
  "message": "Batch job batch-123e4567-e89b-12d3-a456-426614174000 cancelled",
  "batch_id": "batch-123e4567-e89b-12d3-a456-426614174000",
  "status": "cancelled"
}
```

**Error Responses**:
- **404 Not Found**: Batch job not found
- **403 Forbidden**: Access denied to batch job
- **400 Bad Request**: Cannot cancel completed or failed batch job

### Webhooks

#### POST /webhooks/register
Register a new webhook endpoint.

**Success Response (200 OK)**:
```json
{
  "webhook_id": "webhook_1",
  "url": "https://example.com/webhook",
  "events": ["face.indexed", "face.searched"],
  "message": "Webhook registered successfully"
}
```

**Error Responses**:
- **400 Bad Request**: Invalid webhook configuration or events
- **500 Internal Server Error**: Registration failure

#### GET /webhooks/list
List all webhook endpoints for the current tenant.

**Success Response (200 OK)**:
```json
{
  "webhooks": [
    {
      "url": "https://example.com/webhook",
      "events": ["face.indexed", "face.searched"],
      "has_secret": true,
      "timeout": 30,
      "retry_count": 3,
      "created_at": 1640995200.0,
      "last_used": 1640995300.0,
      "success_count": 15,
      "failure_count": 2
    }
  ]
}
```

#### DELETE /webhooks/unregister
Unregister a webhook endpoint.

**Success Response (200 OK)**:
```json
{
  "message": "Webhook unregistered successfully",
  "url": "https://example.com/webhook"
}
```

**Error Responses**:
- **404 Not Found**: Webhook not found

#### POST /webhooks/test
Test a webhook endpoint.

**Success Response (200 OK)**:
```json
{
  "success": true,
  "message": "Test webhook sent successfully",
  "details": {
    "url": "https://example.com/webhook",
    "success": true,
    "status_code": 200,
    "attempt": 1
  }
}
```

#### GET /webhooks/stats
Get webhook delivery statistics.

**Success Response (200 OK)**:
```json
{
  "total_endpoints": 2,
  "total_success": 45,
  "total_failures": 3,
  "success_rate": 0.9375,
  "endpoints": [
    {
      "url": "https://example.com/webhook",
      "events": ["face.indexed"],
      "success_count": 25,
      "failure_count": 1,
      "last_used": 1640995300.0
    }
  ]
}
```

### Admin Operations

#### POST /admin/cleanup
Run cleanup jobs manually.

**Success Response (200 OK)**:
```json
{
  "status": "success",
  "results": {
    "cleaned_old_thumbs": 150,
    "cleaned_old_audit_logs": 25
  },
  "message": "Cleanup jobs completed successfully"
}
```

**Error Responses**:
- **500 Internal Server Error**: Cleanup failure

#### POST /batch/cleanup
Clean up old completed batch jobs.

**Success Response (200 OK)**:
```json
{
  "message": "Cleaned up 5 old batch jobs",
  "cleaned_count": 5,
  "max_age_hours": 24
}
```

---

## Example Responses

### Success Response (HTTP 200)

When all dependencies are healthy:

```json
{
  "ready": true,
  "timestamp": 1640995200.0,
  "response_time_ms": 125.5,
  "dependencies": {
    "minio": {
      "status": "healthy",
      "response_time_ms": 45.2,
      "storage_type": "MinIO",
      "endpoint": "http://minio:9000",
      "buckets": ["raw-images", "thumbnails"],
      "required_buckets_present": true,
      "dependency_name": "MinIO Object Storage",
      "dependency_description": "Object storage for images and thumbnails",
      "thumbnails_bucket_test": {
        "objects_listed": true,
        "object_count": 15,
        "error": null
      },
      "presigned_url_test": {
        "url_generated": true,
        "test_object": "tenant123/ab/cd/abcd1234_thumb.jpg",
        "error": null
      }
    },
    "qdrant": {
      "status": "healthy",
      "response_time_ms": 38.7,
      "vector_db_type": "Qdrant",
      "url": "http://qdrant:6333",
      "collections": ["faces_v1"],
      "target_collection": "faces_v1",
      "dependency_name": "Qdrant Vector Database",
      "dependency_description": "Vector database for face embeddings",
      "collection_validation": {
        "exists": true,
        "dimension": 512,
        "distance_metric": "Cosine",
        "points_count": 1250,
        "error": null
      }
    }
  },
  "summary": {
    "total_dependencies": 2,
    "healthy_dependencies": 2,
    "unhealthy_dependencies": 0,
    "unhealthy_dependency_names": []
  }
}
```

### Failure Response (HTTP 503)

When one or more dependencies are unhealthy:

```json
{
  "ready": false,
  "timestamp": 1640995200.0,
  "response_time_ms": 125.5,
  "dependencies": {
    "minio": {
      "status": "degraded",
      "response_time_ms": 45.2,
      "storage_type": "MinIO",
      "endpoint": "http://minio:9000",
      "buckets": ["raw-images", "thumbnails"],
      "required_buckets_present": true,
      "dependency_name": "MinIO Object Storage",
      "dependency_description": "Object storage for images and thumbnails",
      "thumbnails_bucket_test": {
        "objects_listed": false,
        "object_count": 0,
        "error": "Access denied"
      },
      "presigned_url_test": {
        "url_generated": false,
        "error": "Cannot generate presigned URL"
      }
    },
    "qdrant": {
      "status": "degraded",
      "response_time_ms": 38.7,
      "vector_db_type": "Qdrant",
      "url": "http://qdrant:6333",
      "collections": ["faces_v1"],
      "target_collection": "faces_v1",
      "dependency_name": "Qdrant Vector Database",
      "dependency_description": "Vector database for face embeddings",
      "collection_validation": {
        "exists": true,
        "dimension": 256,
        "distance_metric": "Cosine",
        "points_count": 1250,
        "error": "Expected dimension 512, got 256"
      }
    }
  },
  "summary": {
    "total_dependencies": 2,
    "healthy_dependencies": 0,
    "unhealthy_dependencies": 2,
    "unhealthy_dependency_names": ["minio", "qdrant"]
  },
  "errors": [
    "MinIO Object Storage: Access denied",
    "Qdrant Vector Database: Expected dimension 512, got 256"
  ],
  "error_summary": "2 dependency(ies) not ready: minio, qdrant"
}
```

### Connection Failure Response (HTTP 503)

When dependencies are unreachable:

```json
{
  "ready": false,
  "timestamp": 1640995200.0,
  "response_time_ms": 125.5,
  "dependencies": {
    "minio": {
      "status": "unhealthy",
      "error": "Connection refused",
      "dependency_name": "MinIO Object Storage",
      "dependency_description": "Object storage for images and thumbnails"
    },
    "qdrant": {
      "status": "unhealthy",
      "error": "Connection timeout",
      "dependency_name": "Qdrant Vector Database",
      "dependency_description": "Vector database for face embeddings"
    }
  },
  "summary": {
    "total_dependencies": 2,
    "healthy_dependencies": 0,
    "unhealthy_dependencies": 2,
    "unhealthy_dependency_names": ["minio", "qdrant"]
  },
  "errors": [
    "MinIO Object Storage: Connection refused",
    "Qdrant Vector Database: Connection timeout"
  ],
  "error_summary": "2 dependency(ies) not ready: minio, qdrant"
}
```

---

## Response Field Descriptions

### Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `ready` | boolean | Overall system readiness status |
| `timestamp` | number | Unix timestamp when check was performed |
| `response_time_ms` | number | Total time for all health checks in milliseconds |
| `dependencies` | object | Health status of each dependency |
| `summary` | object | Summary statistics of dependency health |
| `errors` | array | List of error messages (only present when not ready) |
| `error_summary` | string | Human-readable summary of issues (only present when not ready) |

### Dependency Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Health status: "healthy", "degraded", or "unhealthy" |
| `response_time_ms` | number | Individual dependency response time in milliseconds |
| `dependency_name` | string | Human-readable name of the dependency |
| `dependency_description` | string | Description of what the dependency provides |

### MinIO-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `storage_type` | string | Type of storage: "MinIO" or "AWS S3" |
| `endpoint` | string | Storage endpoint URL |
| `buckets` | array | List of available buckets |
| `required_buckets_present` | boolean | Whether required buckets exist |
| `thumbnails_bucket_test` | object | Results of thumbnails bucket listing test |
| `presigned_url_test` | object | Results of presigned URL generation test |

### Qdrant-Specific Fields

| Field | Type | Description |
|-------|------|-------------|
| `vector_db_type` | string | Type of vector database: "Qdrant" or "Pinecone" |
| `url` | string | Vector database URL |
| `collections` | array | List of available collections |
| `target_collection` | string | Name of the target collection (faces_v1) |
| `collection_validation` | object | Detailed collection validation results |

### Collection Validation Fields

| Field | Type | Description |
|-------|------|-------------|
| `exists` | boolean | Whether the collection exists |
| `dimension` | number | Vector dimension (must be 512) |
| `distance_metric` | string | Distance metric used (Cosine) |
| `points_count` | number | Number of vectors in the collection |
| `error` | string | Error message if validation fails |

---

## Health Check Details

### MinIO Storage Check

The MinIO health check performs the following validations:

1. **Connectivity**: Establishes connection to MinIO server
2. **Bucket Listing**: Lists all available buckets
3. **Required Buckets**: Verifies that `raw-images` and `thumbnails` buckets exist
4. **Object Listing**: Lists objects in the `thumbnails` bucket
5. **Presigned URL Generation**: Tests generating presigned URLs for objects

**Status Determination**:
- `healthy`: All checks pass
- `degraded`: Buckets exist but listing or presigned URL generation fails
- `unhealthy`: Connection or authentication failures

### Qdrant Vector Database Check

The Qdrant health check performs the following validations:

1. **Connectivity**: Establishes connection to Qdrant server
2. **Collection Listing**: Lists all available collections
3. **Target Collection**: Verifies that `faces_v1` collection exists
4. **Collection Details**: Retrieves detailed collection information
5. **Dimension Validation**: Confirms collection has exactly 512 dimensions
6. **Distance Metric**: Verifies Cosine distance metric is configured

**Status Determination**:
- `healthy`: Collection exists with correct 512 dimensions
- `degraded`: Collection exists but has wrong dimension or other issues
- `unhealthy`: Connection failures or collection doesn't exist

---

## How to Test

### Prerequisites
- API server running on `http://localhost:8000`
- Valid tenant ID (minimum 3 characters)
- Test image file (JPEG or PNG, max 10MB)

### Basic Authentication
All API endpoints (except health checks) require the `X-Tenant-ID` header:
```bash
-H "X-Tenant-ID: tenant123"
```

### Health Check Tests

#### Test System Readiness
```bash
curl -X GET "http://localhost:8000/ready" \
  -H "Accept: application/json" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: System ready, all dependencies healthy
- `503 Service Unavailable`: One or more dependencies unhealthy

### Face Operations Tests

#### Test Face Indexing
```bash
curl -X POST "http://{{base_url}}/index_face" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Face successfully indexed
- `400 Bad Request`: Invalid image format or no faces detected
- `413 Payload Too Large`: Image exceeds 10MB limit
- `429 Too Many Requests`: Rate limit exceeded

#### Test Face Search
```bash
curl -X POST "http://{{base_url}}/search_face?top_k=10&threshold=0.25" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Similar faces found
- `400 Bad Request`: Invalid image format, size, or no faces detected
- `413 Payload Too Large`: Image exceeds 10MB limit
- `429 Too Many Requests`: Rate limit exceeded

#### Test Face Comparison (Search Only)
```bash
curl -X POST "http://{{base_url}}/compare_face?top_k=5&threshold=0.5" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Face comparison completed
- `400 Bad Request`: Invalid image format, size, or no faces detected
- `413 Payload Too Large`: Image exceeds 10MB limit
- `429 Too Many Requests`: Rate limit exceeded

### Batch Processing Tests

#### Create Batch Job
```bash
curl -X POST "http://{{base_url}}/batch/index" \
  -H "X-Tenant-ID: tenant123" \
  -H "Content-Type: application/json" \
  -d '{
    "image_urls": [
      "https://example.com/image1.jpg",
      "https://example.com/image2.jpg",
      "https://example.com/image3.jpg"
    ],
    "metadata": {
      "source": "test_batch"
    }
  }' \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Batch job created successfully
- `400 Bad Request`: No URLs provided or batch too large (>100 images)
- `429 Too Many Requests`: Rate limit exceeded

#### Check Batch Status
```bash
curl -X GET "http://{{base_url}}/batch/batch-123e4567-e89b-12d3-a456-426614174000/status" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Batch status retrieved
- `404 Not Found`: Batch job not found
- `403 Forbidden`: Access denied to batch job

#### List Batch Jobs
```bash
curl -X GET "http://{{base_url}}/batch/list?limit=10&offset=0" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Batch list retrieved

#### Cancel Batch Job
```bash
curl -X DELETE "http://{{base_url}}/batch/batch-123e4567-e89b-12d3-a456-426614174000" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Batch job cancelled
- `404 Not Found`: Batch job not found
- `403 Forbidden`: Access denied to batch job
- `400 Bad Request`: Cannot cancel completed or failed batch job

### Webhook Tests

#### Register Webhook
```bash
curl -X POST "http://{{base_url}}/webhooks/register" \
  -H "X-Tenant-ID: tenant123" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/webhook",
    "events": ["face.indexed", "face.searched"],
    "secret": "webhook_secret_123",
    "timeout": 30,
    "retry_count": 3
  }' \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Webhook registered successfully
- `400 Bad Request`: Invalid webhook configuration or events

#### List Webhooks
```bash
curl -X GET "http://{{base_url}}/webhooks/list" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Webhook list retrieved

#### Test Webhook
```bash
curl -X POST "http://{{base_url}}/webhooks/test" \
  -H "X-Tenant-ID: tenant123" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/webhook"
  }' \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Webhook test completed

#### Get Webhook Statistics
```bash
curl -X GET "http://{{base_url}}/webhooks/stats" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Webhook statistics retrieved

#### Unregister Webhook
```bash
curl -X DELETE "http://{{base_url}}/webhooks/unregister?url=https://example.com/webhook" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Webhook unregistered successfully
- `404 Not Found`: Webhook not found

### Admin Operations Tests

#### Run Cleanup Jobs
```bash
curl -X POST "http://{{base_url}}/admin/cleanup" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Cleanup jobs completed successfully
- `500 Internal Server Error`: Cleanup failure

#### Clean Up Old Batch Jobs
```bash
curl -X POST "http://{{base_url}}/batch/cleanup?max_age_hours=24" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Codes:**
- `200 OK`: Old batch jobs cleaned up

### Error Testing

#### Test Missing Tenant ID
```bash
curl -X POST "http://{{base_url}}/index_face" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Code:**
- `400 Bad Request`: "X-Tenant-ID header is required"

#### Test Invalid Tenant ID
```bash
curl -X POST "http://{{base_url}}/index_face" \
  -H "X-Tenant-ID: ab" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Code:**
- `400 Bad Request`: "X-Tenant-ID must be at least 3 characters long"

#### Test Invalid Image Format
```bash
curl -X POST "http://{{base_url}}/index_face" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_document.pdf" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Code:**
- `400 Bad Request`: "Invalid image format. Please upload a JPG or PNG image."

#### Test Image Too Large
```bash
curl -X POST "http://{{base_url}}/index_face" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@large_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Status Code:**
- `413 Payload Too Large`: "Image size exceeds the maximum allowed size of 10MB."

### Rate Limiting Tests

#### Test Rate Limit Exceeded
```bash
# Send multiple requests quickly to trigger rate limiting
for i in {1..60}; do
  curl -X POST "http://{{base_url}}/index_face" \
    -H "X-Tenant-ID: tenant123" \
    -F "file=@test_image.jpg" \
    -w "HTTP Status: %{http_code}\n" \
    -o /dev/null -s
  sleep 0.1
done
```

**Expected Status Code:**
- `429 Too Many Requests`: "Too many requests"

### Performance Testing

#### Test Response Times
```bash
curl -X POST "http://{{base_url}}/search_face" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\nTime to First Byte: %{time_starttransfer}s\n"
```

**Expected Performance:**
- Response time: < 5 seconds (P95 threshold)
- Time to first byte: < 1 second

### Integration Testing Script

Create a comprehensive test script:

```bash
#!/bin/bash
# integration_test.sh

BASE_URL="http://localhost:8000"
TENANT_ID="tenant123"
TEST_IMAGE="test_image.jpg"

echo "Starting integration tests..."

# Test health check
echo "Testing health check..."
curl -f -s "$BASE_URL/ready" || { echo "Health check failed"; exit 1; }

# Test face indexing
echo "Testing face indexing..."
INDEX_RESPONSE=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/index_face" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -F "file=@$TEST_IMAGE")

INDEX_STATUS="${INDEX_RESPONSE: -3}"
if [ "$INDEX_STATUS" != "200" ]; then
  echo "Face indexing failed with status $INDEX_STATUS"
  exit 1
fi

echo "Integration tests completed successfully!"
```

---

## Usage Examples

### cURL Examples

**Check system readiness:**
```bash
curl -X GET "http://localhost:8000/ready" \
  -H "Accept: application/json"
```

**Check with verbose output:**
```bash
curl -X GET "http://localhost:8000/ready" \
  -H "Accept: application/json" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

### Monitoring Integration

**Load Balancer Health Check:**
```bash
# Returns 0 if healthy, 1 if unhealthy
curl -f -s "http://localhost:8000/ready" > /dev/null && echo "0" || echo "1"
```

**Kubernetes Readiness Probe:**
```yaml
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 3
```

**Docker Health Check:**
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/ready || exit 1
```

---

## Logging

The readiness endpoint logs the following information:

- **Start**: When readiness check begins
- **Component Status**: Individual health check results for each dependency
- **Duration**: Total time taken for all health checks
- **Final Result**: Overall readiness status with summary statistics
- **Errors**: Detailed error messages for troubleshooting

**Example Log Output:**
```
INFO: Starting readiness check for MinIO and Qdrant dependencies
INFO: MinIO health check completed: healthy (response_time: 45.2ms)
INFO: Qdrant health check completed: healthy (response_time: 38.7ms)
INFO: Readiness check completed: ready=True, duration=125.5ms, healthy_deps=2/2
INFO: System ready - all dependencies healthy
```

---

## Troubleshooting

### Common Issues

1. **MinIO Connection Refused**
   - Check if MinIO service is running
   - Verify endpoint URL and port
   - Check network connectivity

2. **Qdrant Connection Timeout**
   - Check if Qdrant service is running
   - Verify Qdrant URL and port
   - Check firewall settings

3. **Collection Dimension Mismatch**
   - Ensure `faces_v1` collection has 512 dimensions
   - Recreate collection if necessary
   - Check vector embedding configuration

4. **Bucket Access Denied**
   - Verify MinIO credentials
   - Check bucket permissions
   - Ensure required buckets exist

### Debug Commands

**Check MinIO connectivity:**
```bash
curl -I http://minio:9000/minio/health/live
```

**Check Qdrant connectivity:**
```bash
curl http://qdrant:6333/collections
```

**Check collection details:**
```bash
curl http://qdrant:6333/collections/faces_v1
```
