## API Conventions

These conventions apply to all endpoints (Upload, Search-by-URL, Results, Face details, Webhook test). They align the MVP with MinIO + Qdrant while keeping storage/vector adapters swappable for S3/Pinecone.

### Base URL and Versioning
- **Base**: `/api`
- **Versioning**: path-based (`/api/v1/...`) once introduced. Current MVP may be unversioned; document as v1 behavior.

### Required Headers
- **X-Tenant-ID**: **REQUIRED** for all requests; scopes storage, vector, and metadata operations per tenant. Must be at least 3 characters long.
- **Content-Type**: `application/json` for JSON APIs; `multipart/form-data` for file uploads.
- **Accept**: `application/json` unless retrieving images via proxy.
- Optional: **Idempotency-Key** for safely retrying `POST` requests (if provided, server must de-duplicate per tenant + key within a 24h window).

### Tenant Scoping
- **All endpoints require X-Tenant-ID header**
- **Tenant isolation**: All data operations are scoped by tenant ID
- **Storage isolation**: Object keys are prefixed with tenant ID
- **Vector isolation**: Search results are filtered by tenant ID
- **Database isolation**: All records include tenant_id field

### Response Headers
- **X-Request-ID**: Correlates logs and audit records (echoes incoming `X-Request-ID` or generated server-side).
- **X-RateLimit-Limit**, **X-RateLimit-Remaining**, **X-RateLimit-Reset**: Per-tenant limits for the current window.

### Multi-Tenancy
- Every request must include `X-Tenant-ID`.
- All persisted objects (object store keys, vector ids/collections, DB rows) are namespaced or tagged by tenant and isolated in queries.

### Rate Limiting
- Per-tenant basic rate limit enforced at the edge or API layer.
- On limit exceeded, return `429 Too Many Requests` with a body using the shared error model and the rate-limit headers above.

### Request Size Caps
- Max upload size: **10MB per image**. Enforce pre-body (gateway) when possible.
- On violation, return `413 Payload Too Large` using the shared error model.

### Presigned URL TTL
- Any presigned upload/download URLs must have TTL ≤ **10 minutes**.
- The TTL is configurable via environment variables and should default to ≤10 minutes in all environments.

### Latency Objective
- End-to-end P95 (upload → results render) ≤ **5 seconds** on dev-sized data.
- Include timing metrics per request in observability, keyed by `X-Request-ID` and `X-Tenant-ID`.

### Error Model (Shared)
All errors return JSON with the following shape:

```json
{
  "error": {
    "code": "string",                // stable machine-readable code (e.g., "rate_limit_exceeded")
    "message": "string",             // human-readable summary
    "details": { "...": "..." },    // optional structured context (validation fields, etc.)
    "requestId": "uuid-or-string",   // mirrors X-Request-ID
    "retryable": false                 // guidance for clients
  }
}
```

HTTP status codes:
- 400 Bad Request: validation errors, unsupported content types.
- 401 Unauthorized / 403 Forbidden: authz/authn when enabled.
- 404 Not Found: missing resources/keys.
- 409 Conflict: duplicate idempotency key or state conflicts.
- 413 Payload Too Large: request exceeds size limits.
- 415 Unsupported Media Type: non-image uploads.
- 422 Unprocessable Entity: semantic validation failures.
- 429 Too Many Requests: per-tenant rate limit exceeded.
- 500 Internal Server Error: unexpected errors (never leak vendor details).

### Audit Logging
Record an audit entry for every request and response, including:
- `tenantId`, `requestId`, endpoint, action, status code, and timing.
- For uploads/searches: store only derived artifacts (e.g., content hash, pHash) and storage keys, not raw image content in logs.
- For results: include vector backend used and top-k metadata (scores, ids), not embeddings.

### Data Retention
- Crawled thumbnails: default retention **90 days**.
- User query images: default retention **≤ 24 hours**.
- Retention is configurable via environment variables; enforcement handled by lifecycle policies and/or periodic jobs.

### Vendor-Agnostic Adapters
- Storage and vector operations must route through adapter interfaces; no vendor-specific types or params surface in API requests/responses.
- Configuration via environment variables selects the implementation (e.g., MinIO vs S3, Qdrant vs Pinecone) without changing API behavior.

### Security and Validation
- Accept only `image/jpeg` and `image/png` for image uploads.
- Validate image presence and type before processing.
- Redact sensitive fields from logs and API errors.

### Observability
- Emit structured logs keyed by `X-Request-ID` and `X-Tenant-ID` for each step (ingest, process, search, respond).
- Expose minimal health/metrics endpoints (non-tenant) for liveness and performance dashboards.

---

With these conventions in place, the endpoint-specific docs (Upload, Search-by-URL, Results, Face details, Webhook test) will inherit consistent headers, limits, error format, and audit/retention guarantees.

## Current API Endpoints

### Face Indexing
- **POST** `/api/index_face` - Upload image, extract embeddings, and upsert to vector DB
- **POST** `/api/search_face` - Upload image, embed, and query top matches (also upserts)
- **POST** `/api/compare_face` - Search-only endpoint (no storage/DB persistence)

### Image Serving
- **GET** `/api/images/{bucket}/{key:path}` - Proxy endpoint to serve images from storage

### Admin
- **POST** `/api/admin/cleanup` - Run cleanup jobs manually

### Health & Configuration
- **GET** `/healthz` - Basic health check
- **GET** `/healthz/detailed` - Detailed health check with service status
- **GET** `/config` - Get current configuration (without sensitive data)

## Upload Flow

Two-step upload using presigned URLs to keep the API stateless and storage/vendor agnostic.

### 1) Request Presigned Upload URL
- **Method**: `POST`
- **Path**: `/api/v1/uploads/presign`
- **Headers**: `X-Tenant-ID`, optional `Idempotency-Key`, `Content-Type: application/json`
- **Body**:
```json
{
  "contentType": "image/jpeg",        
  "filename": "optional-original-name.jpg", 
  "sizeBytes": 123456                  
}
```
- **Validation**:
  - `contentType` must be `image/jpeg` or `image/png`.
  - `sizeBytes` ≤ 10,485,760 (10MB).
  - Enforce per-tenant rate limit.
- **Response** `201 Created`:
```json
{
  "uploadId": "uuid",
  "presignedUrl": "https://...",      
  "headers": { "Content-Type": "image/jpeg" },
  "expiresInSeconds": 600,              
  "objectKey": "tenants/{tenantId}/raw/{uploadId}",
  "thumbKey": "tenants/{tenantId}/thumb/{uploadId}" 
}
```
- **Notes**:
  - `expiresInSeconds` ≤ 600 (10 minutes) by policy; configurable via env.
  - No vendor-specific fields (e.g., S3 conditions) surface; clients only see URL + headers.
  - Audit log includes tenant, requestId, size, contentType, expiry, and keys (not the URL).

### 2) Client Uploads Directly to Object Store
- Use returned `presignedUrl` and `headers` to PUT the image bytes.
- Clients must complete within the TTL. Large uploads are rejected by the store after expiry.

### 3) Confirm Upload and Trigger Processing
- **Method**: `POST`
- **Path**: `/api/v1/uploads/{uploadId}/confirm`
- **Headers**: `X-Tenant-ID`, optional `Idempotency-Key`
- **Body** (optional metadata):
```json
{
  "labels": ["optional", "tags"],
  "source": "ui|crawler|api",
  "trace": {"clientTs": 0}
}
```
- **Behavior**:
  - Validates object exists at `objectKey` and is ≤ 10MB and of allowed type.
  - Generates thumbnail and stores at `thumbKey`.
  - Detects faces, computes embeddings and pHash, upserts to vector DB.
  - Returns a lightweight receipt clients can use to fetch results.
- **Response** `202 Accepted`:
```json
{
  "uploadId": "uuid",
  "thumbUrl": "/api/v1/images/{bucket}/{thumbKey}",
  "facesQueued": 1,
  "vectorBackend": "qdrant|pinecone"
}
```
- **Audit**: records request/response, processing timings, counts, vector backend, and keys (not embeddings or raw bytes).

### Retention and Lifecycle
- Raw uploads (user query images): retained ≤ 24 hours by default, configurable via env; lifecycle job or bucket policy purges.
- Thumbnails derived from uploads may follow the same ≤ 24 hours policy unless reconfigured; crawled thumbnails default to 90 days.
- Metadata and vector entries persist per product policy; ensure tenant tags for selective purge.

## Search-by-URL

Allows clients to submit a public image URL (or tenant-scoped signed URL) for on-demand search without persisting the raw image beyond processing.

### Request
- **Method**: `POST`
- **Path**: `/api/v1/search/url`
- **Headers**: `X-Tenant-ID`, optional `Idempotency-Key`, `Content-Type: application/json`
- **Body**:
```json
{
  "url": "https://example.com/image.jpg",
  "topK": 10,
  "filters": { "label": "optional" }
}
```
- **Validation**:
  - `url` must be `http(s)` and resolve within a short timeout (e.g., 3s connect, 5s total).
  - Response `Content-Type` must be `image/jpeg` or `image/png`.
  - Enforce a max content length of 10MB via `HEAD`/`Content-Length` and stream guard.
  - `topK` default 10, min 1, max 50.
- **Security**:
  - Block private/link-local IPs (SSRF guard).
  - Allowlist/denylist can be configured via env.

### Response
- **200 OK**:
```json
{
  "facesFound": 1,
  "phash": "abcd1234",
  "thumbUrl": "/api/v1/images/{bucket}/{thumbKey}",
  "results": [
    {
      "id": "face-id",
      "score": 0.92,
      "thumbUrl": "/api/v1/images/{bucket}/{thumbKey}",
      "metadata": { "bbox": [x,y,w,h], "detScore": 0.98 }
    }
  ],
  "vectorBackend": "qdrant|pinecone"
}
```
- **Notes**:
  - Raw bytes fetched from the URL are not retained after processing; only derived thumbnail and metadata may be stored per retention policy.
  - Per-tenant rate limits apply; on exceed return `429` with shared error model.
  - Audit includes URL domain, size, timings, counts, and vector backend; do not log full URL if sensitive.

## Results Retrieval

MVP supports synchronous search responses; this endpoint describes a polling-compatible shape if async jobs are introduced.

### Request
- **Method**: `GET`
- **Path**: `/api/v1/results/{uploadId}`
- **Headers**: `X-Tenant-ID`, `Accept: application/json`
- **Query**: `page` (default 1), `pageSize` (default 10, max 50)

### Response
- **200 OK** (when ready):
```json
{
  "status": "complete",
  "uploadId": "uuid",
  "summary": { "facesFound": 1, "phash": "abcd1234" },
  "results": [
    { "id": "face-id", "score": 0.92, "thumbUrl": "/api/v1/images/...", "metadata": { "bbox": [x,y,w,h] } }
  ],
  "page": 1,
  "pageSize": 10,
  "total": 1,
  "vectorBackend": "qdrant|pinecone"
}
```
- **202 Accepted** (if still processing):
```json
{ "status": "processing", "uploadId": "uuid", "estimateMs": 500 }
```
- **404 Not Found** if `uploadId` unknown for the tenant.

### Semantics and SLOs
- Target P95 end-to-end (upload → results render) ≤ 5s; clients may poll up to 5s with backoff.
- Results are read-only and stable once `complete`.
- Audit logs include each retrieval with page parameters and timing; do not include embeddings.

## Face Details

Retrieve detailed metadata for a specific face ID, including detection confidence, bounding box, and associated image references.

### Request
- **Method**: `GET`
- **Path**: `/api/v1/faces/{faceId}`
- **Headers**: `X-Tenant-ID`, `Accept: application/json`

### Response
- **200 OK**:
```json
{
  "id": "face-id",
  "detScore": 0.98,
  "bbox": [x, y, width, height],
  "phash": "abcd1234",
  "rawUrl": "/api/v1/images/{bucket}/{rawKey}",
  "thumbUrl": "/api/v1/images/{bucket}/{thumbKey}",
  "createdAt": "2024-01-01T00:00:00Z",
  "labels": ["optional", "tags"],
  "source": "ui|crawler|api"
}
```

### Access Control and Redactions
- Face details are tenant-scoped; requests for other tenants' faces return `404`.
- Sensitive fields (e.g., raw embeddings) are never exposed in API responses.
- Audit logs include face ID access but not embedding vectors.

### Retention
- Face metadata follows the same retention policy as associated images.
- When images are purged, corresponding face records should be cleaned up to maintain referential integrity.

## Webhook Test

Test endpoint for webhook delivery and payload validation during development and integration.

### Request
- **Method**: `POST`
- **Path**: `/api/v1/webhooks/test`
- **Headers**: `X-Tenant-ID`, `Content-Type: application/json`
- **Body**:
```json
{
  "webhookUrl": "https://example.com/webhook",
  "eventType": "face.detected|search.completed",
  "payload": { "custom": "data" }
}
```

### Response
- **200 OK**:
```json
{
  "delivered": true,
  "statusCode": 200,
  "responseTimeMs": 150,
  "webhookId": "uuid"
}
```

### Webhook Payloads
Standard webhook payloads include:
```json
{
  "event": "face.detected",
  "tenantId": "tenant-123",
  "timestamp": "2024-01-01T00:00:00Z",
  "data": {
    "faceId": "face-id",
    "uploadId": "upload-id",
    "detScore": 0.98,
    "thumbUrl": "/api/v1/images/{bucket}/{thumbKey}"
  }
}
```

### Retry and Signatures
- Webhooks are retried up to 3 times with exponential backoff (1s, 2s, 4s).
- Optional signature header `X-Webhook-Signature` using HMAC-SHA256 for payload verification.
- Failed deliveries are logged with error details but not retried beyond the limit.

## Error Model

All endpoints use a consistent error response format for better client handling and debugging.

### Error Response Format
```json
{
  "error": {
    "code": "rate_limit_exceeded",
    "message": "Request rate limit exceeded for tenant",
    "details": {
      "limit": 100,
      "window": "1m",
      "retryAfter": 30
    },
    "requestId": "req-123",
    "retryable": true
  }
}
```

### Common Error Codes
- `validation_failed`: Request validation errors (400)
- `rate_limit_exceeded`: Per-tenant rate limit hit (429)
- `payload_too_large`: Request exceeds 10MB limit (413)
- `unsupported_media_type`: Non-image content type (415)
- `tenant_not_found`: Invalid or missing X-Tenant-ID (403)
- `resource_not_found`: Face/upload ID not found (404)
- `processing_failed`: Face detection/embedding errors (422)
- `internal_error`: Unexpected server errors (500)

### Rate Limit Response
When rate limits are exceeded, include additional headers:
```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1640995200
```

## cURL Examples

### Current API Endpoints

#### Index Face
```bash
curl -X POST "http://localhost:8000/api/index_face" \
  -H "X-Tenant-ID: tenant-123" \
  -F "file=@image.jpg"
```

#### Search Face
```bash
curl -X POST "http://localhost:8000/api/search_face" \
  -H "X-Tenant-ID: tenant-123" \
  -F "file=@image.jpg"
```

#### Compare Face (Search Only)
```bash
curl -X POST "http://localhost:8000/api/compare_face" \
  -H "X-Tenant-ID: tenant-123" \
  -F "file=@image.jpg"
```

#### Health Check
```bash
curl -X GET "http://localhost:8000/healthz"
curl -X GET "http://localhost:8000/healthz/detailed"
```

#### Configuration
```bash
curl -X GET "http://localhost:8000/config"
```

#### Admin Cleanup
```bash
curl -X POST "http://localhost:8000/api/admin/cleanup" \
  -H "X-Tenant-ID: admin"
```

### Future API Endpoints (Planned)

#### Upload Flow
```bash
# 1. Request presigned URL
curl -X POST "https://api.example.com/api/v1/uploads/presign" \
  -H "X-Tenant-ID: tenant-123" \
  -H "Content-Type: application/json" \
  -d '{"contentType": "image/jpeg", "sizeBytes": 1024000}'

# 2. Upload to presigned URL
curl -X PUT "$PRESIGNED_URL" \
  -H "Content-Type: image/jpeg" \
  --data-binary @image.jpg

# 3. Confirm upload
curl -X POST "https://api.example.com/api/v1/uploads/$UPLOAD_ID/confirm" \
  -H "X-Tenant-ID: tenant-123"
```

#### Search by URL
```bash
curl -X POST "https://api.example.com/api/v1/search/url" \
  -H "X-Tenant-ID: tenant-123" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/image.jpg", "topK": 10}'
```

#### Get Face Details
```bash
curl -X GET "https://api.example.com/api/v1/faces/face-123" \
  -H "X-Tenant-ID: tenant-123"
```

#### Test Webhook
```bash
curl -X POST "https://api.example.com/api/v1/webhooks/test" \
  -H "X-Tenant-ID: tenant-123" \
  -H "Content-Type: application/json" \
  -d '{"webhookUrl": "https://example.com/webhook", "eventType": "face.detected"}'
```

## Adapter Configuration

The API remains vendor-agnostic through adapter interfaces. Configuration is environment-driven:

### Environment Variables
```bash
# Storage Adapter
STORAGE_ADAPTER=minio|s3
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
S3_BUCKET=face-scanning
S3_REGION=us-east-1

# Vector Adapter  
VECTOR_ADAPTER=qdrant|pinecone
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=faces
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-west1-gcp

# API Configuration
PRESIGNED_URL_TTL_SECONDS=600
MAX_UPLOAD_SIZE_BYTES=10485760
RATE_LIMIT_PER_TENANT=100
RATE_LIMIT_WINDOW_SECONDS=60

# Retention Policies
CRAWLED_THUMB_RETENTION_DAYS=90
USER_QUERY_RETENTION_HOURS=24
```

### Adapter Interfaces
- **Storage**: `get_presigned_url()`, `save_object()`, `get_object()`, `delete_object()`
- **Vector**: `upsert_embeddings()`, `search_similar()`, `delete_by_ids()`
- **Face**: `detect_and_embed()`, `compute_phash()`

No vendor-specific types or parameters surface in API requests/responses.

## Audit Logging

Every API request and response generates structured audit logs for compliance and debugging.

### Log Format
```json
{
  "timestamp": "2024-01-01T00:00:00Z",
  "level": "INFO",
  "requestId": "req-123",
  "tenantId": "tenant-123",
  "endpoint": "POST /api/v1/search/url",
  "statusCode": 200,
  "durationMs": 150,
  "userAgent": "curl/7.68.0",
  "ipAddress": "192.168.1.1",
  "metadata": {
    "facesFound": 1,
    "vectorBackend": "qdrant",
    "contentSize": 1024000
  }
}
```

### Sensitive Data Handling
- **Never log**: Raw image bytes, embeddings, presigned URLs, API keys
- **Log safely**: Content hashes (pHash), object keys, tenant IDs, request IDs
- **Redact**: Full URLs (log domain only), personal identifiers in metadata

### Retention and Access
- Audit logs retained per compliance requirements (default 1 year)
- Logs are tenant-scoped for access control
- Structured format enables automated analysis and alerting

