# API Error Documentation

This document describes the HTTP error codes and error responses used by the Mordeaux Face Scanning API.

## Error Response Format

All error responses follow a canonical JSON format:

```json
{
  "code": "error_code",
  "message": "Human-readable error message",
  "request_id": "unique-request-identifier"
}
```

### Fields

- **`code`**: Lowercase error identifier (e.g., `"rate_limited"`, `"invalid_image_format"`)
- **`message`**: Human-readable error description
- **`request_id`**: Unique UUID for request tracking and debugging
- **`details`**: Optional field containing additional error context (when applicable)

## Complete Error Code Reference

### Validation Errors (1000-1999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `invalid_image_format` | 400 | Invalid image format. Please upload a JPG or PNG image. | Unsupported file format uploaded |
| `image_too_large` | 413 | Image size exceeds the maximum allowed size of 10MB. | File upload exceeds 10MB limit |
| `empty_file` | 400 | No file provided or file is empty. | Empty or missing file upload |
| `missing_tenant_id` | 400 | X-Tenant-ID header is required. | Missing required tenant header |
| `invalid_tenant_id` | 400 | X-Tenant-ID must be at least 3 characters long. | Invalid tenant ID format |
| `invalid_batch_size` | 400 | Batch size cannot exceed 100 images. | Batch request too large |
| `no_image_urls` | 400 | No image URLs provided for batch processing. | Missing URLs in batch request |
| `invalid_webhook_events` | 400 | Invalid webhook events provided. | Invalid webhook configuration |
| `invalid_top_k` | 400 | top_k parameter must be between 1 and 50. | Invalid top_k parameter value |

### Authentication/Authorization Errors (2000-2999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `tenant_access_denied` | 403 | Access denied to this resource. | General tenant access denied |
| `batch_access_denied` | 403 | Access denied to this batch job. | Attempting to access another tenant's batch |

### Rate Limiting Errors (3000-3999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `rate_limit_exceeded` | 429 | Rate limit exceeded. Please try again later. | Legacy rate limit format |
| `rate_limited` | 429 | Too many requests | Rate limit exceeded (canonical format) |

### Resource Not Found Errors (4000-4999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `batch_not_found` | 404 | Batch job not found. | Batch ID doesn't exist |
| `image_not_found` | 404 | Image not found in storage. | Image file not found in storage |
| `webhook_not_found` | 404 | Webhook endpoint not found. | Webhook URL not registered |

### Storage Errors (5000-5999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `storage_upload_failed` | 500 | Failed to upload image to storage. | MinIO/S3 upload failure |
| `storage_download_failed` | 500 | Failed to download image from URL. | Image download from URL failed |
| `storage_connection_failed` | 503 | Failed to connect to storage service. | MinIO/S3 connection failure |

### Vector Database Errors (6000-6999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `vector_db_connection_failed` | 503 | Failed to connect to vector database. | Qdrant/Pinecone connection failure |
| `vector_db_upsert_failed` | 500 | Failed to store face embeddings in vector database. | Vector database write failure |
| `vector_db_search_failed` | 500 | Failed to search for similar faces. | Vector database search failure |

### Face Processing Errors (7000-7999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `face_detection_failed` | 500 | Failed to detect faces in the image. | Face detection model failure |
| `no_faces_detected` | 400 | No faces detected in the uploaded image. | Image contains no detectable faces |
| `face_embedding_failed` | 500 | Failed to generate face embeddings. | Face embedding model failure |
| `phash_computation_failed` | 500 | Failed to compute perceptual hash. | Perceptual hash computation failure |

### Batch Processing Errors (8000-8999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `batch_creation_failed` | 500 | Failed to create batch processing job. | Batch job creation failure |
| `batch_processing_failed` | 500 | Batch processing failed. | Batch job execution failure |
| `batch_cancellation_failed` | 400 | Cannot cancel completed or failed batch job. | Attempting to cancel non-cancelable job |

### Cache Errors (9000-9999)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `cache_operation_failed` | 500 | Cache operation failed. | Redis cache operation failure |

### System Errors (10000+)

| Error Code | HTTP Status | Message | When It Occurs |
|------------|-------------|---------|----------------|
| `internal_server_error` | 500 | An internal server error occurred. | Unhandled system error |
| `service_unavailable` | 503 | Service is temporarily unavailable. | System maintenance or overload |

## HTTP Error Codes

### 400 Bad Request

The request was invalid or cannot be served. The client should not repeat the request without modification.

#### Common 400 Errors

| Error Code | Description | When It Occurs |
|------------|-------------|----------------|
| `missing_tenant_id` | X-Tenant-ID header is required | Missing required tenant header |
| `invalid_tenant_id` | X-Tenant-ID must be at least 3 characters long | Invalid tenant ID format |
| `invalid_image_format` | Invalid image format. Please upload a JPG or PNG image | Unsupported file format |
| `empty_file` | No file provided or file is empty | Empty or missing file upload |
| `invalid_top_k` | top_k parameter must be between 1 and 50 | Invalid top_k parameter value |
| `invalid_batch_size` | Batch size cannot exceed 100 images | Batch request too large |
| `no_image_urls` | No image URLs provided for batch processing | Missing URLs in batch request |
| `invalid_webhook_events` | Invalid webhook events provided | Invalid webhook configuration |
| `no_faces_detected` | No faces detected in the uploaded image | Image contains no detectable faces |
| `batch_cancellation_failed` | Cannot cancel completed or failed batch job | Attempting to cancel non-cancelable job |

#### Example Response

```json
{
  "code": "invalid_image_format",
  "message": "Invalid image format. Please upload a JPG or PNG image.",
  "request_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

### 401 Unauthorized

**Note**: Currently not used by the API. All authentication is handled through tenant validation which returns 403 Forbidden instead.

### 403 Forbidden

The request was valid, but the server is refusing to process it due to insufficient permissions or access restrictions.

#### Common 403 Errors

| Error Code | Description | When It Occurs |
|------------|-------------|----------------|
| `tenant_access_denied` | Access denied to this resource | General tenant access denied |
| `batch_access_denied` | Access denied to this batch job | Attempting to access another tenant's batch |
| Tenant ID not in allow-list | Tenant ID not in allow-list | Tenant not in configured allow-list |
| Invalid tenant ID: tenant not found | Invalid tenant ID: tenant not found | Tenant doesn't exist in database |
| Tenant account is suspended | Tenant account is suspended | Tenant account is suspended |
| Tenant account has been deleted | Tenant account has been deleted | Tenant account is deleted |
| Tenant account is in [status] status | Tenant account is in [status] status | Tenant in invalid status |

#### Example Response

```json
{
  "code": "tenant_access_denied",
  "message": "Access denied to this resource",
  "request_id": "b2c3d4e5-f6g7-8901-bcde-f23456789012"
}
```

#### Tenant Status Errors

When a tenant account has an invalid status, the API returns specific error messages:

- **Suspended**: `"Tenant account is suspended"`
- **Deleted**: `"Tenant account has been deleted"`
- **Other Status**: `"Tenant account is in [status] status"`

### 413 Payload Too Large

The request entity is larger than the server is willing or able to process.

#### Common 413 Errors

| Error Code | Description | When It Occurs |
|------------|-------------|----------------|
| `image_too_large` | Image size exceeds the maximum allowed size of 10MB | File upload exceeds 10MB limit |
| Request size exceeds [X]MB limit | Request size exceeds [X]MB limit | Request body exceeds configured limit |

#### Example Response

```json
{
  "code": "image_too_large",
  "message": "Image size exceeds the maximum allowed size of 10MB.",
  "request_id": "c3d4e5f6-g7h8-9012-cdef-345678901234"
}
```

#### Upload Size Limits

- **Maximum image size**: 10MB per image
- **Validation**: Applied at both middleware and endpoint levels
- **Content-Type**: Only JPEG and PNG images are accepted

### 429 Too Many Requests

The user has sent too many requests in a given amount of time ("rate limiting").

#### Rate Limiting Configuration

The API implements sophisticated per-tenant rate limiting with burst capacity:

- **Sustained Rate**: 10 requests per second
- **Burst Capacity**: 50 requests
- **Legacy Limits**: 60 requests per minute, 1000 requests per hour

#### Common 429 Errors

| Error Code | Description | When It Occurs |
|------------|-------------|----------------|
| `rate_limited` | Too many requests | Rate limit exceeded (canonical format) |
| `rate_limit_exceeded` | Rate limit exceeded. Please try again later | Legacy rate limit format |

#### Example Response

```json
{
  "code": "rate_limited",
  "message": "Too many requests",
  "request_id": "d4e5f6g7-h8i9-0123-def0-456789012345",
  "details": {
    "sustained_rate_per_second": 10.0,
    "burst_capacity": 50,
    "per_minute_limit": 60,
    "per_hour_limit": 1000
  }
}
```

#### Rate Limiting Behavior

1. **Burst Handling**: First 50 requests are allowed immediately
2. **Sustained Rate**: After burst, limited to 10 requests per second
3. **Per-Tenant**: Each tenant has independent rate limits
4. **Token Bucket**: Uses token bucket algorithm for smooth rate limiting

## Error Handling Best Practices

### For API Clients

1. **Check Status Code**: Always check the HTTP status code first
2. **Read Error Code**: Use the `code` field for programmatic error handling
3. **Log Request ID**: Include `request_id` in error logs for debugging
4. **Retry Logic**: Implement appropriate retry logic for rate-limited requests
5. **User Messages**: Use the `message` field for user-facing error displays

### Common Client Patterns

#### Rate Limiting
```javascript
if (response.status === 429) {
  const error = await response.json();
  console.log(`Rate limited: ${error.message}`);
  // Implement exponential backoff retry
  setTimeout(() => retryRequest(), 1000);
}
```

#### Validation Errors
```javascript
if (response.status === 400) {
  const error = await response.json();
  if (error.code === 'invalid_image_format') {
    showUserMessage('Please upload a JPEG or PNG image');
  }
}
```

#### Access Denied
```javascript
if (response.status === 403) {
  const error = await response.json();
  if (error.code === 'tenant_access_denied') {
    // Redirect to authentication or show access denied message
    redirectToLogin();
  }
}
```

## Error Code Categories

Errors are categorized by their type for easier handling:

- **Validation**: Input validation failures (400)
- **Authorization**: Access control violations (403)
- **Rate Limit**: Rate limiting violations (429)
- **Payload**: Request size violations (413)
- **Not Found**: Resource not found (404)
- **System**: Internal server errors (500)

## Debugging

### Using Request IDs

Every error response includes a unique `request_id` that can be used to:

1. **Correlate with logs**: Find the corresponding server logs
2. **Support tickets**: Reference specific requests in support cases
3. **Monitoring**: Track error patterns and frequency

### Common Debugging Steps

1. **Check Request ID**: Use the `request_id` to find server logs
2. **Verify Headers**: Ensure `X-Tenant-ID` header is present and valid
3. **Check Rate Limits**: Verify you're not exceeding rate limits
4. **Validate Input**: Ensure request format and content are correct
5. **Check Tenant Status**: Verify tenant account is active

## Rate Limiting Details

### Token Bucket Algorithm

The API uses a token bucket algorithm for rate limiting:

1. **Bucket Capacity**: 50 tokens (burst capacity)
2. **Refill Rate**: 10 tokens per second (sustained rate)
3. **Per-Tenant**: Each tenant has an independent token bucket
4. **Redis Storage**: Token bucket state stored in Redis for distributed systems

### Rate Limit Headers

The API may include rate limiting information in response headers:

- `X-RateLimit-Limit`: Maximum requests allowed
- `X-RateLimit-Remaining`: Remaining requests in current window
- `X-RateLimit-Reset`: Time when rate limit resets

## Migration Notes

- **Canonical Format**: All errors now use the canonical format with lowercase codes
- **Request ID**: All error responses include request IDs for better debugging
- **Backward Compatibility**: Existing error handling continues to work
- **Enhanced Details**: Rate limiting errors include detailed configuration information

