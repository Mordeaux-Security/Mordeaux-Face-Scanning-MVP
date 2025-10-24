# Integration Test Smoke Log Template

## Test Execution Summary

**Date**: [DATE]  
**Time**: [TIME] UTC  
**Tester**: [TESTER_NAME]  
**Environment**: [ENVIRONMENT]  
**API Version**: v1.0  

## Test Status

**Overall Result**: [PASS/FAIL/PARTIAL] - [DESCRIPTION]

**Service Status**: [HEALTHY/DEGRADED/UNHEALTHY] - API server accessible at `http://localhost:8000`

## Test Plan Executed

### 1. Health Check Tests

#### Basic Health Check
**Command**:
```bash
curl -f -s "http://localhost:8000/healthz"
```

**Expected Result**: 
- HTTP 200 OK with basic health status
- JSON response: `{"ok": true, "status": "healthy"}`

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Detailed Health Check
**Command**:
```bash
curl -f -s "http://localhost:8000/healthz/detailed"
```

**Expected Result**: 
- HTTP 200 OK with comprehensive system status
- JSON response with all service health details

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Readiness Check
**Command**:
```bash
curl -f -s "http://localhost:8000/ready"
```

**Expected Result**: 
- HTTP 200 OK with system readiness status
- JSON response with dependencies health check

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

### 2. Face Operations Tests

#### Face Indexing
**Command**:
```bash
curl -X POST "http://localhost:8000/api/index_face" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with indexed face count and thumbnail URL

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Face Search
**Command**:
```bash
curl -X POST "http://localhost:8000/api/search_face?top_k=10&threshold=0.25" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with similar faces found

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Face Comparison (Search Only)
**Command**:
```bash
curl -X POST "http://localhost:8000/api/compare_face?top_k=5&threshold=0.5" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with face comparison results

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

### 3. Batch Processing Tests

#### Create Batch Job
**Command**:
```bash
curl -X POST "http://localhost:8000/api/batch/index" \
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

**Expected Result**:
- HTTP 200 OK
- JSON response with batch job ID and status

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Check Batch Status
**Command**:
```bash
curl -X GET "http://localhost:8000/api/batch/{batch_id}/status" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with batch status details

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### List Batch Jobs
**Command**:
```bash
curl -X GET "http://localhost:8000/api/batch/list?limit=10&offset=0" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with batch list

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Cancel Batch Job
**Command**:
```bash
curl -X DELETE "http://localhost:8000/api/batch/{batch_id}" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with cancellation confirmation

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

### 4. Webhook Tests

#### Register Webhook
**Command**:
```bash
curl -X POST "http://localhost:8000/api/webhooks/register" \
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

**Expected Result**:
- HTTP 200 OK
- JSON response with webhook registration confirmation

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### List Webhooks
**Command**:
```bash
curl -X GET "http://localhost:8000/api/webhooks/list" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with webhook list

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Test Webhook
**Command**:
```bash
curl -X POST "http://localhost:8000/api/webhooks/test" \
  -H "X-Tenant-ID: tenant123" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com/webhook"
  }' \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with webhook test results

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Get Webhook Statistics
**Command**:
```bash
curl -X GET "http://localhost:8000/api/webhooks/stats" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with webhook statistics

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Unregister Webhook
**Command**:
```bash
curl -X DELETE "http://localhost:8000/api/webhooks/unregister?url=https://example.com/webhook" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with unregistration confirmation

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

### 5. Admin Operations Tests

#### Run Cleanup Jobs
**Command**:
```bash
curl -X POST "http://localhost:8000/api/admin/cleanup" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with cleanup results

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Clean Up Old Batch Jobs
**Command**:
```bash
curl -X POST "http://localhost:8000/api/batch/cleanup?max_age_hours=24" \
  -H "X-Tenant-ID: tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with cleanup count

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

### 6. Cache Management Tests

#### Get Cache Statistics
**Command**:
```bash
curl -X GET "http://localhost:8000/cache/stats" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with cache statistics

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Clear Tenant Cache
**Command**:
```bash
curl -X DELETE "http://localhost:8000/cache/tenant/tenant123" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with cache clear results

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

#### Clear All Cache
**Command**:
```bash
curl -X DELETE "http://localhost:8000/cache/all" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 200 OK
- JSON response with cache clear results

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

## Error Testing Results

### 1. Missing Tenant ID Test

**Command**:
```bash
curl -X POST "http://localhost:8000/api/index_face" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 400 Bad Request
- JSON error: `{"code": "missing_tenant_id", "message": "X-Tenant-ID header is required."}`

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

### 2. Invalid Tenant ID Test

**Command**:
```bash
curl -X POST "http://localhost:8000/api/index_face" \
  -H "X-Tenant-ID: ab" \
  -F "file=@test_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 400 Bad Request
- JSON error: `{"code": "invalid_tenant_id", "message": "X-Tenant-ID must be at least 3 characters long."}`

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

### 3. Invalid Image Format Test

**Command**:
```bash
curl -X POST "http://localhost:8000/api/index_face" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@test_document.pdf" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 400 Bad Request
- JSON error: `{"code": "invalid_image_format", "message": "Invalid image format. Please upload a JPG or PNG image."}`

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

### 4. Image Too Large Test

**Command**:
```bash
curl -X POST "http://localhost:8000/api/index_face" \
  -H "X-Tenant-ID: tenant123" \
  -F "file=@large_image.jpg" \
  -w "HTTP Status: %{http_code}\nTotal Time: %{time_total}s\n"
```

**Expected Result**:
- HTTP 413 Payload Too Large
- JSON error: `{"code": "image_too_large", "message": "Image size exceeds the maximum allowed size of 10MB."}`

**Actual Result**: 
- [STATUS_CODE] - [RESPONSE]

## Performance Metrics (Expected)

Based on API documentation and configuration:

- **Response Time**: < 5 seconds (P95 threshold)
- **Time to First Byte**: < 1 second
- **Rate Limiting**: 10 requests/second sustained, 50 burst capacity
- **Image Size Limit**: 10MB maximum
- **Batch Size Limit**: 100 images maximum

## Dependencies Status

**Required Services**:
- [ ] MinIO Storage (configured)
- [ ] Qdrant Vector Database (configured)
- [ ] Redis Cache (configured)
- [ ] PostgreSQL Database (configured)
- [ ] API Server (running)

## Test Environment Setup

**Prerequisites Check**:
- [ ] Test image files available
- [ ] Tenant ID configured (tenant123)
- [ ] API documentation complete
- [ ] Error codes documented
- [ ] API server running

## Test Data Requirements

**Test Images Required**:
- `test_image.jpg` - Valid image with detectable faces
- `large_image.jpg` - Image > 10MB for size limit testing
- `test_document.pdf` - Non-image file for format validation

**Test Tenant IDs**:
- `tenant123` - Valid tenant ID
- `ab` - Invalid tenant ID (too short)
- `invalid_tenant` - Non-existent tenant ID

### Integration Test Script

Create the following test script for automated testing:

```bash
#!/bin/bash
# integration_test.sh

BASE_URL="http://localhost:8000"
TENANT_ID="tenant123"
TEST_IMAGE="test_image.jpg"

echo "Starting integration tests..."

# Test health check
echo "Testing health check..."
HEALTH_RESPONSE=$(curl -s -w "%{http_code}" "$BASE_URL/ready")
HEALTH_STATUS="${HEALTH_RESPONSE: -3}"

if [ "$HEALTH_STATUS" != "200" ]; then
  echo "Health check failed with status $HEALTH_STATUS"
  exit 1
fi

echo "Health check passed"

# Test face indexing
echo "Testing face indexing..."
INDEX_RESPONSE=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/api/index_face" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -F "file=@$TEST_IMAGE")

INDEX_STATUS="${INDEX_RESPONSE: -3}"
if [ "$INDEX_STATUS" != "200" ]; then
  echo "Face indexing failed with status $INDEX_STATUS"
  exit 1
fi

echo "Face indexing passed"

# Test face search
echo "Testing face search..."
SEARCH_RESPONSE=$(curl -s -w "%{http_code}" -X POST "$BASE_URL/api/search_face?top_k=5" \
  -H "X-Tenant-ID: $TENANT_ID" \
  -F "file=@$TEST_IMAGE")

SEARCH_STATUS="${SEARCH_RESPONSE: -3}"
if [ "$SEARCH_STATUS" != "200" ]; then
  echo "Face search failed with status $SEARCH_STATUS"
  exit 1
fi

echo "Face search passed"

echo "Integration tests completed successfully!"
```

## Next Steps

1. **Start the API server** using the provided commands
2. **Run the integration test script** to verify all endpoints
3. **Monitor performance metrics** during testing
4. **Validate error handling** with invalid inputs
5. **Test rate limiting** with multiple concurrent requests

## Test Data

**Test Images Required**:
- `test_image.jpg` - Valid image with detectable faces
- `large_image.jpg` - Image > 10MB for size limit testing
- `test_document.pdf` - Non-image file for format validation

**Test Tenant IDs**:
- `tenant123` - Valid tenant ID
- `ab` - Invalid tenant ID (too short)
- `invalid_tenant` - Non-existent tenant ID

## Conclusion

The integration test framework is ready and documented. All test cases have been defined with expected results. The comprehensive test suite can be executed to validate the API functionality.

**Status**: ✅ **READY FOR EXECUTION** - Template complete with all current endpoints
