# Phase 3 — Mock Server + Fixtures ✅

## Summary

A fully functional mock server with comprehensive fixture sets has been successfully implemented for Phase 3, decoupling frontend development from backend readiness.

**Status**: Complete and Ready for Use  
**Date**: 2025-11-14  
**Location**: `mock-server/`

---

## Goal Achievement

✅ **Goal**: Decouple UX from backend readiness

The mock server provides a complete API implementation that matches exact Phase 2 contracts, allowing frontend developers to work independently of backend development.

---

## Deliverables

### 1. Simple Mock Endpoints ✅

**Technology**: FastAPI  
**File**: `mock-server/app.py`  
**Port**: `http://localhost:8000`

**Implemented Endpoints**:

| Endpoint | Method | Description | Phase 2 Contract Match |
|----------|--------|-------------|----------------------|
| `/api/v1/health` | GET | Health check | ✅ Exact match |
| `/api/v1/search` | POST | Face similarity search | ✅ Exact match |
| `/api/v1/faces/{face_id}` | GET | Get face by ID | ✅ Exact match |
| `/api/v1/stats` | GET | Pipeline statistics | ✅ Exact match |
| `/mock/fixtures` | GET | List available fixtures | Mock-specific |
| `/mock/config` | POST | Update mock configuration | Mock-specific |

**Features**:
- ✅ CORS enabled for frontend development
- ✅ Configurable latency simulation (50-300ms)
- ✅ Random error injection (configurable rate)
- ✅ Auto-generated API docs at `/docs`
- ✅ Supports multipart/form-data and JSON requests
- ✅ Returns presigned URL format matching backend

### 2. Fixture Sets ✅

**File**: `mock-server/fixtures.py`

#### Tiny Fixture (10 results)
- **Purpose**: Testing best matches, quick UI development
- **Score Range**: 0.86 - 0.95
- **Mean Score**: 0.905
- **Sites**: 5 unique sites
- **Broken URLs**: 0
- **Use Case**: Testing UI with minimal data, debugging

```json
{
  "name": "tiny",
  "count": 10,
  "score_range": "0.8600 - 0.9500",
  "broken_urls": 0
}
```

#### Medium Fixture (200 results) ✅
- **Purpose**: Realistic development workload
- **Score Range**: 0.41 - 0.95
- **Mean Score**: 0.674
- **Score Distribution**: Normal (bell curve centered at 0.70)
  - Top 20: Very high scores (0.85-0.95)
  - Next 60: High scores (0.70-0.85)
  - Next 80: Medium scores (0.55-0.70)
  - Last 40: Lower scores (0.40-0.55)
- **Sites**: 6 unique sites
- **Broken URLs**: 5 (for error testing)
- **Use Case**: Primary development fixture, pagination testing

```json
{
  "name": "medium",
  "count": 200,
  "score_range": "0.4078 - 0.9500",
  "broken_urls": 5
}
```

#### Large Fixture (2000 results) ✅
- **Purpose**: Stress testing, performance validation
- **Score Range**: 0.30 - 0.98
- **Mean Score**: 0.597
- **Score Distribution**: Wide distribution
  - Top 50: Excellent matches (0.90-0.98)
  - Next 250: Very good matches (0.75-0.90)
  - Next 700: Good matches (0.60-0.75)
  - Next 600: Fair matches (0.45-0.60)
  - Last 400: Poor matches (0.30-0.45)
- **Sites**: 6 unique sites
- **Broken URLs**: 20 (for error testing)
- **Use Case**: Virtual scrolling, pagination stress testing, performance benchmarks

```json
{
  "name": "large",
  "count": 2000,
  "score_range": "0.3006 - 0.9800",
  "broken_urls": 20
}
```

#### Edge Cases Fixture (15 results) ✅
- **Purpose**: Testing edge cases and boundary conditions
- **Score Range**: 0.01 - 1.00
- **Special Cases**:
  - Perfect match (score 1.0)
  - Threshold boundaries (0.74, 0.75, 0.76)
  - Very low scores (0.01, 0.05, 0.15, 0.25)
- **Sites**: 5 unique sites
- **Broken URLs**: 0
- **Use Case**: Testing threshold filtering, edge case handling

```json
{
  "name": "edge_cases",
  "count": 15,
  "score_range": "0.0100 - 1.0000",
  "special_cases": ["perfect_match", "threshold_boundaries", "very_low_scores"]
}
```

#### Errors Fixture (20 results) ✅
- **Purpose**: Error handling and broken URL testing
- **Score Range**: 0.52 - 0.90
- **Mean Score**: 0.710
- **Sites**: 6 unique sites
- **Broken URLs**: 20 (100% - all URLs are intentionally broken)
- **Broken URL Types**:
  - 404 errors (`https://broken-cdn.example.com/404/...`)
  - Expired presigned URLs (`https://expired.minio.local/...?expired=true`)
  - Invalid domains (`https://invalid-domain-12345.local/...`)
- **Use Case**: Testing image load error handling, fallback UI

```json
{
  "name": "errors",
  "count": 20,
  "score_range": "0.5200 - 0.9000",
  "broken_urls": 20
}
```

### 3. Realistic Data Variety ✅

**Bounding Box Variety**:
- Different positions: centered, off-center, edge cases
- Different sizes: 80px - 400px (realistic face detection sizes)
- Aspect ratios: 0.8 - 1.2 (roughly square, as faces are)
- Random positioning within 1024x1024 image bounds

**Score Distributions**:
- Tiny: High scores only (realistic "best matches" scenario)
- Medium: Normal distribution (realistic search results)
- Large: Wide distribution (stress testing, edge cases)
- Edge Cases: Specific boundary values
- Errors: Mid-to-high scores with broken URLs

**Other Realistic Elements**:
- ✅ Timestamps spread over 30 days
- ✅ Quality scores (0.3-1.0, 80% good quality, 20% poor)
- ✅ Perceptual hashes (16-character hex strings)
- ✅ Multiple sites (6 unique mock sites)
- ✅ Presigned URL format matching MinIO/S3
- ✅ Face IDs in UUID-like format (`face-{16-char-hex}`)

---

## Acceptance Criteria Verification

### ✅ Frontend Can Fully Develop Against Mocks

**API Contract Compliance**:
- All endpoints return exact Phase 2 contract structure
- SearchResponse schema matches `api/openapi.yaml`
- SearchHit schema matches Phase 2 specification
- FaceDetailResponse matches spec
- ErrorResponse follows Phase 2 format

**Example Response Structure**:

```json
{
  "query": {
    "tenant_id": "demo-tenant",
    "search_mode": "image",
    "top_k": 50,
    "threshold": 0.75
  },
  "hits": [
    {
      "face_id": "face-4a1095eccf1e7fb5",
      "score": 0.95,
      "payload": {
        "site": "sample-images.org",
        "url": "https://sample-images.org/images/photo-8135.jpg",
        "ts": "2025-11-13T14:49:42Z",
        "bbox": [61, 579, 103, 118],
        "p_hash": "ffc10e011cfd6bb4",
        "quality": 0.787
      },
      "thumb_url": "https://minio.example.com/thumbnails/demo-tenant/face-4a1095eccf1e7fb5_thumb.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=minioadmin&X-Amz-Date=1763079342&X-Amz-Expires=1763079942&X-Amz-SignedHeaders=host&X-Amz-Signature=30efea876430bda517317e680913dc28"
    }
  ],
  "count": 1
}
```

**Frontend Development Features**:
- ✅ No backend dependencies required
- ✅ Instant response times (or configurable latency)
- ✅ All fixtures available via query parameter
- ✅ Error scenarios for testing error handling
- ✅ Configurable behavior via `/mock/config`

### ✅ Fixtures Cover Edge Cases

**Edge Cases Covered**:

1. **Perfect Match** ✅
   - Score: 1.0
   - Tests maximum similarity display

2. **Threshold Boundaries** ✅
   - Scores: 0.74, 0.75, 0.76
   - Tests filtering logic around common threshold

3. **Very Low Scores** ✅
   - Scores: 0.01, 0.05, 0.15, 0.25
   - Tests UI with poor matches

4. **Empty Results** ✅
   - Error scenario: `?error_scenario=no_results`
   - Returns 0 hits

5. **Broken Thumbnail URLs** ✅
   - Errors fixture: 100% broken URLs
   - Medium: 5 broken URLs
   - Large: 20 broken URLs
   - Tests image loading error handling

6. **Error Responses** ✅
   - 400: Missing tenant ID
   - 404: Face not found
   - 500: Internal server error (simulated)
   - 429: Rate limit exceeded (simulated)
   - 504: Gateway timeout (simulated)

7. **Extreme Dataset Sizes** ✅
   - Tiny: 10 results (minimal UI)
   - Large: 2000 results (pagination stress test)

8. **Score Filtering** ✅
   - Full range: 0.01 - 1.0
   - Tests threshold slider at any value

9. **Site Variety** ✅
   - 6 different mock sites
   - Tests site filtering dropdown

10. **Timestamp Distribution** ✅
    - Spread over 30 days
    - Tests date-based features

---

## File Structure

```
mock-server/
├── app.py                    # FastAPI server (292 lines)
├── fixtures.py               # Fixture generator (413 lines)
├── requirements.txt          # Dependencies (3 packages)
├── README.md                 # User documentation (434 lines)
├── test_mock_server.py       # Test suite (538 lines)
├── start.ps1                 # Windows start script
└── start.sh                  # Linux/Mac start script
```

**Total**: 7 files, ~1,677 lines of code and documentation

---

## Quick Start Guide

### 1. Install Dependencies

```bash
cd mock-server
pip install -r requirements.txt
```

**Dependencies**:
- `fastapi==0.115.0` - Web framework
- `uvicorn[standard]==0.30.6` - ASGI server
- `python-multipart==0.0.9` - File upload support

### 2. Start Server

**Windows**:
```powershell
cd mock-server
.\start.ps1
```

**Linux/Mac**:
```bash
cd mock-server
./start.sh
```

**Or directly**:
```bash
cd mock-server
python app.py
```

Server runs on: `http://localhost:8000`

API docs: `http://localhost:8000/docs`

### 3. Test Endpoints

**Health Check**:
```bash
curl http://localhost:8000/api/v1/health
```

**Search (default medium fixture)**:
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg" \
  -F "top_k=10" \
  -F "threshold=0.75"
```

**Search with specific fixture**:
```bash
curl -X POST "http://localhost:8000/api/v1/search?fixture=tiny" \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg"
```

**List available fixtures**:
```bash
curl http://localhost:8000/mock/fixtures
```

---

## Usage Examples

### Frontend Integration

Update your frontend API configuration:

```javascript
// config.js
const API_BASE_URL = 'http://localhost:8000';

// Search with fixture selection
const searchFaces = async (imageFile, fixture = 'medium') => {
  const formData = new FormData();
  formData.append('image', imageFile);
  formData.append('tenant_id', 'demo-tenant');
  formData.append('top_k', 50);
  formData.append('threshold', 0.75);
  
  const response = await fetch(
    `${API_BASE_URL}/api/v1/search?fixture=${fixture}`,
    {
      method: 'POST',
      headers: {
        'X-Tenant-ID': 'demo-tenant'
      },
      body: formData
    }
  );
  
  return response.json();
};
```

### Testing Scenarios

**1. Basic Functionality**:
```bash
# Default search (medium fixture)
curl -X POST "http://localhost:8000/api/v1/search" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg"
```

**2. Pagination Testing**:
```bash
# Large dataset for pagination
curl -X POST "http://localhost:8000/api/v1/search?fixture=large" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg" -F "top_k=50"
```

**3. Filtering Testing**:
```bash
# Test threshold filtering
curl -X POST "http://localhost:8000/api/v1/search?fixture=medium" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg" \
  -F "threshold=0.80"  # Should return fewer results
```

**4. Error Handling**:
```bash
# No results
curl -X POST "http://localhost:8000/api/v1/search?error_scenario=no_results" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg"

# Server error
curl -X POST "http://localhost:8000/api/v1/search?error_scenario=api_error" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg"

# Broken URLs
curl -X POST "http://localhost:8000/api/v1/search?fixture=errors" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg"
```

**5. Performance Testing**:
```bash
# Disable latency for faster testing
curl -X POST "http://localhost:8000/mock/config" \
  -H "Content-Type: application/json" \
  -d '{"simulate_latency": false}'

# Test with large dataset
curl -X POST "http://localhost:8000/api/v1/search?fixture=large" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg" -F "top_k=50"
```

---

## Configuration Options

### Mock Server Configuration

The mock server can be configured via the `/mock/config` endpoint:

```bash
curl -X POST "http://localhost:8000/mock/config" \
  -H "Content-Type: application/json" \
  -d '{
    "default_fixture": "large",
    "simulate_latency": true,
    "min_latency_ms": 100,
    "max_latency_ms": 500,
    "error_rate": 0.05
  }'
```

**Configuration Options**:
- `default_fixture`: Default fixture to use (default: `medium`)
- `simulate_latency`: Enable latency simulation (default: `true`)
- `min_latency_ms`: Minimum latency in ms (default: `50`)
- `max_latency_ms`: Maximum latency in ms (default: `300`)
- `error_rate`: Probability of random errors 0-1 (default: `0.0`)

---

## Test Suite

**File**: `mock-server/test_mock_server.py`

**Test Coverage**:

1. ✅ **Health Check** - Verify server is running
2. ✅ **List Fixtures** - Verify all fixtures available
3. ✅ **Basic Search** - Verify search endpoint works
4. ✅ **Fixture Sizes** - Verify tiny (10), medium (200), large (2000)
5. ✅ **Threshold Filtering** - Verify score filtering works
6. ✅ **Error Scenarios** - Verify error responses
7. ✅ **Broken URLs** - Verify error fixture has broken URLs
8. ✅ **Get Face by ID** - Verify face retrieval works
9. ✅ **Latency Simulation** - Verify configurable latency

**Run Tests**:
```bash
cd mock-server
python test_mock_server.py
```

**Expected Output**:
```
================================================================================
Mock Server Test Suite - Phase 3
================================================================================

✓ PASS - Health Check
✓ PASS - List Fixtures
✓ PASS - Basic Search
✓ PASS - Fixture Sizes
✓ PASS - Threshold Filtering
✓ PASS - Error Scenarios
✓ PASS - Broken URLs
✓ PASS - Get Face by ID
✓ PASS - Latency Simulation

Total: 9/9 tests passed
🎉 All tests passed!
```

---

## Phase 2 Contract Compliance

### SearchResponse Contract ✅

```json
{
  "query": {
    "tenant_id": "string",
    "search_mode": "image" | "vector",
    "top_k": 10,
    "threshold": 0.75
  },
  "hits": [
    {
      "face_id": "string",
      "score": 0.95,
      "payload": {
        "site": "string",
        "url": "string",
        "ts": "2024-01-01T12:00:00Z",
        "bbox": [100, 150, 200, 250],
        "p_hash": "string",
        "quality": 0.92
      },
      "thumb_url": "string"
    }
  ],
  "count": 1
}
```

**Compliance**: ✅ Exact match with `api/openapi.yaml` specification

### SearchHit Contract ✅

All required fields present:
- ✅ `face_id` (string, UUID-like format)
- ✅ `score` (float, 0.0-1.0)
- ✅ `payload` (object with site, url, ts, bbox, p_hash, quality)
- ✅ `thumb_url` (string, presigned URL format)

### FaceDetailResponse Contract ✅

```json
{
  "face_id": "string",
  "payload": { ... },
  "thumb_url": "string"
}
```

**Compliance**: ✅ Exact match

### ErrorResponse Contract ✅

```json
{
  "error": "string",
  "message": "string",
  "timestamp": "2024-01-01T12:00:00Z",
  "details": {}
}
```

**Compliance**: ✅ Standard FastAPI error format

---

## Benefits for Frontend Development

1. **Independent Development** ✅
   - No backend dependencies
   - No database required
   - No authentication setup needed

2. **Fast Iteration** ✅
   - Instant response times (or configurable)
   - No network latency
   - No rate limiting

3. **Comprehensive Testing** ✅
   - All edge cases covered
   - Error scenarios available
   - Multiple dataset sizes

4. **Realistic Data** ✅
   - Score distributions match real-world patterns
   - BBox variety matches real detections
   - Timestamps and quality scores realistic

5. **Easy Configuration** ✅
   - Switch fixtures via query parameter
   - Trigger errors on demand
   - Adjust latency for testing

6. **Documentation** ✅
   - Auto-generated API docs at `/docs`
   - README with examples
   - Test suite as examples

---

## Next Steps

### Switch to Real Backend

When backend is ready:

1. Update frontend API base URL:
   ```javascript
   // From:
   const API_BASE_URL = 'http://localhost:8000';  // Mock server
   
   // To:
   const API_BASE_URL = 'http://localhost:8001';  // Real backend
   ```

2. Remove fixture query parameters (not supported by real backend)

3. Test with real data

4. Keep mock server for testing and development

### Mock Server Maintenance

The mock server can continue to be used for:
- Unit testing frontend
- Integration testing
- Demo purposes
- Offline development
- CI/CD pipeline testing

---

## Troubleshooting

### Server Won't Start

```bash
# Check port availability
netstat -ano | findstr :8000   # Windows
lsof -i :8000                  # Linux/Mac

# Use different port
uvicorn app:app --port 8001
```

### Dependencies Missing

```bash
# Install dependencies
cd mock-server
pip install -r requirements.txt
```

### CORS Errors

CORS is enabled for all origins in development. If issues persist, check browser console for specific errors.

### Fixture Not Found

```bash
# List available fixtures
curl http://localhost:8000/mock/fixtures

# Available: tiny, medium, large, edge_cases, errors
```

### Slow Responses

```bash
# Disable latency simulation
curl -X POST "http://localhost:8000/mock/config" \
  -H "Content-Type: application/json" \
  -d '{"simulate_latency": false}'
```

---

## Comparison with Phase 2

| Aspect | Phase 2 (Real Backend) | Phase 3 (Mock Server) |
|--------|----------------------|----------------------|
| **Purpose** | Production API | Development/Testing |
| **Dependencies** | Database, Redis, Qdrant | None |
| **Setup Time** | ~30 minutes | ~1 minute |
| **Response Time** | 100-500ms | Configurable (0-300ms) |
| **Data** | Real search results | Generated fixtures |
| **Flexibility** | Limited by real data | Configurable scenarios |
| **Error Testing** | Limited | Full control |
| **Fixtures** | No | Yes (5 sets) |
| **Contract Match** | Authoritative | Exact match |

---

## Success Metrics

### Phase 3 Acceptance Criteria ✅

- [x] Frontend can fully develop against mocks
  - All endpoints implemented
  - Exact contract compliance
  - Multiple realistic datasets
  
- [x] Fixtures cover edge cases
  - Perfect matches (score 1.0)
  - Threshold boundaries (0.74-0.76)
  - Empty results
  - Very low scores
  - Broken URLs (5, 20, 100%)
  - Error responses (400, 404, 429, 500, 504)

### Additional Achievements ✅

- [x] 5 comprehensive fixture sets
- [x] Realistic score distributions
- [x] BBox variety
- [x] Multiple sites (6)
- [x] Timestamp variety (30 days)
- [x] Quality score realism (80% good, 20% poor)
- [x] Configurable latency simulation
- [x] Random error injection
- [x] Auto-generated API docs
- [x] Comprehensive test suite (9 tests)
- [x] Cross-platform support (Windows/Linux/Mac)
- [x] Detailed documentation

---

## Conclusion

**Phase 3 Status**: ✅ Complete

The mock server successfully decouples frontend development from backend readiness by providing:

1. ✅ **Exact Phase 2 API contracts** - All endpoints match specification
2. ✅ **Comprehensive fixtures** - Tiny (10), medium (200), large (2000) with realistic distributions
3. ✅ **Edge case coverage** - Perfect matches, boundaries, errors, broken URLs
4. ✅ **Developer-friendly** - Easy setup, configuration, testing
5. ✅ **Production-quality** - FastAPI, proper error handling, CORS support

**Frontend teams can now**:
- Develop independently of backend
- Test all scenarios (success, errors, edge cases)
- Iterate quickly with instant responses
- Validate UI with realistic data
- Switch seamlessly to real backend when ready

**Next Phase**: API Integration (Phase 4) - Connect frontend to real backend

---

**Document Version**: 1.0  
**Implementation Date**: 2025-11-14  
**Status**: Complete and Ready for Use  
**Files**: 7 files, ~1,677 lines of code and documentation


