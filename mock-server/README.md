# Mock Server - Phase 3

Simple mock server for face search API that returns exact Phase 2 contracts.

## Features

✅ **Multiple Fixture Sets**
- `tiny` (10 results): High scores (0.85-0.95) - testing best matches
- `medium` (200 results): Normal distribution (0.40-0.95) - realistic workload, includes 5 broken URLs
- `large` (2000 results): Wide distribution (0.30-0.98) - stress testing, includes 20 broken URLs
- `edge_cases` (15 results): Perfect matches, threshold boundaries, very low scores
- `errors` (20 results): All broken URLs for error testing

✅ **Realistic Data**
- Decreasing similarity scores
- Multiple mock sites
- Variety in bounding boxes (position, size)
- Quality scores (0.3-1.0)
- Perceptual hashes
- Timestamps spread over 30 days
- Mock presigned URLs

✅ **Error Scenarios**
- Broken thumbnail URLs (404)
- Expired presigned URLs (403)
- API errors (500)
- Rate limiting (429)
- Empty result sets
- Configurable random error injection

✅ **Developer-Friendly**
- Configurable latency simulation (50-300ms)
- CORS enabled for frontend development
- Auto-generated API docs (`/docs`)
- Mock-specific configuration endpoints

## Quick Start

### 1. Install Dependencies

```bash
cd mock-server
pip install -r requirements.txt
```

### 2. Start Server

```bash
python app.py
```

Server runs on: http://localhost:8000

API docs: http://localhost:8000/docs

### 3. Test Endpoints

**Health Check:**
```bash
curl http://localhost:8000/api/v1/health
```

**Search (default fixture - medium):**
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg" \
  -F "top_k=10" \
  -F "threshold=0.75"
```

**Search with specific fixture:**
```bash
curl -X POST "http://localhost:8000/api/v1/search?fixture=tiny" \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg"
```

**Search with error scenario:**
```bash
curl -X POST "http://localhost:8000/api/v1/search?error_scenario=no_results" \
  -H "X-Tenant-ID: demo-tenant" \
  -F "image=@test.jpg"
```

**Get face by ID:**
```bash
curl http://localhost:8000/api/v1/faces/face-abc123 \
  -H "X-Tenant-ID: demo-tenant"
```

**List all fixtures:**
```bash
curl http://localhost:8000/mock/fixtures
```

## Frontend Integration

Update your frontend API configuration to point to the mock server:

```javascript
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

## Query Parameters

### Search Endpoint

- `fixture` (optional): Override default fixture
  - Values: `tiny`, `medium`, `large`, `edge_cases`, `errors`
  - Default: `medium`

- `error_scenario` (optional): Simulate error response
  - Values: `no_results`, `api_error`, `timeout`, `broken_url`, `expired_url`, `rate_limit`, `server_error`

### Example Usage

```bash
# Test with tiny dataset
curl -X POST "http://localhost:8000/api/v1/search?fixture=tiny" ...

# Test with large dataset
curl -X POST "http://localhost:8000/api/v1/search?fixture=large" ...

# Test error handling (no results)
curl -X POST "http://localhost:8000/api/v1/search?error_scenario=no_results" ...

# Test error handling (API error)
curl -X POST "http://localhost:8000/api/v1/search?error_scenario=api_error" ...

# Test with broken URLs
curl -X POST "http://localhost:8000/api/v1/search?fixture=errors" ...
```

## Configuration

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

**Configuration Options:**

- `default_fixture`: Default fixture to use (default: `medium`)
- `simulate_latency`: Enable latency simulation (default: `true`)
- `min_latency_ms`: Minimum latency in ms (default: `50`)
- `max_latency_ms`: Maximum latency in ms (default: `300`)
- `error_rate`: Probability of random errors 0-1 (default: `0.0`)

## Fixture Details

### Tiny (10 results)
Perfect for testing UI with minimal data.

```json
{
  "count": 10,
  "score_range": "0.8500 - 0.9500",
  "sites": 6,
  "broken_urls": 0
}
```

### Medium (200 results)
Realistic workload for development.

```json
{
  "count": 200,
  "score_range": "0.4000 - 0.9500",
  "sites": 6,
  "broken_urls": 5
}
```

### Large (2000 results)
Stress testing pagination and performance.

```json
{
  "count": 2000,
  "score_range": "0.3000 - 0.9800",
  "sites": 6,
  "broken_urls": 20
}
```

### Edge Cases (15 results)
Special cases for thorough testing.

- Perfect match (score 1.0)
- Threshold boundaries (0.74, 0.75, 0.76)
- Very low scores (0.01, 0.05, 0.15)

### Errors (20 results)
All URLs are intentionally broken for error testing.

## Testing Scenarios

### 1. Basic Functionality
```bash
# Default search
curl -X POST "http://localhost:8000/api/v1/search?fixture=medium" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg"
```

### 2. Pagination Testing
```bash
# Large dataset for pagination
curl -X POST "http://localhost:8000/api/v1/search?fixture=large" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg" -F "top_k=50"
```

### 3. Filtering Testing
```bash
# Test threshold filtering
curl -X POST "http://localhost:8000/api/v1/search?fixture=medium" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg" \
  -F "threshold=0.80"  # Should return fewer results
```

### 4. Error Handling
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

### 5. Performance Testing
```bash
# Disable latency for faster testing
curl -X POST "http://localhost:8000/mock/config" \
  -H "Content-Type: application/json" \
  -d '{"simulate_latency": false}'

# Test with large dataset
curl -X POST "http://localhost:8000/api/v1/search?fixture=large" \
  -H "X-Tenant-ID: demo-tenant" -F "image=@test.jpg" -F "top_k=50"
```

## API Contract Compliance

The mock server returns **exact Phase 2 API contracts** as defined in `api/openapi.yaml`:

### SearchResponse
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

## Troubleshooting

**Server won't start:**
```bash
# Check port availability
lsof -i :8000

# Use different port
uvicorn app:app --port 8001
```

**CORS errors:**
- CORS is enabled for all origins in development
- Check browser console for specific errors

**Fixture not found:**
```bash
# List available fixtures
curl http://localhost:8000/mock/fixtures
```

**Slow responses:**
```bash
# Disable latency simulation
curl -X POST "http://localhost:8000/mock/config" \
  -d '{"simulate_latency": false}'
```

## Development

### Regenerate Fixtures
```bash
cd mock-server
python fixtures.py
```

### View Fixture Summary
```bash
python fixtures.py
```

Output:
```
==============================================================================
FIXTURE SUMMARY
==============================================================================

TINY (10 results):
  Description: 10 high-quality matches (scores 0.85-0.95)
  Score range: 0.8500 - 0.9500
  Mean score: 0.9000
  Sites: 6 (demo-site.org, example.com, ...)
  Broken URLs: 0
  
...
```

### Add New Fixture
Edit `fixtures.py`:

```python
def generate_fixture_custom() -> Dict:
    hits = []
    # Your fixture logic here
    return {
        "name": "custom",
        "description": "Custom fixture",
        "hits": hits
    }

# Add to FIXTURE_SETS
FIXTURE_SETS = {
    # ... existing fixtures
    "custom": generate_fixture_custom()
}
```

## Acceptance Criteria

✅ Frontend can fully develop against mocks
- All Phase 2 API endpoints implemented
- Exact contract compliance
- Multiple realistic datasets

✅ Fixtures cover edge cases
- Perfect matches (score 1.0)
- Threshold boundaries (0.74-0.76)
- Empty results
- Very low scores
- Broken URLs
- Expired URLs

✅ Error scenarios
- 404 (not found)
- 403 (forbidden)
- 429 (rate limit)
- 500 (server error)
- Timeout simulation
- Configurable error injection

✅ Realistic data distributions
- Tiny: 10 results, high scores
- Medium: 200 results, normal distribution
- Large: 2000 results, wide distribution
- Multiple sites, timestamps, bbox variety

## Next Steps

1. Update frontend to use mock server
2. Test all user journeys with different fixtures
3. Test error handling with error scenarios
4. Performance test with large fixture
5. When backend is ready, swap API_BASE_URL back

## Notes

- Mock server is for **development only**
- Not suitable for production
- No authentication/authorization
- No persistence (restarts reset state)
- Fixtures are generated on server start

