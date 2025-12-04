# End-to-End Integration Guide
## Frontend ↔ Backend ↔ MinIO ↔ Face Pipeline ↔ Qdrant

**Last Updated**: December 4, 2025  
**Status**: ✅ Integration Ready

---

## 🎯 Overview

This guide explains how images flow through the complete Mordeaux system, from frontend upload to search results display with full metadata (tenant_id, site, similarity scores, etc.).

---

## 🔄 Complete Data Flow

```
┌─────────────┐
│  Frontend   │  User uploads image
│  (React)    │
└──────┬──────┘
       │ POST /api/v1/search (with image_b64)
       ↓
┌─────────────┐
│  Backend    │  Proxies to face-pipeline
│  (FastAPI)  │
└──────┬──────┘
       │ POST /api/v1/search
       ↓
┌─────────────┐
│Face Pipeline│  1. Detect faces
│  (FastAPI)  │  2. Generate embeddings
└──────┬──────┘  3. Search Qdrant
       │
       ├──────────────┐
       │              │
       ↓              ↓
┌─────────────┐  ┌─────────────┐
│   Qdrant    │  │   MinIO     │
│ (Vector DB) │  │  (Storage)  │
└──────┬──────┘  └──────┬──────┘
       │                │
       │ Return matches │ Presigned URLs
       ↓                ↓
┌─────────────────────────┐
│  Search Results         │
│  - face_id              │
│  - score (similarity)   │
│  - tenant_id            │
│  - site                 │
│  - url (original)       │
│  - thumb_url (MinIO)    │
│  - bbox, quality, etc.  │
└─────────────────────────┘
```

---

## 📍 Key Endpoints

### Frontend → Backend

**Search for Similar Faces**
```http
POST http://localhost/api/v1/search
Content-Type: application/json

{
  "tenant_id": "demo-tenant",
  "image_b64": "<base64-encoded-image>",
  "top_k": 50,
  "threshold": 0.70
}
```

**Response:**
```json
{
  "count": 10,
  "hits": [
    {
      "face_id": "abc123...",
      "score": 0.95,
      "payload": {
        "tenant_id": "demo-tenant",
        "site": "example.com",
        "url": "https://example.com/photo.jpg",
        "ts": "2024-01-15T10:30:00Z",
        "bbox": [100, 150, 200, 250],
        "quality": 0.89,
        "quality_is_usable": true
      },
      "thumb_url": "http://localhost:9000/thumbnails/demo-tenant/abc123_thumb.jpg?X-Amz-..."
    }
  ]
}
```

### Backend → Face Pipeline

The backend proxies the request to the face pipeline:

```http
POST http://face-pipeline:8001/api/v1/search
```

The face pipeline:
1. Decodes the base64 image
2. Detects faces using InsightFace
3. Generates 512-dim embeddings using ArcFace
4. Searches Qdrant for similar vectors
5. Returns results with presigned MinIO URLs

---

## 🧪 Testing the Integration

### Option 1: Upload Test Page (Recommended)

1. **Start all localhost services** (see `LOCALHOST_COMPLETE_SETUP.md`)
   ```powershell
   docker-compose up -d
   cd frontend
   npm.cmd run dev
   ```

2. **Open the upload test page**
   ```
   http://localhost:5173/upload-test
   ```

3. **Upload a test image**
   - Select an image with a face
   - Click "Search for Similar Faces"
   - View results with full metadata

4. **Verify the data flow**
   - ✅ Image is sent to backend
   - ✅ Face pipeline processes it
   - ✅ Qdrant returns matches
   - ✅ MinIO presigned URLs work
   - ✅ All metadata is displayed (tenant_id, site, score, etc.)

### Option 2: Python Test Script

```powershell
python test_e2e_integration.py
```

This script tests:
- Service health checks
- Image upload and search
- MinIO and Qdrant connectivity
- Full metadata in responses

### Option 3: Manual cURL Test

```bash
# 1. Encode an image to base64
$base64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("path/to/image.jpg"))

# 2. Send search request
curl -X POST http://localhost/api/v1/search `
  -H "Content-Type: application/json" `
  -d "{\"tenant_id\":\"demo-tenant\",\"image_b64\":\"$base64\",\"top_k\":50,\"threshold\":0.70}"
```

---

## 🗂️ Data Storage Structure

### MinIO Buckets

| Bucket | Purpose | Example Path |
|--------|---------|--------------|
| `raw-images` | Original uploaded images | `demo-tenant/photo_123.jpg` |
| `face-crops` | Cropped face images (112x112) | `demo-tenant/face_abc123.jpg` |
| `thumbnails` | Thumbnail versions | `demo-tenant/face_abc123_thumb.jpg` |
| `face-metadata` | JSON metadata files | `demo-tenant/face_abc123.json` |

### Qdrant Collections

| Collection | Purpose | Vector Dim | Key Fields |
|------------|---------|------------|------------|
| `faces_v1` | All detected faces | 512 | tenant_id, site, url, bbox, quality |
| `identities_v1` | Enrolled identities | 512 | tenant_id, identity_id |

### Metadata Fields

Every face in Qdrant includes:

```json
{
  "tenant_id": "demo-tenant",        // Multi-tenant isolation
  "site": "example.com",             // Source website
  "url": "https://...",              // Original image URL
  "ts": "2024-01-15T10:30:00Z",      // Timestamp
  "bbox": [x, y, width, height],     // Face bounding box
  "quality": 0.89,                   // Quality score (0-1)
  "quality_is_usable": true,         // Passed quality checks
  "image_sha256": "abc...",          // Image hash
  "p_hash": "def...",                // Perceptual hash
  "identity_id": "user123"           // Optional: enrolled identity
}
```

---

## 🔍 Verifying Data is Synced

### Check if data exists in Qdrant

```powershell
# Get collection info
curl http://localhost:6333/collections/faces_v1

# Count points (faces)
curl http://localhost:6333/collections/faces_v1 | jq '.result.points_count'

# Scroll through some faces
curl -X POST http://localhost:6333/collections/faces_v1/points/scroll `
  -H "Content-Type: application/json" `
  -d '{"limit": 10, "with_payload": true}'
```

### Check MinIO storage

1. Open MinIO Console: http://localhost:9001
2. Login: `minioadmin` / `minioadmin`
3. Browse buckets:
   - `raw-images`
   - `face-crops`
   - `thumbnails`
   - `face-metadata`

### Check backend logs

```powershell
# Backend API logs
docker-compose logs -f api

# Face pipeline logs
docker-compose logs -f face-pipeline
```

---

## 🚀 Populating the Database

If your database is empty, you need to ingest images first.

### Method 1: Upload via Frontend

Use the upload test page at http://localhost:5173/upload-test

### Method 2: Ingest API

```bash
curl -X POST http://localhost/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo-tenant",
    "bucket": "raw-images",
    "key": "demo-tenant/photo.jpg",
    "site": "example.com",
    "meta": {"test": true}
  }'
```

### Method 3: Batch Ingest

```bash
curl -X POST http://localhost/api/v1/ingest/batch \
  -H "Content-Type: application/json" \
  -d '{
    "items": [
      {
        "tenant_id": "demo-tenant",
        "url": "https://example.com/photo1.jpg",
        "site": "example.com"
      },
      {
        "tenant_id": "demo-tenant",
        "url": "https://example.com/photo2.jpg",
        "site": "example.com"
      }
    ]
  }'
```

### Method 4: Run Crawler

```powershell
cd backend/scripts
python crawl_images.py --site example.com --tenant demo-tenant
```

---

## 🎨 Frontend Integration

### Current Status

✅ **Upload Test Page** (`/upload-test`)
- Direct image upload
- Real-time search
- Full metadata display
- Tenant ID integration

🚧 **Search Dev Page** (`/dev/search`)
- Currently uses mock data
- Needs update to use real API

### Updating SearchDevPage to Use Real Data

Replace the mock data section in `frontend/src/pages/SearchDevPage.tsx`:

```typescript
// OLD: Mock data
const mockResults: SearchHit[] = Array.from({ length: 100 }, ...);

// NEW: Real API call
const [results, setResults] = useState<SearchHit[]>([]);
const [loading, setLoading] = useState(false);

useEffect(() => {
  async function loadResults() {
    setLoading(true);
    try {
      const response = await fetch('http://localhost/api/v1/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          tenant_id: 'demo-tenant',
          image_b64: queryImageBase64,
          top_k: 100,
          threshold: 0.70
        })
      });
      const data = await response.json();
      setResults(data.hits || []);
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setLoading(false);
    }
  }
  loadResults();
}, [queryImageBase64]);
```

---

## 🔧 Troubleshooting

### No search results

**Possible causes:**
1. Database is empty → Run ingestion first
2. Threshold too high → Lower threshold to 0.5
3. No faces detected → Use images with clear faces
4. Tenant ID mismatch → Check tenant_id in request

**Solution:**
```powershell
# Check if database has data
curl http://localhost:6333/collections/faces_v1

# If empty, run ingestion
python test_e2e_integration.py
```

### Images not displaying

**Possible causes:**
1. MinIO presigned URLs expired
2. CORS issues
3. Wrong bucket/key

**Solution:**
```powershell
# Check MinIO is accessible
curl http://localhost:9001

# Check bucket exists
# Login to MinIO console and verify buckets
```

### Search timeout

**Possible causes:**
1. Face pipeline not responding
2. Large image processing
3. Qdrant slow query

**Solution:**
```powershell
# Check face pipeline health
curl http://localhost/pipeline/api/v1/health

# Check logs
docker-compose logs -f face-pipeline

# Restart if needed
docker-compose restart face-pipeline
```

---

## 📊 Monitoring

### Check Service Health

```powershell
# All services
docker-compose ps

# Backend API
curl http://localhost/api/v1/health

# Face Pipeline
curl http://localhost/pipeline/api/v1/health

# Qdrant
curl http://localhost:6333/readyz

# MinIO
curl http://localhost:9001
```

### View Logs

```powershell
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f face-pipeline
docker-compose logs -f qdrant
```

### Performance Metrics

```powershell
# Qdrant metrics
curl http://localhost:6333/metrics

# Collection stats
curl http://localhost:6333/collections/faces_v1
```

---

## ✅ Integration Checklist

- [x] Backend API running and healthy
- [x] Face pipeline running and healthy
- [x] MinIO accessible with buckets created
- [x] Qdrant accessible with collections created
- [x] Frontend dev server running on port 5173
- [x] Upload test page accessible at /upload-test
- [ ] Database populated with test data
- [ ] Search returns results with metadata
- [ ] Tenant ID correctly isolated
- [ ] Presigned URLs working for images
- [ ] All metadata fields displayed correctly

---

## 📚 Related Documentation

- **Localhost Setup**: `LOCALHOST_COMPLETE_SETUP.md`
- **API Documentation**: `docs/api.md`
- **Face Pipeline**: `face-pipeline/README.md`
- **Frontend Guide**: `frontend/README_SEARCH_DEV.md`
- **Docker Setup**: `DOCKER_README.md`

---

## 🎯 Next Steps

1. **Test the upload page**: http://localhost:5173/upload-test
2. **Verify data flow**: Upload image → Check Qdrant → See results
3. **Populate database**: Run crawler or batch ingest
4. **Update SearchDevPage**: Replace mock data with real API calls
5. **Add more features**: Filters, pagination, advanced search

---

**Questions or Issues?**
- Check logs: `docker-compose logs -f`
- Verify services: `docker-compose ps`
- Test endpoints: Use the Python test script
- Review this guide: All integration points documented above

**Happy Testing!** 🚀

