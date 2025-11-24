# Face Pipeline & Vectorization Architecture Diagram

## 📁 File Structure & Components

```
Mordeaux-Face-Scanning-MVP/
│
├── face-pipeline/                    # Main face processing service
│   ├── main.py                       # FastAPI app (search API endpoints)
│   ├── worker.py                     # Redis Streams consumer (processes images)
│   │
│   ├── config/
│   │   └── settings.py               # Configuration (Qdrant, Redis, MinIO settings)
│   │
│   ├── pipeline/                     # Core processing modules
│   │   ├── processor.py              # ⭐ MAIN ORCHESTRATOR - process_image()
│   │   ├── detector.py                # Face detection (InsightFace)
│   │   │   ├── detect_faces()        # Detect faces in image
│   │   │   ├── detect_faces_raw()    # Raw InsightFace Face objects
│   │   │   └── align_and_crop()      # Align & crop faces to 112x112
│   │   │
│   │   ├── embedder.py                # Generate 512-dim embeddings
│   │   │   └── embed()                # Convert face crop → vector
│   │   │
│   │   ├── quality.py                 # Face quality assessment
│   │   │   └── evaluate()             # Check blur, size, pose, etc.
│   │   │
│   │   ├── dedup.py                   # Deduplication logic
│   │   │   ├── is_duplicate()         # Exact match (pHash)
│   │   │   └── should_skip()          # Near-duplicate (Hamming distance)
│   │   │
│   │   ├── indexer.py                 # ⭐ QDRANT INTERFACE
│   │   │   ├── ensure_collection()    # Create collection if missing
│   │   │   ├── upsert()               # Insert vectors to Qdrant
│   │   │   ├── search()               # Search similar faces
│   │   │   └── make_point()           # Create Qdrant PointStruct
│   │   │
│   │   ├── storage.py                 # MinIO storage operations
│   │   │   ├── put_bytes()            # Save images to MinIO
│   │   │   └── presign()              # Generate presigned URLs
│   │   │
│   │   ├── utils.py                   # Utilities (pHash, timestamps)
│   │   ├── stats.py                   # Statistics tracking
│   │   └── face_helpers.py            # Helper functions
│   │
│   ├── services/
│   │   └── search_api.py              # Search API endpoints
│   │
│   └── face_quality.py                # Quality configs (ENROLL, VERIFY, SEARCH)
│
├── backend/                           # Backend API service
│   ├── app/
│   │   ├── main.py                    # FastAPI app entry
│   │   │
│   │   ├── api/
│   │   │   └── routes.py              # API routes
│   │   │       ├── /api/v1/ingest     # Single image ingest
│   │   │       ├── /api/v1/ingest/batch  # Batch ingest
│   │   │       └── /api/v1/search     # Search passthrough
│   │   │
│   │   └── services/
│   │       ├── crawler.py             # ⭐ IMAGE CRAWLER
│   │       │   ├── crawl_page()       # Crawl single page
│   │       │   ├── crawl_site()       # Crawl multiple pages
│   │       │   └── _trigger_vectorization()  # Auto-vectorization trigger
│   │       │
│   │       ├── vector.py              # Vector DB abstraction
│   │       │   ├── upsert_embeddings() # Upsert to Qdrant/Pinecone
│   │       │   └── search_similar()   # Search Qdrant/Pinecone
│   │       │
│   │       ├── storage.py             # MinIO/S3 storage
│   │       ├── face.py                # Face detection (backend)
│   │       └── cache.py               # Hybrid cache (Redis + PostgreSQL)
│   │
│   └── requirements.txt               # Backend dependencies
│
└── docker-compose.yml                 # Service definitions
    ├── api (backend)
    ├── face-pipeline
    ├── worker (face-pipeline worker)
    ├── redis
    ├── minio
    └── qdrant
```

---

## 🔄 Data Flow: Image → Vector

### **Flow 1: Crawler → Vectorization (Auto)**

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. CRAWLER PHASE                                                │
└─────────────────────────────────────────────────────────────────┘
         │
         │ backend/app/services/crawler.py
         │
         ├─► crawl_page() / crawl_site()
         │   ├─► Downloads images from web
         │   ├─► Detects faces (optional)
         │   └─► Saves to MinIO (raw-images bucket)
         │
         │   saved_raw_keys = ["tenant_id/path/to/image.jpg", ...]
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. AUTO-VECTORIZATION TRIGGER                                   │
└─────────────────────────────────────────────────────────────────┘
         │
         │ backend/app/services/crawler.py::_trigger_vectorization()
         │
         ├─► POST /api/v1/ingest/batch
         │   └─► backend/app/api/routes.py::ingest_batch()
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. REDIS STREAMS                                                │
└─────────────────────────────────────────────────────────────────┘
         │
         │ Stream: "face:ingest"
         │ Message format:
         │ {
         │   "tenant_id": "...",
         │   "bucket": "raw-images",
         │   "key": "tenant_id/path/to/image.jpg",
         │   "site": "example.com",
         │   "image_sha256": "...",
         │   ...
         │ }
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. WORKER PROCESSING                                            │
└─────────────────────────────────────────────────────────────────┘
         │
         │ face-pipeline/worker.py
         │
         ├─► Consumes from Redis Streams
         ├─► Calls process_image() for each message
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. FACE PIPELINE PROCESSING                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         │ face-pipeline/pipeline/processor.py::process_image()
         │
         ├─► STEP 1: Download image from MinIO
         │   └─► pipeline/storage.py
         │
         ├─► STEP 2: Decode image (PIL/OpenCV)
         │   └─► pipeline/image_utils.py
         │
         ├─► STEP 3: Detect faces
         │   └─► pipeline/detector.py::detect_faces()
         │       └─► Uses InsightFace FaceAnalysis
         │
         ├─► STEP 4: For each face:
         │   │
         │   ├─► Align & crop to 112x112
         │   │   └─► pipeline/detector.py::align_and_crop()
         │   │
         │   ├─► Quality assessment
         │   │   └─► pipeline/quality.py::evaluate()
         │   │       └─► Uses face_quality.py configs
         │   │
         │   ├─► Compute pHash (deduplication)
         │   │   └─► pipeline/utils.py::compute_phash()
         │   │
         │   ├─► Check duplicates
         │   │   └─► pipeline/dedup.py::is_duplicate() / should_skip()
         │   │
         │   ├─► Generate embedding (512-dim vector)
         │   │   └─► pipeline/embedder.py::embed()
         │   │       └─► Uses InsightFace recognition model
         │   │
         │   ├─► Save face crop & thumbnail to MinIO
         │   │   └─► pipeline/storage.py::put_bytes()
         │   │
         │   └─► Create Qdrant point
         │       └─► pipeline/indexer.py::make_point()
         │
         ├─► STEP 5: Batch upsert to Qdrant
         │   └─► pipeline/indexer.py::upsert()
         │       └─► Qdrant collection: "faces_v1"
         │
         └─► STEP 6: Return results
             └─► Counts, artifacts, timings
```

### **Flow 2: Manual Ingest**

```
User/API → POST /api/v1/ingest
         │
         ├─► backend/app/api/routes.py::ingest_now()
         │   └─► Publishes to Redis Streams
         │
         └─► [Same as Flow 1, Step 3+]
```

### **Flow 3: Search**

```
User/API → POST /api/v1/search
         │
         ├─► backend/app/api/routes.py::search_passthrough()
         │   └─► Forwards to face-pipeline
         │
         ├─► face-pipeline/main.py::search_faces()
         │   ├─► Detect face in query image (if image provided)
         │   ├─► Generate embedding
         │   └─► Search Qdrant
         │       └─► pipeline/indexer.py::search()
         │
         └─► Return similar faces with scores
```

---

## 🗄️ Storage Locations

### **MinIO Buckets:**
- `raw-images/` - Original crawled/downloaded images
  - Path: `{tenant_id}/{image_sha256}.jpg`
- `thumbnails/` - Thumbnails (from crawler)
  - Path: `{tenant_id}/{image_sha256}_thumb.jpg`
- `face-crops/` - Cropped face regions (from pipeline)
  - Path: `{tenant_id}/{image_sha256}_face_{i}.jpg`
- `face-thumbs/` - Face thumbnails (from pipeline)
  - Path: `{tenant_id}/{image_sha256}_face_{i}_thumb.jpg`

### **Qdrant Collections:**
- `faces_v1` - Face embeddings
  - Vector: 512-dim float32 (L2 normalized)
  - Payload:
    ```json
    {
      "tenant_id": "...",
      "image_sha256": "...",
      "face_index": 0,
      "crop_key": "tenant_id/..._face_0.jpg",
      "thumb_key": "tenant_id/..._face_0_thumb.jpg",
      "p_hash": "...",
      "p_hash_prefix": "...",
      "site": "...",
      "quality_score": 0.95,
      "det_score": 0.98,
      ...
    }
    ```
- `identities_v1` - Identity centroids (for verification)

---

## 🔑 Key Files Summary

### **Core Processing:**
| File | Purpose |
|------|---------|
| `face-pipeline/pipeline/processor.py` | Main orchestrator - `process_image()` |
| `face-pipeline/pipeline/detector.py` | Face detection & alignment |
| `face-pipeline/pipeline/embedder.py` | Vector embedding generation |
| `face-pipeline/pipeline/quality.py` | Face quality assessment |
| `face-pipeline/pipeline/dedup.py` | Deduplication logic |
| `face-pipeline/pipeline/indexer.py` | Qdrant operations (upsert/search) |
| `face-pipeline/pipeline/storage.py` | MinIO operations |

### **Services:**
| File | Purpose |
|------|---------|
| `face-pipeline/worker.py` | Redis Streams consumer |
| `face-pipeline/main.py` | FastAPI search API |
| `backend/app/services/crawler.py` | Image crawler + auto-vectorization |
| `backend/app/api/routes.py` | Ingest & search endpoints |
| `backend/app/services/vector.py` | Vector DB abstraction |

### **Configuration:**
| File | Purpose |
|------|---------|
| `face-pipeline/config/settings.py` | All settings (Qdrant, Redis, MinIO) |
| `face-pipeline/face_quality.py` | Quality configs (ENROLL, VERIFY, SEARCH) |

---

## 🔌 Integration Points

### **1. Crawler → Vectorization:**
- **File:** `backend/app/services/crawler.py`
- **Method:** `_trigger_vectorization()`
- **Calls:** `POST http://localhost:8000/api/v1/ingest/batch`
- **Triggered:** After `crawl_page()` or `crawl_site()` completes

### **2. Ingest API → Redis:**
- **File:** `backend/app/api/routes.py`
- **Endpoints:** 
  - `/api/v1/ingest` (single)
  - `/api/v1/ingest/batch` (batch)
- **Stream:** `face:ingest` (configurable via `REDIS_STREAM_NAME`)

### **3. Worker → Pipeline:**
- **File:** `face-pipeline/worker.py`
- **Method:** `process_image()` from `pipeline/processor.py`
- **Consumes:** Redis Streams messages

### **4. Pipeline → Qdrant:**
- **File:** `face-pipeline/pipeline/indexer.py`
- **Methods:** `upsert()`, `search()`
- **Collection:** `faces_v1` (configurable via `QDRANT_COLLECTION`)

### **5. Pipeline → MinIO:**
- **File:** `face-pipeline/pipeline/storage.py`
- **Methods:** `put_bytes()`, `presign()`
- **Buckets:** `raw-images`, `face-crops`, `face-thumbs`

---

## 🎯 Key Functions Reference

### **Face Detection:**
```python
# face-pipeline/pipeline/detector.py
detect_faces(img_np_bgr) -> List[Dict]  # Detect faces
align_and_crop(img_bgr, landmarks) -> np.ndarray  # Align to 112x112
```

### **Embedding:**
```python
# face-pipeline/pipeline/embedder.py
embed(aligned_bgr_112) -> np.ndarray  # 512-dim vector
```

### **Quality:**
```python
# face-pipeline/pipeline/quality.py
evaluate(img_bgr, face) -> QualityResult  # Quality assessment
```

### **Deduplication:**
```python
# face-pipeline/pipeline/dedup.py
is_duplicate(phex) -> bool  # Exact match
should_skip(tenant_id, pfx, phex, max_dist) -> bool  # Near-duplicate
```

### **Qdrant:**
```python
# face-pipeline/pipeline/indexer.py
upsert(points: List[PointStruct])  # Insert vectors
search(vector, top_k, tenant_id, threshold) -> List[ScoredPoint]  # Search
```

### **Storage:**
```python
# face-pipeline/pipeline/storage.py
put_bytes(bucket, key, bytes)  # Save to MinIO
presign(bucket, key, expires) -> str  # Generate presigned URL
```

---

## 🔧 Environment Variables

### **Face Pipeline:**
- `QDRANT_URL` - Qdrant server URL
- `QDRANT_COLLECTION` - Collection name (default: `faces_v1`)
- `REDIS_URL` - Redis connection URL
- `REDIS_STREAM_NAME` - Stream name (default: `face:ingest`)
- `MINIO_ENDPOINT` - MinIO server endpoint
- `VECTOR_DIM` - Vector dimension (default: 512)

### **Backend:**
- `CRAWLER_AUTO_VECTORIZATION` - Enable auto-vectorization (default: `true`)
- `REDIS_STREAM_NAME` - Stream name for ingest
- `MINIO_ENDPOINT` - MinIO server endpoint

---

## 📊 Data Structures

### **Pipeline Input (Redis Message):**
```python
{
    "tenant_id": "tenant-123",
    "bucket": "raw-images",
    "key": "tenant-123/abc123.jpg",
    "url": "https://example.com/image.jpg",
    "site": "example.com",
    "image_sha256": "abc123...",
    "image_phash": "0000...",
    "ts": 1234567890,
    "meta": {},
    "face_hints": None
}
```

### **Pipeline Output:**
```python
{
    "image_sha256": "abc123...",
    "counts": {
        "faces_total": 2,
        "faces_accepted": 1,
        "faces_rejected": 1,
        "faces_dup_skipped": 0
    },
    "artifacts": {
        "crops": ["tenant-123/abc123_face_0.jpg"],
        "thumbs": ["tenant-123/abc123_face_0_thumb.jpg"],
        "metadata": ["tenant-123/abc123_face_0.json"]
    },
    "timings_ms": {...}
}
```

### **Qdrant Point:**
```python
PointStruct(
    id="uuid-from-face-id",
    vector=[0.1, 0.2, ..., 0.9],  # 512 floats
    payload={
        "tenant_id": "tenant-123",
        "image_sha256": "abc123...",
        "face_index": 0,
        "crop_key": "tenant-123/abc123_face_0.jpg",
        "thumb_key": "tenant-123/abc123_face_0_thumb.jpg",
        "p_hash": "abc123...",
        "p_hash_prefix": "abc1",
        "site": "example.com",
        "quality_score": 0.95,
        "det_score": 0.98,
        ...
    }
)
```

---

## 🚀 Quick Reference: Where to Find What

| What You Need | File Location |
|---------------|---------------|
| Change face detection model | `face-pipeline/pipeline/detector.py` |
| Change embedding model | `face-pipeline/pipeline/embedder.py` |
| Change quality thresholds | `face-pipeline/face_quality.py` |
| Change deduplication logic | `face-pipeline/pipeline/dedup.py` |
| Change Qdrant collection | `face-pipeline/config/settings.py` |
| Change crawler behavior | `backend/app/services/crawler.py` |
| Change ingest API | `backend/app/api/routes.py` |
| Change search API | `face-pipeline/main.py` |
| Change storage buckets | `face-pipeline/pipeline/storage.py` |
| Add new pipeline step | `face-pipeline/pipeline/processor.py` |

---

**Last Updated:** 2025-11-17
**Branch:** identity-safe-search-endpoint-(pipeline-+-backend)

