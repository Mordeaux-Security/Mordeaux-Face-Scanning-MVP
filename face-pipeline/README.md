# Face Pipeline Service - Developer Documentation

**Version**: 0.1.0 (DEV2 Phase)  
**Status**: Infrastructure Complete, Implementation In Progress  
**Owner**: Dev Team B (Face Processing Pipeline)

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Service Responsibilities](#service-responsibilities)
3. [Data Contracts](#data-contracts)
4. [API Contracts](#api-contracts)
5. [Storage & Artifacts](#storage--artifacts)
6. [Vector Database Schema](#vector-database-schema)
7. [Running Locally](#running-locally)
8. [Testing](#testing)
9. [Next Milestones](#next-milestones)
10. [Integration Guide](#integration-guide)

---

## Overview

The **Face Pipeline Service** is a modular face detection, quality assessment, embedding generation, and similarity search system. It processes images from a message queue, extracts faces, generates embeddings, and indexes them in a vector database for similarity search.

### Architecture

```
┌─────────────┐
│   Queue     │ ──► PipelineInput
│  (RabbitMQ) │     (JSON message)
└─────────────┘
       │
       ▼
┌─────────────────────────────────────────────┐
│         Face Processing Pipeline            │
│                                             │
│  1. Download image (MinIO)                  │
│  2. Detect faces (InsightFace)              │
│  3. Quality checks (blur, size, pose)       │
│  4. Generate embeddings (512-dim)           │
│  5. Deduplicate (pHash + Hamming distance)  │
│  6. Store artifacts (crops, thumbs)         │
│  7. Index in Qdrant (vector search)         │
└─────────────────────────────────────────────┘
       │
       ├──► MinIO (face crops, thumbnails, metadata)
       ├──► Qdrant (face embeddings + metadata)
       └──► Response (counts, artifacts, timings)
```

### Tech Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Face Detection**: InsightFace (buffalo_l model)
- **Storage**: MinIO (S3-compatible)
- **Vector DB**: Qdrant
- **Queue**: RabbitMQ (consumed by worker)
- **Testing**: pytest

---

## Service Responsibilities

### What This Service Does ✅

1. **Face Detection**: Detect faces in images using InsightFace
2. **Quality Assessment**: Filter low-quality faces (blur, size, pose)
3. **Embedding Generation**: Generate 512-dim face embeddings
4. **Deduplication**: Detect duplicate faces using pHash + Hamming distance
5. **Artifact Storage**: Store face crops, thumbnails, and metadata in MinIO
6. **Vector Indexing**: Index embeddings in Qdrant for similarity search
7. **Search API**: Provide REST API for face similarity search

### What This Service Does NOT Do ❌

1. **Image Crawling**: Done by upstream crawler service
2. **Raw Image Storage**: Raw images stored by crawler before queue
3. **Face Recognition**: This is face embedding, not identity matching
4. **User Management**: Done by main backend
5. **Authentication**: API is internal, auth handled by gateway

---

## Data Contracts

### 1. Queue Message Schema (PipelineInput)

**Source**: Messages from RabbitMQ queue (sent by crawler)

```python
{
    "image_sha256": str,       # SHA-256 hash of image (hex)
    "bucket": str,             # MinIO bucket name (e.g., "raw-images")
    "key": str,                # Object key/path (e.g., "tenant1/2024/01/abc123.jpg")
    "tenant_id": str,          # Multi-tenant ID (for filtering)
    "site": str,               # Source site/domain (e.g., "example.com")
    "url": HttpUrl,            # Original image URL
    "image_phash": str,        # Perceptual hash (16-char hex)
    "face_hints": Optional[List[Dict]]  # Optional upstream face hints
}
```

**Example**:
```json
{
    "image_sha256": "abc123def456789...",
    "bucket": "raw-images",
    "key": "tenant1/2024/10/image_001.jpg",
    "tenant_id": "acme-corp",
    "site": "example.com",
    "url": "https://example.com/photos/person.jpg",
    "image_phash": "8f373c9c3c9c3c1e",
    "face_hints": null
}
```

**face_hints** (optional):
```json
[
    {
        "bbox": [x1, y1, x2, y2],  // Bounding box
        "confidence": 0.99,         // Detection confidence
        "landmarks": [[x, y], ...] // Optional facial landmarks
    }
]
```

---

### 2. Pipeline Output Schema

**Returned** by `process_image()` function after processing.

```python
{
    "image_sha256": str,       # Same as input
    "counts": {
        "faces_total": int,    # Total faces detected
        "accepted": int,       # Passed quality checks
        "rejected": int,       # Failed quality checks
        "dup_skipped": int     # Skipped as duplicates
    },
    "artifacts": {
        "crops": List[str],    # Face crop storage keys
        "thumbs": List[str],   # Thumbnail storage keys
        "metadata": List[str]  # Metadata JSON storage keys
    },
    "timings_ms": {
        "download_ms": float,
        "decode_ms": float,
        "detection_ms": float,
        "alignment_ms": float,
        "quality_ms": float,
        "phash_ms": float,
        "dedup_ms": float,
        "embedding_ms": float,
        "upsert_ms": float
    }
}
```

**Example**:
```json
{
    "image_sha256": "abc123def456789...",
    "counts": {
        "faces_total": 3,
        "accepted": 2,
        "rejected": 1,
        "dup_skipped": 0
    },
    "artifacts": {
        "crops": [
            "face-crops/acme-corp/abc123_face_0.jpg",
            "face-crops/acme-corp/abc123_face_1.jpg"
        ],
        "thumbs": [
            "thumbnails/acme-corp/abc123_face_0_thumb.jpg",
            "thumbnails/acme-corp/abc123_face_1_thumb.jpg"
        ],
        "metadata": [
            "face-metadata/acme-corp/abc123_face_0.json",
            "face-metadata/acme-corp/abc123_face_1.json"
        ]
    },
    "timings_ms": {
        "download_ms": 45.2,
        "decode_ms": 12.3,
        "detection_ms": 89.7,
        "alignment_ms": 15.4,
        "quality_ms": 23.1,
        "phash_ms": 8.5,
        "dedup_ms": 5.2,
        "embedding_ms": 67.8,
        "upsert_ms": 18.9
    }
}
```

---

## API Contracts

### Base URL
```
http://localhost:8000
```

### 1. POST /api/v1/search

**Purpose**: Search for similar faces by image or embedding vector

**Request**:
```json
{
    "image": "bytes (optional)",           // Upload image for search
    "vector": [0.1, 0.2, ...],            // OR pre-computed 512-dim vector
    "top_k": 10,                          // Max results (1-100)
    "tenant_id": "acme-corp",             // Required: tenant filter
    "threshold": 0.75                     // Min similarity (0.0-1.0)
}
```

**Response** (200 OK):
```json
{
    "query": {
        "tenant_id": "acme-corp",
        "top_k": 10,
        "threshold": 0.75,
        "search_mode": "vector"
    },
    "hits": [
        {
            "face_id": "face-uuid-123",
            "score": 0.95,                 // Cosine similarity
            "payload": {
                "tenant_id": "acme-corp",
                "site": "example.com",
                "url": "https://...",
                "bbox": [10, 20, 100, 200],
                "quality": {...},
                "ts": "2024-10-14T12:00:00Z",
                "p_hash": "8f373c9c...",
                "image_sha256": "abc123..."
            },
            "thumb_url": "https://minio.../presigned-url"
        }
    ],
    "count": 1
}
```

**Current Status**: Returns empty list (placeholder)

---

### 2. GET /api/v1/faces/{face_id}

**Purpose**: Retrieve detailed information about a specific face

**Request**:
```http
GET /api/v1/faces/face-uuid-123
```

**Response** (200 OK):
```json
{
    "face_id": "face-uuid-123",
    "payload": {
        "tenant_id": "acme-corp",
        "site": "example.com",
        "url": "https://...",
        "bbox": [10, 20, 100, 200],
        "quality": {
            "blur": 145.7,
            "size": [112, 112],
            "pass": true,
            "reason": "ok"
        },
        "ts": "2024-10-14T12:00:00Z",
        "p_hash": "8f373c9c3c9c3c1e",
        "p_hash_prefix": "8f37",
        "image_sha256": "abc123..."
    },
    "thumb_url": "https://minio.../presigned-url"
}
```

**Current Status**: Returns placeholder (empty payload)

---

### 3. GET /api/v1/stats

**Purpose**: Get pipeline processing statistics

**Request**:
```http
GET /api/v1/stats
```

**Response** (200 OK):
```json
{
    "processed": 12543,      // Total faces indexed
    "rejected": 892,         // Faces failed quality
    "dup_skipped": 234       // Duplicate faces skipped
}
```

**Current Status**: Returns 0,0,0 (placeholder)

---

### 4. GET /ready

**Purpose**: Kubernetes readiness probe

**Request**:
```http
GET /ready
```

**Response** (503 Service Unavailable):
```json
{
    "ready": false,
    "reason": "models_not_loaded",
    "checks": {
        "models": false,      // InsightFace models loaded?
        "storage": false,     // MinIO accessible?
        "vector_db": false    // Qdrant accessible?
    }
}
```

**When Ready** (200 OK):
```json
{
    "ready": true,
    "reason": "all_systems_operational",
    "checks": {
        "models": true,
        "storage": true,
        "vector_db": true
    }
}
```

---

### 5. GET /health

**Purpose**: Liveness probe (simple health check)

**Response** (200 OK):
```json
{
    "status": "ok"
}
```

---

## Storage & Artifacts

### MinIO Buckets

| Bucket | Purpose | Example Path |
|--------|---------|--------------|
| `raw-images` | Original images (from crawler) | `tenant1/2024/10/abc123.jpg` |
| `face-crops` | Extracted face crops (112x112) | `tenant1/abc123_face_0.jpg` |
| `thumbnails` | Face thumbnails (64x64) | `tenant1/abc123_face_0_thumb.jpg` |
| `face-metadata` | Face metadata JSON | `tenant1/abc123_face_0.json` |

### Artifact Path Structure

**Face Crop**:
```
face-crops/{tenant_id}/{image_sha256}_face_{index}.jpg
```

**Thumbnail**:
```
thumbnails/{tenant_id}/{image_sha256}_face_{index}_thumb.jpg
```

**Metadata**:
```
face-metadata/{tenant_id}/{image_sha256}_face_{index}.json
```

**Metadata JSON Content**:
```json
{
    "face_id": "uuid",
    "image_sha256": "abc123...",
    "bbox": [x1, y1, x2, y2],
    "landmarks": [[x, y], ...],
    "quality": {
        "blur": 145.7,
        "size": [112, 112],
        "pass": true,
        "reason": "ok"
    },
    "embedding_checksum": "md5-hash",
    "tenant_id": "acme-corp",
    "site": "example.com",
    "url": "https://...",
    "created_at": "2024-10-14T12:00:00Z"
}
```

---

## Vector Database Schema

### Qdrant Collection

**Collection Name**: `faces_v1`  
**Vector Dimension**: 512  
**Distance Metric**: Cosine similarity

### Payload Fields (Required 9 Fields)

```python
{
    "tenant_id": str,        # Multi-tenant filter
    "site": str,             # Source site/domain
    "url": str,              # Original image URL
    "ts": str,               # Timestamp (ISO 8601)
    "p_hash": str,           # Perceptual hash (16-char hex)
    "p_hash_prefix": str,    # First 4 hex chars (for dedup filtering)
    "bbox": List[int],       # Bounding box [x1, y1, x2, y2]
    "quality": Dict,         # Quality metrics
    "image_sha256": str      # Image hash
}
```

### Example Qdrant Point

```python
{
    "id": "face-uuid-123",
    "vector": [0.123, -0.456, ...],  # 512-dim float32
    "payload": {
        "tenant_id": "acme-corp",
        "site": "example.com",
        "url": "https://example.com/photo.jpg",
        "ts": "2024-10-14T12:00:00Z",
        "p_hash": "8f373c9c3c9c3c1e",
        "p_hash_prefix": "8f37",
        "bbox": [10, 20, 100, 200],
        "quality": {
            "blur": 145.7,
            "size": [112, 112],
            "pass": true,
            "reason": "ok"
        },
        "image_sha256": "abc123def456..."
    }
}
```

### Query Filters

**Search by tenant**:
```python
{
    "must": [
        {"key": "tenant_id", "match": {"value": "acme-corp"}}
    ]
}
```

**Search by site**:
```python
{
    "must": [
        {"key": "site", "match": {"value": "example.com"}}
    ]
}
```

**Deduplication filter** (same pHash prefix):
```python
{
    "must": [
        {"key": "p_hash_prefix", "match": {"value": "8f37"}}
    ]
}
```

---

## Running Locally

### Prerequisites

1. **Python 3.10+**
2. **Docker & Docker Compose** (for dependencies)
3. **MinIO** (running on port 9000)
4. **Qdrant** (running on port 6333)

### Quick Start

#### 1. Install Dependencies

```bash
cd face-pipeline
pip install -r requirements.txt
```

#### 2. Configure Environment

Create `.env` file in project root:

```bash
# MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
MINIO_BUCKET_RAW=raw-images
MINIO_BUCKET_CROPS=face-crops
MINIO_BUCKET_THUMBS=thumbnails
MINIO_BUCKET_METADATA=face-metadata

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=faces_v1
VECTOR_DIM=512

# Quality Thresholds
MIN_FACE_SIZE=80
BLUR_MIN_VARIANCE=120.0
MIN_OVERALL_QUALITY=0.7

# Pipeline Settings
MAX_FACES_PER_IMAGE=10
ENABLE_DEDUPLICATION=true
PRESIGN_TTL_SEC=600

# API
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

#### 3. Start Dependencies (Docker Compose)

```bash
# From project root
docker-compose up -d minio qdrant
```

#### 4. Run the Service

```bash
cd face-pipeline
python main.py
```

**API will be available at**:
- Main: http://localhost:8000
- Docs: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

### Using Makefile (If Available)

```bash
# Start all services
make up

# Start just the API
make run

# Run tests
make test

# Stop all services
make down
```

---

## Testing

### Run All Tests

```bash
cd face-pipeline
python -m pytest tests/ -v
```

### Run Specific Test File

```bash
python -m pytest tests/test_quality.py -v
python -m pytest tests/test_embedder.py -v
python -m pytest tests/test_processor_integration.py -v
```

### Test Coverage

```bash
python -m pytest tests/ --cov=pipeline --cov-report=html
```

### Manual API Testing

```bash
# Health check
curl http://localhost:8000/health

# Readiness check
curl http://localhost:8000/ready

# Get stats
curl http://localhost:8000/api/v1/stats

# Search (placeholder)
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2],
    "top_k": 10,
    "tenant_id": "test-tenant",
    "threshold": 0.75
  }'
```

---

## Next Milestones

### ✅ Completed (Steps 1-11)

- [x] Data contracts and pipeline orchestration
- [x] Storage interfaces (MinIO)
- [x] Detector interfaces (InsightFace stubs)
- [x] Quality gates (stubs)
- [x] Embedding interfaces (stubs)
- [x] Indexing interfaces (Qdrant stubs)
- [x] Deduplication helpers (pHash)
- [x] Full 12-step orchestration flow
- [x] Search API contracts (4 endpoints)
- [x] Observability (timer + /ready endpoint)
- [x] Test infrastructure (33+ tests)

### 🚧 In Progress (DEV2 Phase)

#### Priority 1: Core Pipeline Implementation
- [ ] **Implement `detect_faces()`**: Load InsightFace buffalo_l model, detect faces
- [ ] **Implement `embed()`**: Generate real 512-dim embeddings
- [ ] **Implement `evaluate()`**: Real quality checks (blur, size, brightness, pose)
- [ ] **Implement alignment**: Crop and align faces using landmarks
- [ ] **Add timing instrumentation**: Use `timer()` throughout pipeline

#### Priority 2: Storage & Indexing
- [ ] **Implement MinIO operations**: `get_bytes()`, `put_bytes()`, `presign()`
- [ ] **Implement Qdrant operations**: `ensure_collection()`, `upsert()`, `search()`
- [ ] **Test with real dependencies**: Integration tests with MinIO + Qdrant

#### Priority 3: Search API Implementation
- [ ] **Implement POST /search**: Image → embedding → Qdrant search
- [ ] **Implement GET /faces/{id}**: Retrieve by ID with presigned URLs
- [ ] **Implement GET /stats**: Track processed/rejected/dup_skipped
- [ ] **Implement /ready checks**: Model + MinIO + Qdrant health

#### Priority 4: Real Assertions
- [ ] **Add test fixtures**: Sample faces, non-faces, low quality images
- [ ] **Test actual detection**: Verify faces are found
- [ ] **Test actual embeddings**: Verify similarity (same face → high score)
- [ ] **Test actual quality**: Verify threshold filtering
- [ ] **Integration tests**: End-to-end with real services

#### Priority 5: Production Readiness
- [ ] **Error handling**: Retry logic, graceful degradation
- [ ] **Monitoring**: Prometheus metrics export
- [ ] **Performance**: Batch processing, async optimization
- [ ] **Documentation**: API versioning, migration guides

---

## Integration Guide

### For Dev A (Crawler Team)

**Your Responsibility**: Send messages to RabbitMQ queue

**Message Format**:
```json
{
    "image_sha256": "computed-sha256-hash",
    "bucket": "raw-images",
    "key": "tenant1/2024/10/image.jpg",
    "tenant_id": "acme-corp",
    "site": "example.com",
    "url": "https://example.com/photo.jpg",
    "image_phash": "computed-phash",
    "face_hints": null
}
```

**Contract**:
- Image must already be uploaded to MinIO
- SHA-256 and pHash must be pre-computed
- Queue name: `face-processing-queue`

---

### For Dev C (Frontend Team)

**Your Responsibility**: Call Search API endpoints

**Available Endpoints**:

1. **Search**: `POST /api/v1/search`
   - Upload image or send vector
   - Get similar faces
   
2. **Get Face**: `GET /api/v1/faces/{id}`
   - Retrieve face details
   - Get presigned thumbnail URL
   
3. **Stats**: `GET /api/v1/stats`
   - Display pipeline metrics

**OpenAPI Docs**: http://localhost:8000/docs

**TypeScript Client Generation**:
```bash
npx openapi-typescript http://localhost:8000/openapi.json --output api.d.ts
```

---

### For DevOps

**Health Checks**:
- Liveness: `GET /health` (always returns 200)
- Readiness: `GET /ready` (returns 200 when deps are ready)

**Kubernetes Deployment**:
```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

**Metrics**: (Coming in DEV2)
- Prometheus metrics at `/metrics`
- Timing histograms for all pipeline steps
- Success/failure counters

---

## Module Boundaries

### What Face-Pipeline Owns

- **Face detection logic** (InsightFace integration)
- **Quality assessment** (blur, size, pose, brightness)
- **Embedding generation** (512-dim vectors)
- **Deduplication** (pHash comparison)
- **Vector indexing** (Qdrant management)
- **Search API** (similarity search)

### What Face-Pipeline Does NOT Own

- **Image crawling** (owned by crawler service)
- **Raw image storage** (done before queue)
- **User management** (owned by main backend)
- **Authentication** (handled by API gateway)
- **Billing/quotas** (owned by main backend)

---

## File Structure

```
face-pipeline/
├── config/
│   └── settings.py           # All configuration
├── pipeline/
│   ├── processor.py          # Main orchestration (12 steps)
│   ├── detector.py           # Face detection
│   ├── embedder.py           # Embedding generation
│   ├── quality.py            # Quality assessment
│   ├── indexer.py            # Qdrant operations
│   ├── storage.py            # MinIO operations
│   └── utils.py              # Helpers (timer, pHash, etc.)
├── services/
│   └── search_api.py         # REST API endpoints
├── tests/
│   ├── test_quality.py       # Quality tests
│   ├── test_embedder.py      # Embedder tests
│   └── test_processor_integration.py  # Pipeline tests
├── main.py                   # FastAPI application
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container image
└── README.md                 # This file
```

---

## Support & Documentation

- **Development Status**: See `CONTEXT.md`
- **API Documentation**: http://localhost:8000/docs
- **Step-by-Step Guides**: See `STEP*.md` files
- **Test Infrastructure**: See `STEP11_TESTS_SUMMARY.md`
- **Observability**: See `STEP10_OBSERVABILITY_SUMMARY.md`

---

## Contributing

### Development Workflow

1. Read this README
2. Review `CONTEXT.md` for current status
3. Check `tests/` for interface contracts
4. Implement features following TODO markers
5. Run tests: `pytest tests/ -v`
6. Update documentation

### Code Style

- **Linting**: ruff
- **Formatting**: black
- **Type Hints**: Required for public functions
- **Docstrings**: Google style

---

**Last Updated**: October 14, 2025  
**Contact**: Dev Team B  
**Status**: Infrastructure Complete, Ready for DEV2 Implementation
