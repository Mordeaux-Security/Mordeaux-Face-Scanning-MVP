# Face Pipeline - Simple Guide

## 🔄 How It Works

```
1. CRAWLER downloads images
   └─► Saves to MinIO (raw-images bucket)
   
2. AUTO-VECTORIZATION triggers
   └─► Calls /api/v1/ingest/batch
   
3. REDIS STREAMS queues message
   └─► Stream: "face:ingest"
   
4. WORKER processes message
   └─► Calls process_image()
   
5. PIPELINE processes image:
   ├─► Detect faces
   ├─► Crop & align faces
   ├─► Check quality
   ├─► Check duplicates
   ├─► Generate 512-dim vector
   ├─► Save face crops to MinIO
   └─► Save vector to Qdrant
```

---

## 📁 Key Files

### **Crawler (Auto-Vectorization)**
- `backend/app/services/crawler.py`
  - `crawl_page()` - Crawl single page
  - `_trigger_vectorization()` - Auto-vectorization after crawl

### **Ingest API**
- `backend/app/api/routes.py`
  - `/api/v1/ingest/batch` - Batch ingest endpoint

### **Face Pipeline**
- `face-pipeline/worker.py` - Consumes Redis, calls processor
- `face-pipeline/pipeline/processor.py` - Main processing function
- `face-pipeline/pipeline/detector.py` - Face detection
- `face-pipeline/pipeline/embedder.py` - Generate 512-dim vector
- `face-pipeline/pipeline/quality.py` - Quality check
- `face-pipeline/pipeline/dedup.py` - Duplicate check
- `face-pipeline/pipeline/indexer.py` - Qdrant upsert/search
- `face-pipeline/pipeline/storage.py` - MinIO save/load

---

## 🗄️ Storage

**MinIO Buckets:**
- `raw-images/` - Original images
- `face-crops/` - Cropped faces
- `face-thumbs/` - Face thumbnails

**Qdrant:**
- Collection: `faces_v1`
- Vector: 512-dim float32
- Payload: metadata (tenant_id, image_sha256, crop_key, etc.)

---

## 🔧 Configuration

**Settings:**
- `face-pipeline/config/settings.py` - All configs
- `face-pipeline/face_quality.py` - Quality thresholds

**Environment:**
- `QDRANT_URL` - Qdrant server
- `REDIS_STREAM_NAME` - Stream name (default: "face:ingest")
- `CRAWLER_AUTO_VECTORIZATION` - Enable auto-vectorization (default: true)

---

## 🎯 Quick Reference

| What | Where |
|------|-------|
| Change detection | `face-pipeline/pipeline/detector.py` |
| Change embedding | `face-pipeline/pipeline/embedder.py` |
| Change quality | `face-pipeline/face_quality.py` |
| Change dedup | `face-pipeline/pipeline/dedup.py` |
| Change Qdrant | `face-pipeline/pipeline/indexer.py` |
| Change crawler | `backend/app/services/crawler.py` |


