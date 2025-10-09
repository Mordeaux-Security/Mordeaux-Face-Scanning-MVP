# Face Pipeline - Project Context for Continuation

## 📍 Current Status

**Project**: Mordeaux Face Scanning MVP - Face Pipeline Module  
**Branch**: `debloated`  
**Last Commit**: `42c0136` - "feat: Add face-pipeline module with minimal working API"  
**Workspace**: `/Users/lando/Mordeaux-Face-Scanning-MVP-2`

### ✅ Completed Steps (0A-0D)

- **Step 0A**: Project structure with placeholder modules ✅
- **Step 0B**: Configuration management (.env.example, settings.py) ✅
- **Step 0C**: Dependencies & Makefile (requirements.txt, 30+ targets) ✅
- **Step 0D**: Minimal working API (FastAPI app with /health endpoint) ✅

**Status**: 26 files created, 551+ lines of working code, all tests passing

---

## 🏗️ Project Structure

```
Mordeaux-Face-Scanning-MVP-2/
├── .env.example                    # Unified config (108 lines, all variables)
├── CONFIGURATION.md                # Config consolidation guide
├── backend/                        # Existing backend (DO NOT DUPLICATE)
│   ├── app/services/face.py       # Has InsightFace detection/embedding
│   ├── app/services/storage.py    # Has MinIO storage
│   └── app/services/crawler.py    # Has basic quality checks
├── face-pipeline/                  # NEW MODULE (our work)
│   ├── main.py                    # FastAPI app (180 lines) ✅ WORKING
│   ├── config/settings.py         # Pydantic settings (140 lines) ✅ WORKING
│   ├── services/search_api.py     # API router (200 lines) ✅ WORKING (501s)
│   ├── pipeline/                  # Components (ALL PLACEHOLDERS)
│   │   ├── detector.py           # TODO: Face detection
│   │   ├── embedder.py           # TODO: Face embedding
│   │   ├── quality.py            # TODO: Quality assessment
│   │   ├── storage.py            # TODO: Storage abstraction
│   │   ├── indexer.py            # TODO: Vector indexing
│   │   ├── processor.py          # TODO: Pipeline orchestration ← NEXT
│   │   └── utils.py              # TODO: Utility functions
│   ├── tests/                    # Test structure ready
│   ├── requirements.txt          # 14 packages, version-pinned
│   ├── Makefile                  # 30+ targets
│   └── README.md, QUICKSTART.md, DEPENDENCIES.md
```

---

## 🔧 Key Configuration

### Environment Variables (from .env.example at root)

```bash
# Storage - MinIO
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=changeme
MINIO_SECRET_KEY=changeme
MINIO_SECURE=false
MINIO_BUCKET_RAW=raw-images
MINIO_BUCKET_CROPS=face-crops
MINIO_BUCKET_THUMBS=thumbnails
MINIO_BUCKET_METADATA=face-metadata

# Vector DB - Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=faces_v1

# Pipeline Settings
MAX_CONCURRENT=4
JOB_TIMEOUT_SEC=300
FACE_MIN_SIZE=80
BLUR_MIN_VARIANCE=120.0
PRESIGN_TTL_SEC=600

# Face Detection
DETECTOR_MODEL=buffalo_l
DETECTOR_CTX_ID=-1
EMBEDDING_DIM=512

# Quality Thresholds
MIN_FACE_QUALITY=0.5
MIN_SHARPNESS=100.0
MIN_BRIGHTNESS=30.0
MAX_BRIGHTNESS=225.0
```

**Important**: Settings support both `MINIO_*` and `S3_*` prefixes for backward compatibility with docker-compose.yml

---

## 📦 Dependencies

### Installed Packages (face-pipeline/requirements.txt)
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
pydantic==2.9.2
pydantic-settings==2.5.2
python-multipart==0.0.9
Pillow==10.4.0
numpy==1.26.4
opencv-python-headless==4.10.0.84
imagehash==4.3.1
minio==7.2.9
qdrant-client==1.10.1
loguru==0.7.2
pytest==8.3.3
black==24.8.0
ruff==0.6.8
```

**Note**: InsightFace (0.7.3) and onnxruntime (1.19.2) are in backend, not duplicated.

---

## 🎯 Current API Endpoints

### ✅ Working
- `GET /health` → `{"status": "ok"}`
- `GET /` → API information
- `GET /info` → Configuration details
- `GET /docs` → Swagger UI

### 🚧 TODO (Return 501)
- `POST /api/v1/search` → Face similarity search
- `GET /api/v1/faces/{id}` → Get face by ID
- `GET /api/v1/stats` → Pipeline statistics

---

## 🚨 Anti-Duplication Strategy

### ❌ DO NOT Duplicate These (Already in backend/)

1. **Face Detection/Embedding**: `backend/app/services/face.py`
   - Has InsightFace (buffalo_l model)
   - Functions: `detect_and_embed()`, `compute_phash()`, `crop_face_from_image()`
   - **Action**: Import and use OR refactor into new modules

2. **Storage**: `backend/app/services/storage.py`
   - Has MinIO client setup
   - Functions: `save_raw_and_thumb()`, `save_raw_image_only()`, etc.
   - **Action**: Create adapter or extend

3. **Quality Checks**: `backend/app/services/crawler.py` (basic)
   - Has: `check_face_quality()` with min size/score checks
   - **Action**: Extract and enhance in new quality.py

### ✅ What's New/Enhanced in face-pipeline

1. **Advanced Quality Assessment** (quality.py)
   - Blur detection (Laplacian variance)
   - Brightness/contrast checks
   - Pose estimation
   - Occlusion detection
   - Comprehensive scoring

2. **Vector Indexing** (indexer.py)
   - Qdrant integration for similarity search
   - Batch indexing
   - Metadata filtering

3. **Pipeline Orchestration** (processor.py)
   - End-to-end workflow
   - Batch processing
   - Error handling

---

## 📝 Next Steps: Data Contract & Processor Entrypoint

### Task 1: Data Contract

Define data models for pipeline processing:

```python
# Pipeline input/output models
class FaceDetectionResult:
    bbox: List[float]  # [x1, y1, x2, y2]
    landmarks: Optional[np.ndarray]
    det_score: float
    
class FaceProcessingResult:
    face_id: str
    image_id: str
    bbox: List[float]
    embedding: List[float]
    quality_score: float
    storage_keys: Dict[str, str]  # raw_key, crop_key, thumb_key
    metadata: Dict[str, Any]
    
class PipelineInput:
    image_bytes: bytes
    image_id: Optional[str]
    metadata: Optional[Dict]
    
class PipelineOutput:
    success: bool
    faces_detected: int
    faces_processed: int
    results: List[FaceProcessingResult]
    errors: List[str]
    processing_time_ms: float
```

**Location**: `pipeline/models.py` (new file) OR in existing modules

### Task 2: Processor Entrypoint

Implement `pipeline/processor.py`:

```python
class FacePipelineProcessor:
    async def process_image(
        self, 
        image_bytes: bytes,
        image_id: Optional[str] = None
    ) -> PipelineOutput:
        """
        Main entrypoint for face processing pipeline.
        
        Steps:
        1. Detect faces (use backend's face.py or implement)
        2. Assess quality (new quality.py)
        3. Crop faces
        4. Generate embeddings (use backend's face.py or implement)
        5. Store images (use backend's storage.py or new storage.py)
        6. Index embeddings (new indexer.py with Qdrant)
        7. Return results
        """
```

**Integration Points**:
- Can reuse `backend/app/services/face.py` functions
- Can reuse `backend/app/services/storage.py` functions
- Add new quality checks, indexing, orchestration

---

## 🛠️ Development Commands

```bash
# Navigate to face-pipeline
cd /Users/lando/Mordeaux-Face-Scanning-MVP-2/face-pipeline

# Install dependencies
make install

# Run server
make run
# OR: uvicorn main:app --reload

# Test
make test

# Format & lint
make format
make lint

# Check environment
make check-env
```

---

## 📚 Key Files to Reference

### Configuration
- **Root**: `.env.example` - All environment variables
- **Face-pipeline**: `config/settings.py` - Pydantic settings with all vars loaded

### Existing Backend Code (DO NOT DUPLICATE)
- `backend/app/services/face.py` - Face detection/embedding (InsightFace)
- `backend/app/services/storage.py` - MinIO storage client
- `backend/app/services/crawler.py` - Basic quality checks (lines 418-477)

### Documentation
- `face-pipeline/README.md` - Project overview, duplicate analysis
- `face-pipeline/QUICKSTART.md` - Testing guide
- `face-pipeline/DEPENDENCIES.md` - Package info
- `CONFIGURATION.md` - Config consolidation strategy

---

## 🎨 Code Style & Conventions

### Imports
```python
# Standard library
import logging
from typing import List, Optional, Dict, Any

# Third-party
from fastapi import APIRouter
from pydantic import BaseModel, Field
import numpy as np

# Local
from config.settings import settings
from pipeline.utils import some_function
```

### Logging
```python
from loguru import logger  # Use loguru in face-pipeline
# OR
import logging
logger = logging.getLogger(__name__)
```

### Async/Await
- Prefer async functions for I/O operations
- Use thread pool for CPU-intensive (face detection, embedding)

### Error Handling
```python
try:
    result = await process()
except Exception as e:
    logger.error(f"Error: {e}")
    # Handle gracefully
```

---

## 🔑 Important Decisions Made

1. **Config Consolidation**: Single `.env.example` at root, not per-module
2. **Naming Convention**: MINIO_* preferred, but support S3_* for compatibility
3. **Dependency Strategy**: Reuse backend's heavy deps (InsightFace), add unique tools
4. **Module Design**: Modular components that can work together or independently
5. **Quality Focus**: Enhanced quality assessment beyond basic checks
6. **Vector Search**: Qdrant as primary vector DB (Pinecone as alternative)

---

## 🎯 Success Criteria for Next Tasks

### Data Contract
- [ ] Define all input/output models with Pydantic
- [ ] Type hints for all fields
- [ ] Validation rules where applicable
- [ ] Clear documentation

### Processor Entrypoint
- [ ] Implement `process_image()` method
- [ ] Integrate with existing backend services (face.py, storage.py)
- [ ] Add quality assessment
- [ ] Store results in Qdrant
- [ ] Return structured PipelineOutput
- [ ] Handle errors gracefully
- [ ] Add logging

---

## 🚀 Quick Context Summary

**What we have**: Complete FastAPI app structure, settings, placeholder modules, all dependencies installed, docs written.

**What's next**: Define data models (contracts) and implement the main processor that orchestrates the entire face processing pipeline.

**Where to work**: `face-pipeline/` directory, specifically `pipeline/processor.py` and potentially `pipeline/models.py`.

**How to avoid duplication**: Import from `backend/app/services/` where possible (face.py for detection/embedding, storage.py for MinIO).

**Testing**: Use `make test` and `./test_app.sh` to verify changes.

---

**Last Updated**: After Step 0D completion and git push to origin/debloated (commit 42c0136)

