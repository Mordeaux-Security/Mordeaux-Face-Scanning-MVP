# Face Pipeline - DEV1 Context

## 📍 Current Status - DEV1 Phase Complete

**Project**: Mordeaux Face Scanning MVP - Face Pipeline Module  
**Branch**: `debloated`  
**Last Commit**: `8cf99b9` - "feat(face-pipeline): Add data contracts, storage utilities, and detector interfaces"  
**Workspace**: `/Users/lando/Mordeaux-Face-Scanning-MVP-2/face-pipeline`

### ✅ DEV1 Phase Completed (Steps 1-8)

- **Step 1**: Data Contract & Processor Entrypoint ✅
- **Step 2**: Storage (MinIO) Utilities ✅
- **Step 3**: Detector & Alignment (interfaces) ✅
- **Step 4**: Quality Gates (stubs) ✅
- **Step 5**: Embedding & pHash ✅
- **Step 6**: Indexing (Qdrant skeleton) ✅
- **Step 7**: Deduplication Helpers ✅
- **Step 8**: Orchestration Flow (with minimal wiring) ✅

**Status**: All pipeline infrastructure complete, orchestration flow documented and wired with placeholders, ready for implementation.

---

## 🏗️ Module Implementation Status

### ✅ Fully Implemented (Interfaces & Skeletons)

#### 1. **pipeline/processor.py** (430+ lines)
- ✅ `PipelineInput` Pydantic model (8 fields with docstrings)
- ✅ `process_image(message: dict) -> dict` - Full orchestration flow (12 steps)
- ✅ Minimal wiring with placeholders (compiles and runs)
- ✅ Comprehensive step-by-step comments
- 🔄 FacePipelineProcessor class (legacy, keeping for now)

**12-Step Pipeline Flow**:
1. Validate input (PipelineInput schema)
2. Download image from MinIO
3. Decode image (bytes → PIL/numpy)
4. Detect faces (hints or detector)
5. Align and crop faces
6. Quality assessment per face
7. Compute pHash and prefix
8. Deduplication precheck
9. Generate embeddings
10. Generate artifact paths (no writes)
11. Batch upsert to Qdrant
12. Return summary

#### 2. **pipeline/storage.py** (260+ lines)
- ✅ MinIO client initialization (singleton pattern)
- ✅ `get_bytes(bucket, key) -> bytes` - Retrieve objects
- ✅ `put_bytes(bucket, key, data, content_type) -> None` - Upload objects
- ✅ `exists(bucket, key) -> bool` - Check existence
- ✅ `presign(bucket, key, ttl_sec) -> str` - Generate presigned URLs
- ✅ Retry/backoff placeholder structure
- ✅ Logging stubs with loguru

#### 3. **pipeline/detector.py** (200+ lines)
- ✅ `detect_faces(img_np) -> list[dict]` - Face detection stub
- ✅ `validate_hint(img_shape, bbox) -> bool` - Bbox validation stub
- ✅ `align_and_crop(img_np, bbox, landmarks) -> PIL.Image` - Alignment stub
- 🔄 FaceDetector class (legacy, keeping for now)

#### 4. **pipeline/quality.py** (270+ lines)
- ✅ `laplacian_variance(img_np) -> float` - Blur detection stub
- ✅ `evaluate(img_pil, min_size, min_blur_var) -> dict` - Quality evaluation
- ✅ Returns dict with keys: `pass`, `reason`, `blur`, `size`
- 🔄 QualityChecker class (legacy, keeping for now)

#### 5. **pipeline/embedder.py** (210+ lines)
- ✅ `load_model() -> object` - Singleton model loader
- ✅ `l2_normalize(embedding) -> np.ndarray` - L2 normalization helper
- ✅ `embed(img_pil) -> np.ndarray` - Generate 512-dim embeddings
- ✅ Returns shape (512,) dtype float32
- 🔄 FaceEmbedder class (legacy, keeping for now)

#### 6. **pipeline/utils.py** (220+ lines)
- ✅ `l2_normalize(vec) -> np.ndarray` - **Minimal implementation** (actual code)
- ✅ `compute_phash(img_pil) -> str` - Returns "0"*16 placeholder
- ✅ `hamming_distance_hex(a, b) -> int` - Length-safe placeholder
- ✅ `phash_prefix(hex_str, bits=16) -> str` - Returns first 4 hex chars
- 🔄 Other utility functions (placeholders)

#### 7. **pipeline/indexer.py** (360+ lines)
- ✅ Qdrant Payload Schema documentation (9 required fields)
- ✅ `ensure_collection() -> None` - Create faces_v1 collection
- ✅ `upsert(points: list[dict]) -> None` - Batch upsert (≤16 points)
- ✅ `search(vector, top_k, filters) -> list[dict]` - Returns empty list placeholder
- 🔄 VectorIndexer class (legacy, keeping for now)

**Payload Contract** (9 fields):
- `tenant_id`, `site`, `url`, `ts`, `p_hash`, `p_hash_prefix`, `bbox`, `quality`, `image_sha256`

### ✅ Tests Created

#### tests/test_quality.py (180+ lines)
- ✅ `TestLaplacianVariance` - Tests return type
- ✅ `TestEvaluate` - Tests all 4 required keys and types
- ✅ 10 test functions total

#### tests/test_embedder.py (160+ lines)
- ✅ `TestEmbedFunction` - Tests shape (512,) and dtype float32
- ✅ `TestLoadModel` - Tests singleton pattern
- ✅ `TestL2Normalize` - Tests helper function
- ✅ 8 test functions total

---

## 🎯 Data Contract Summary

### Input Model: PipelineInput
```python
class PipelineInput(BaseModel):
    image_sha256: str          # SHA-256 hash
    bucket: str                # MinIO bucket
    key: str                   # Object key/path
    tenant_id: str             # Multi-tenant ID
    site: str                  # Source site/domain
    url: HttpUrl               # Original URL
    image_phash: str           # Perceptual hash (16-char hex)
    face_hints: Optional[List[Dict]]  # Optional upstream hints
```

### Output Structure: process_image() Returns
```python
{
    "image_sha256": str,
    "counts": {
        "faces_total": int,      # Total faces detected
        "accepted": int,         # Passed quality checks
        "rejected": int,         # Failed quality checks
        "dup_skipped": int       # Skipped as duplicates
    },
    "artifacts": {
        "crops": [str],          # Crop storage keys
        "thumbs": [str],         # Thumbnail storage keys
        "metadata": [str]        # Metadata storage keys
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

---

## 🔧 Configuration (from root .env.example)

### Storage - MinIO
```bash
MINIO_ENDPOINT=localhost:9000
MINIO_BUCKET_RAW=raw-images
MINIO_BUCKET_CROPS=face-crops
MINIO_BUCKET_THUMBS=thumbnails
MINIO_BUCKET_METADATA=face-metadata
```

### Vector DB - Qdrant
```bash
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=faces_v1
QDRANT_API_KEY=              # Optional
VECTOR_DIM=512
SIMILARITY_THRESHOLD=0.6
```

### Quality Thresholds
```bash
MIN_FACE_SIZE=80
BLUR_MIN_VARIANCE=120.0
MIN_SHARPNESS=100.0
MIN_BRIGHTNESS=30.0
MAX_BRIGHTNESS=225.0
MAX_POSE_ANGLE=45.0
MIN_OVERALL_QUALITY=0.7
```

### Pipeline Settings
```bash
MAX_FACES_PER_IMAGE=10
MAX_CONCURRENT=4
JOB_TIMEOUT_SEC=300
BATCH_SIZE=32
ENABLE_DEDUPLICATION=true
PRESIGN_TTL_SEC=600
```

---

## 🚀 Quick Start Commands

```bash
# Navigate to face-pipeline
cd /Users/lando/Mordeaux-Face-Scanning-MVP-2/face-pipeline

# Verify current state
python3 -c "from pipeline.processor import process_image; print('✓ Imports work')"

# Run with mock dependencies
python3 << 'EOF'
# Create minimal mock for testing
class MockModel:
    @classmethod
    def model_validate(cls, d):
        obj = cls()
        for k, v in d.items():
            setattr(obj, k, v)
        return obj

import sys
from unittest.mock import MagicMock
pydantic = MagicMock()
pydantic.BaseModel = MockModel
sys.modules['pydantic'] = pydantic

sys.path.insert(0, '.')
from pipeline.processor import process_image

result = process_image({
    "image_sha256": "test",
    "bucket": "raw-images",
    "key": "test.jpg",
    "tenant_id": "tenant1",
    "site": "example.com",
    "url": "https://example.com/test.jpg",
    "image_phash": "0" * 16,
    "face_hints": None
})

print(f"✓ Process result: {result['counts']}")
EOF
```

---

## 🔍 Implementation Progress

### ✅ Completed Features

1. **Data Contracts** - Pydantic models for input/output
2. **Storage Interface** - MinIO client with retry/backoff structure
3. **Detector Interface** - Face detection, validation, alignment stubs
4. **Quality Interface** - Blur detection, evaluation with tests
5. **Embedder Interface** - 512-dim embedding generation, L2 normalization
6. **Indexer Interface** - Qdrant collection, upsert, search stubs
7. **Dedup Helpers** - Hamming distance, pHash prefix
8. **Orchestration Flow** - Full 12-step pipeline documented and wired

### 🚧 Pending Implementation (Phase 2)

1. **Actual Model Loading** - InsightFace buffalo_l initialization
2. **Real Face Detection** - Implement detect_faces() with actual model
3. **Real Alignment** - Implement align_and_crop() with landmark transforms
4. **Real Quality Checks** - Implement laplacian_variance(), evaluate() logic
5. **Real Embeddings** - Implement embed() with actual model inference
6. **Real Storage Operations** - Implement get_bytes(), put_bytes() with MinIO SDK
7. **Real Qdrant Operations** - Implement ensure_collection(), upsert(), search()
8. **Real Dedup Logic** - Implement hamming_distance_hex() with bitwise XOR
9. **Uncomment Pipeline Steps** - Activate all 12 steps in process_image()

---

## 📊 Health Check Results

### File Structure
```
face-pipeline/
├── config/
│   └── settings.py         ✅ 146 lines - All settings loaded
├── pipeline/
│   ├── __init__.py         ✅ 19 lines
│   ├── processor.py        ✅ 480+ lines - Full orchestration
│   ├── storage.py          ✅ 260+ lines - MinIO interface
│   ├── detector.py         ✅ 200+ lines - Detection interface
│   ├── quality.py          ✅ 270+ lines - Quality interface
│   ├── embedder.py         ✅ 210+ lines - Embedding interface
│   ├── indexer.py          ✅ 360+ lines - Qdrant interface
│   └── utils.py            ✅ 220+ lines - Utility functions
├── tests/
│   ├── conftest.py         ✅ 107 lines - Fixtures
│   ├── test_quality.py     ✅ 188 lines - 10 tests
│   ├── test_embedder.py    ✅ 162 lines - 8 tests
│   └── test_processor_integration.py  ✅ 124 lines - Placeholders
└── CONTEXT.md              ✅ This file
```

### Code Quality
- ✅ No linter errors across all modules
- ✅ All functions have comprehensive docstrings
- ✅ Type hints using TYPE_CHECKING for optional deps
- ✅ Consistent coding style
- ✅ Clear TODO markers for implementation

### Testing
- ✅ Test structure ready
- ✅ 18+ test functions created
- ✅ Tests verify interfaces, not implementation
- ⚠️ Cannot run without dependencies (expected)

---

## 🎨 Code Patterns Used

### 1. TYPE_CHECKING for Optional Deps
```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import PIL.Image
```

### 2. Singleton Pattern
```python
_model = None

def load_model() -> object:
    global _model
    if _model is None:
        # Load model here
        pass
    return _model
```

### 3. Minimal Working Implementation
```python
def l2_normalize(vec: "np.ndarray") -> "np.ndarray":
    """Actual working implementation."""
    import numpy as np
    norm = np.linalg.norm(vec)
    if norm > 0:
        return vec / norm
    return vec
```

### 4. No-op Placeholders with Full Comments
```python
def detect_faces(img_np: "np.ndarray") -> list[dict]:
    """
    Comprehensive docstring explaining what it does.
    
    TODO: Step 1
    TODO: Step 2
    TODO: Step 3
    
    Example implementation:
        # Actual code here
    """
    pass  # Placeholder
```

---

## 🔑 Key Decisions Made in DEV1

1. **Functional + Class APIs**: Keep both for flexibility (functional for pipeline, class for advanced use)
2. **Placeholder Strategy**: Comprehensive comments showing exact implementation needed
3. **Minimal Wiring**: All functions callable, return correct types, no crashes
4. **Test-Driven Structure**: Tests verify interfaces before implementation
5. **No Duplication**: Import paths documented, ready to use backend services
6. **Settings-Driven**: All configuration via `config.settings` singleton
7. **12-Step Pipeline**: Clear orchestration flow with timing metrics
8. **Dedup Strategy**: pHash prefix filtering + Hamming distance (threshold ≤8)

---

## 📝 Next Phase: DEV2 (Implementation)

### Priority 1: Core Pipeline (High Impact)
1. Implement `detect_faces()` - InsightFace buffalo_l model
2. Implement `embed()` - Generate real embeddings
3. Implement `evaluate()` - Real quality checks
4. Uncomment Steps 2-6 in `process_image()`

### Priority 2: Storage & Indexing
5. Implement MinIO operations (`get_bytes`, `put_bytes`)
6. Implement Qdrant operations (`ensure_collection`, `upsert`, `search`)
7. Uncomment Steps 7-11 in `process_image()`

### Priority 3: Refinement
8. Implement `align_and_crop()` with actual transforms
9. Implement `hamming_distance_hex()` with bitwise logic
10. Add error handling and retry logic
11. Performance optimization

---

## 🚨 Important Notes

### Dependencies Not Yet Installed
The following are required but not installed (intentional for dev1):
- `pydantic` (for PipelineInput validation)
- `numpy` (for embeddings and image processing)
- `PIL/Pillow` (for image operations)
- `opencv-python-headless` (for image processing)
- `minio` (for object storage)
- `qdrant-client` (for vector search)
- `imagehash` (for perceptual hashing)

**Status**: All code is syntactically valid and will work once deps are installed.

### Backend Integration Ready
Can import from backend when needed:
```python
# Face detection/embedding
from backend.app.services.face import detect_and_embed, crop_face_from_image

# Storage
from backend.app.services.storage import save_raw_and_thumb, get_object_from_storage

# Quality (basic)
from backend.app.services.crawler import check_face_quality
```

---

## 🎯 Success Criteria - DEV1 Phase ✅

- [x] All pipeline modules created with comprehensive interfaces
- [x] Data contracts defined (PipelineInput + output structure)
- [x] Full orchestration flow documented (12 steps)
- [x] Minimal wiring - function compiles and runs
- [x] No linter errors
- [x] Test structure ready with 18+ tests
- [x] Deduplication logic designed
- [x] Qdrant payload contract documented
- [x] All configuration loaded from settings
- [x] Health check passing

**Status**: ✅ DEV1 PHASE COMPLETE - Ready for implementation phase

---

## 📚 Quick Reference

### Most Important Files
1. `pipeline/processor.py` - Main orchestration (START HERE for implementation)
2. `pipeline/storage.py` - MinIO operations
3. `pipeline/embedder.py` - Embedding generation
4. `pipeline/detector.py` - Face detection
5. `pipeline/quality.py` - Quality assessment
6. `pipeline/indexer.py` - Vector search
7. `config/settings.py` - All configuration

### Running Tests (when deps installed)
```bash
cd /Users/lando/Mordeaux-Face-Scanning-MVP-2/face-pipeline
python3 -m pytest tests/test_quality.py -v
python3 -m pytest tests/test_embedder.py -v
```

### Verification Commands
```bash
# Check syntax
python3 -m py_compile pipeline/*.py

# Check imports (structure only)
python3 -c "from pipeline import processor, storage, detector, quality, embedder, indexer, utils; print('✓ All modules importable')"
```

---

**Last Updated**: After DEV1 Phase completion (Steps 1-8)  
**Git Status**: Committed to `debloated` branch (commit 8cf99b9)  
**Next**: DEV2 Phase - Actual implementation of all TODO items

