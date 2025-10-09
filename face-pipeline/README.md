# Face Pipeline - Modular Face Processing System

A modular, production-ready face processing pipeline for detection, embedding, quality assessment, storage, and similarity search.

## 📁 Project Structure

```
face-pipeline/
├── main.py                          # Main entry point and orchestration
├── config/
│   └── settings.py                  # Configuration management with pydantic-settings
├── services/
│   └── search_api.py                # Face similarity search API
├── pipeline/
│   ├── __init__.py                  # Pipeline module exports
│   ├── detector.py                  # Face detection using InsightFace
│   ├── embedder.py                  # Face embedding/encoding
│   ├── quality.py                   # Quality assessment (blur, brightness, pose, etc.)
│   ├── storage.py                   # Storage abstraction (MinIO/S3)
│   ├── indexer.py                   # Vector database indexing for search
│   ├── processor.py                 # End-to-end pipeline orchestration
│   └── utils.py                     # Utility functions
├── tests/
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_quality.py              # Quality checker tests
│   ├── test_embedder.py             # Embedder tests
│   └── test_processor_integration.py # Integration tests
├── Dockerfile                        # Production container image
├── requirements.txt                  # Python dependencies
├── Makefile                         # Common development commands
└── env.example                      # Environment configuration template
```

## ⚠️ Potential Duplicates with Existing Codebase

This face-pipeline structure was created to modularize and optimize face processing. Here are potential overlaps with your existing code:

### 🔴 **DEFINITE DUPLICATES** (Refactor Recommended)

1. **`pipeline/detector.py` ↔️ `backend/app/services/face.py`**
   - Both implement face detection using InsightFace
   - Recommendation: Refactor `face.py` to use the new `detector.py` module
   - New module adds: batch detection, async support, better error handling

2. **`pipeline/embedder.py` ↔️ `backend/app/services/face.py`**
   - Both implement face embedding extraction
   - Recommendation: Refactor `face.py` to use the new `embedder.py` module
   - New module adds: batch embedding, normalization options, similarity computation

3. **`pipeline/storage.py` ↔️ `backend/app/services/storage.py`**
   - Both implement MinIO storage management
   - Recommendation: Create adapter to use existing `storage.py` or merge functionality
   - New module adds: metadata storage, multiple backend support, versioning

### 🟡 **PARTIAL DUPLICATES** (Enhancement Recommended)

4. **`pipeline/quality.py` ↔️ `backend/app/services/crawler.py`**
   - `crawler.py` has basic quality checks (face size, detection score)
   - New module adds: blur detection, brightness, contrast, pose estimation, occlusion detection
   - Recommendation: Extract quality logic from crawler, use new quality module

5. **`pipeline/processor.py` ↔️ `backend/app/services/crawler.py`**
   - Both orchestrate image processing workflows
   - `crawler.py` focuses on web crawling + face processing
   - `processor.py` focuses on pure face processing pipeline
   - Recommendation: Keep both, but have crawler use processor for face operations

### 🟢 **NEW FUNCTIONALITY** (No Duplicates)

6. **`pipeline/indexer.py`** - NEW
   - Vector database indexing for fast similarity search
   - No existing equivalent in codebase

7. **`services/search_api.py`** - NEW
   - Face similarity search API
   - No existing equivalent in codebase

8. **`pipeline/utils.py`** - NEW
   - Shared utility functions for pipeline
   - Complements existing utilities

## 🎯 Recommended Integration Strategy

### Option 1: Gradual Refactor (Recommended)
1. Keep existing code working
2. Implement new pipeline modules one by one
3. Create adapters to bridge old and new code
4. Gradually migrate functionality to new modules
5. Deprecate old code once migration is complete

### Option 2: Parallel Development
1. Keep both systems running
2. Use new pipeline for new features
3. Gradually migrate old features to new pipeline
4. Eventually consolidate

### Option 3: Clean Break
1. Implement all new modules
2. Update all existing services to use new pipeline
3. Delete old duplicate code
4. Higher risk but cleaner result

## 🚀 Next Steps

1. **Implement Core Modules** (Priority Order):
   - [ ] `detector.py` - Face detection (refactor from existing `face.py`)
   - [ ] `embedder.py` - Face embedding (refactor from existing `face.py`)
   - [ ] `quality.py` - Quality assessment (new + enhanced)
   - [ ] `storage.py` - Storage management (adapter for existing storage)
   - [ ] `indexer.py` - Vector indexing (completely new)
   - [ ] `processor.py` - Pipeline orchestration
   - [ ] `search_api.py` - Search API (completely new)

2. **Testing**:
   - [ ] Write unit tests for each module
   - [ ] Write integration tests for pipeline
   - [ ] Performance benchmarking

3. **Integration**:
   - [ ] Create adapter layer for existing services
   - [ ] Update crawler to use new quality module
   - [ ] Integrate vector search into main API

## 📝 TODO Status

All files currently contain TODO placeholders. No business logic has been implemented yet.

## ⚙️ Configuration

Copy `env.example` to `.env` and configure your settings:
- Face detection model and parameters
- Quality thresholds
- Storage backend (MinIO/S3)
- Vector database (Qdrant/Pinecone/ChromaDB/FAISS)
- Pipeline behavior
- API settings

## 🧪 Testing

```bash
make test              # Run all tests
make test-unit         # Run unit tests only
make test-integration  # Run integration tests only
```

## 🐳 Docker

```bash
make docker-build      # Build Docker image
make docker-run        # Run container
```

---

**Status**: 🏗️ Structure created, awaiting implementation

