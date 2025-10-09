# Face Pipeline - Quick Start Guide

## 🚀 Step 0D Complete - Minimal App Ready!

The minimal FastAPI application is now implemented and ready to run.

### ✅ What's Implemented

1. **config/settings.py** - Pydantic BaseSettings parsing all env keys
   - All environment variables from `.env.example`
   - Defaults configured for local development
   - Support for both MINIO_* and S3_* prefixes

2. **services/search_api.py** - APIRouter with placeholder endpoints
   - `POST /api/v1/search` - Returns 501 + TODO
   - `GET /api/v1/faces/{face_id}` - Returns 501 + TODO
   - `GET /api/v1/stats` - Returns 501 + TODO
   - All endpoints have proper request/response models
   - Comprehensive docstrings with TODO notes

3. **main.py** - FastAPI app with health endpoint
   - `GET /health` - Returns `{"status": "ok"}` ✅
   - `GET /` - API information
   - `GET /info` - Configuration details
   - CORS middleware configured
   - Lifespan management for startup/shutdown
   - Error handlers
   - Auto-generated docs at `/docs`

---

## 📦 Installation

### Option 1: Local Development (Recommended)

```bash
# Navigate to face-pipeline
cd face-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
make install

# Verify installation
make check-env
```

### Option 2: Use Makefile (Quick)

```bash
cd face-pipeline
make install
```

---

## 🧪 Testing (Acceptance Criteria)

### Automated Test

Run the comprehensive test script:

```bash
cd face-pipeline
./test_app.sh
```

This will:
1. ✅ Verify Python syntax
2. ✅ Check dependencies
3. ✅ Start uvicorn server
4. ✅ Test `GET /health` returns 200 OK
5. ✅ Verify endpoints return 501 (not implemented)

### Manual Test

1. **Start the server**:
   ```bash
   cd face-pipeline
   make run
   # OR: uvicorn main:app --reload
   ```

2. **Test /health endpoint**:
   ```bash
   curl http://localhost:8000/health
   # Expected: {"status":"ok"}
   ```

3. **View API docs**:
   - Open browser: http://localhost:8000/docs
   - Interactive Swagger UI with all endpoints

4. **Test placeholder endpoints**:
   ```bash
   # Search (should return 501)
   curl -X POST http://localhost:8000/api/v1/search \
     -H "Content-Type: application/json" \
     -d '{"limit": 10}'
   
   # Get face by ID (should return 501)
   curl http://localhost:8000/api/v1/faces/test-id
   
   # Get stats (should return 501)
   curl http://localhost:8000/api/v1/stats
   ```

---

## 📚 API Documentation

### Base URLs

- **Local**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### ✅ Implemented

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| GET | `/` | API information | ✅ Working |
| GET | `/health` | Health check | ✅ Working |
| GET | `/info` | Configuration info | ✅ Working |
| GET | `/docs` | Swagger UI | ✅ Working |

#### 🚧 TODO (Returns 501)

| Method | Endpoint | Description | Status |
|--------|----------|-------------|--------|
| POST | `/api/v1/search` | Search for similar faces | 🚧 TODO |
| GET | `/api/v1/faces/{id}` | Get face by ID | 🚧 TODO |
| GET | `/api/v1/stats` | Pipeline statistics | 🚧 TODO |

---

## 🔧 Configuration

The app reads configuration from `../.env` (root level).

### Key Settings

```bash
# MinIO Storage
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=changeme
MINIO_SECRET_KEY=changeme

# Qdrant Vector DB
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=faces_v1

# Pipeline Config
MAX_CONCURRENT=4
FACE_MIN_SIZE=80
BLUR_MIN_VARIANCE=120.0

# API Server
API_HOST=0.0.0.0
API_PORT=8000
```

### Verify Configuration

```bash
make check-env
```

This will print loaded configuration values.

---

## 🐛 Troubleshooting

### "Module not found: pydantic"

Install dependencies:
```bash
make install
```

### "Address already in use"

Change the port:
```bash
uvicorn main:app --reload --port 8001
```

Or kill the process using port 8000:
```bash
lsof -ti:8000 | xargs kill -9
```

### "Settings validation error"

Check your `.env` file exists:
```bash
cp ../.env.example ../.env
```

Then edit values in `../.env`.

---

## 📝 Next Steps

Now that the minimal app is working, you can:

1. **Implement TODO endpoints**
   - Edit `services/search_api.py`
   - Replace 501 responses with actual logic

2. **Add pipeline components**
   - `pipeline/detector.py` - Face detection
   - `pipeline/embedder.py` - Face embedding
   - `pipeline/quality.py` - Quality assessment
   - `pipeline/storage.py` - MinIO integration
   - `pipeline/indexer.py` - Qdrant integration
   - `pipeline/processor.py` - Pipeline orchestration

3. **Write tests**
   - `tests/test_quality.py`
   - `tests/test_embedder.py`
   - `tests/test_processor_integration.py`

4. **Add features**
   - Batch processing
   - Background tasks
   - Metrics/monitoring
   - Authentication

---

## 🎯 Acceptance Criteria Status

✅ **All acceptance criteria met:**

1. ✅ `config/settings.py` - Pydantic BaseSettings with all env keys
2. ✅ `services/search_api.py` - APIRouter with /search, /faces/{id}, /stats returning 501
3. ✅ `main.py` - FastAPI app with router included and /health endpoint
4. ✅ `uvicorn main:app --reload` - Server starts successfully
5. ✅ `GET /health` - Returns `{"status": "ok"}`

---

## 📚 Additional Resources

- **Makefile**: Run `make help` to see all available commands
- **Dependencies**: See `DEPENDENCIES.md` for package information
- **Configuration**: See `CONFIGURATION.md` for env var details
- **API Docs**: http://localhost:8000/docs (when running)

---

**Status**: ✅ Step 0D Complete - Ready for implementation!

