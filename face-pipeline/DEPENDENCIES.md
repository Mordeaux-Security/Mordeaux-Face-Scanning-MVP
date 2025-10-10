# Face Pipeline Dependencies

## 📦 Requirements & Makefile Setup Complete

### ✅ All Requested Packages Added:

```txt
fastapi          ✓ 0.115.0
uvicorn          ✓ 0.30.6 [standard]
pillow           ✓ 10.4.0 (Pillow)
numpy            ✓ 1.26.4
opencv-python-headless ✓ 4.10.0.84
qdrant-client    ✓ 1.10.1
minio            ✓ 7.2.9
pydantic         ✓ 2.9.2
python-multipart ✓ 0.0.9
loguru           ✓ 0.7.2
imagehash        ✓ 4.3.1
pytest           ✓ 8.3.3
black            ✓ 24.8.0
ruff             ✓ 0.6.8
```

### ✅ All Requested Makefile Targets:

```makefile
make run         ✓ uvicorn main:app --reload
make test        ✓ pytest
make format      ✓ black .
make lint        ✓ ruff .
```

**Plus 20+ additional helpful targets!** (see `make help`)

---

## 🛡️ Anti-Bloat Strategy

### Shared Dependencies with Backend

The following packages are shared between `backend/` and `face-pipeline/`:

| Package | Backend Version | Face-Pipeline Version | Status |
|---------|----------------|----------------------|--------|
| `fastapi` | 0.115.0 | 0.115.0 | ✅ Synced |
| `uvicorn[standard]` | 0.30.6 | 0.30.6 | ✅ Synced |
| `pydantic-settings` | 2.5.2 | 2.5.2 | ✅ Synced |
| `python-multipart` | 0.0.9 | 0.0.9 | ✅ Synced |
| `Pillow` | 10.4.0 | 10.4.0 | ✅ Synced |
| `numpy` | 1.26.4 | 1.26.4 | ✅ Synced |
| `opencv-python-headless` | 4.10.0.84 | 4.10.0.84 | ✅ Synced |
| `imagehash` | 4.3.1 | 4.3.1 | ✅ Synced |
| `minio` | 7.2.9 | 7.2.9 | ✅ Synced |
| `qdrant-client` | 1.10.1 | 1.10.1 | ✅ Synced |

### Intentionally NOT Duplicated

Heavy dependencies from backend that face-pipeline can reuse:

- **InsightFace** (0.7.3) - Face detection/embedding model
  - Status: ⚠️ Commented out in face-pipeline requirements
  - Reason: 350MB+ download, already in backend container
  - Solution: Face-pipeline imports from backend when run together

- **ONNX Runtime** (1.19.2) - Required for InsightFace
  - Status: ⚠️ Commented out
  - Reason: Large binary dependencies
  - Solution: Shared from backend

### Unique to Face-Pipeline

- **loguru** (0.7.2) - Enhanced logging (backend uses standard logging)
- **pytest-asyncio** (0.24.0) - Async test support
- **pytest-cov** (5.0.0) - Coverage reporting
- **black** (24.8.0) - Code formatter
- **ruff** (0.6.8) - Fast linter

---

## 📊 Installation Strategies

### Strategy 1: Standalone Face-Pipeline ⭐ (Recommended for Development)

Install face-pipeline independently for local development:

```bash
cd face-pipeline/
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
make install
make run
```

**Pros:**
- Fast iteration
- No Docker overhead
- Full IDE support

**Cons:**
- Need to uncomment InsightFace dependencies (adds 350MB+)
- More disk space

### Strategy 2: Use Backend Container (Recommended for Production)

Run face-pipeline using the backend's dependencies:

```bash
# From project root
docker compose up -d backend-cpu
docker compose exec backend-cpu bash

# Inside container
cd /app/face-pipeline
python main.py
```

**Pros:**
- No duplication of heavy deps
- Same environment as production
- Smaller total footprint

**Cons:**
- Slower iteration (need to rebuild)
- Docker overhead

### Strategy 3: Shared Virtual Environment

Create one venv for both:

```bash
# From project root
python -m venv venv
source venv/bin/activate

# Install backend deps (includes InsightFace)
pip install -r backend/requirements.txt

# Install face-pipeline extras
cd face-pipeline/
pip install pytest pytest-asyncio pytest-cov black ruff loguru

# Now you can run either
cd ../backend && uvicorn app.main:app --reload
# OR
cd ../face-pipeline && uvicorn main:app --reload
```

**Pros:**
- One environment for everything
- No duplication at all
- Easy to switch between projects

**Cons:**
- Potential version conflicts
- Larger venv

---

## 🔧 Makefile Features

### Quick Development Commands

```bash
make run          # Start dev server with hot-reload
make test         # Run tests
make format       # Auto-format code
make lint         # Check code quality
```

### Shortcuts

```bash
make r            # Same as 'make run'
make t            # Same as 'make test'
make f            # Same as 'make format'
make l            # Same as 'make lint'
```

### Code Quality

```bash
make format       # Format with black
make format-check # Check formatting only
make lint         # Lint with ruff
make lint-fix     # Auto-fix lint issues
make quality      # Run format + lint
```

### Testing

```bash
make test              # All tests
make test-cov          # With coverage HTML report
make test-unit         # Unit tests only
make test-integration  # Integration tests only
```

### Environment

```bash
make check-env    # Verify .env configuration
make shell        # Python REPL with imports
make deps-check   # Check for outdated packages
```

### CI/CD

```bash
make ci           # Clean, install, test, lint
make pre-commit   # Format, lint, test (before committing)
```

---

## 📝 Version Pinning Strategy

All versions are **pinned exactly** (using `==`) for:

1. **Reproducibility** - Same versions in dev and prod
2. **Security** - Know exactly what's running
3. **Compatibility** - Avoid surprise breaking changes

When updating dependencies:

```bash
# Check for updates
make deps-check

# Update specific package
pip install --upgrade fastapi
pip freeze | grep fastapi
# Copy new version to requirements.txt
```

---

## 🚀 Next Steps

1. **Choose your installation strategy** (see above)
2. **Install dependencies**: `make install`
3. **Verify setup**: `make check-env`
4. **Run tests**: `make test`
5. **Start coding!**: `make run`

---

## 📚 Related Files

- **Main**: `requirements.txt` - Production dependencies
- **Config**: `../.env.example` - Environment variables
- **Build**: `Makefile` - Development commands
- **Docker**: `Dockerfile` - Container build (TODO: implement)

---

**Last Updated**: After requirements.txt and Makefile implementation

