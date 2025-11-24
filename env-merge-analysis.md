# Environment File Merge Analysis

## Major Conflicts & Differences

### 1. **S3_SECRET_KEY** - CONFLICT
- **env.txt (current)**: `S3_SECRET_KEY=minioadmin123`
- **env copy.txt (main)**: `S3_SECRET_KEY=minioadmin`
- **Resolution**: Use `minioadmin` (main branch) - matches S3_ACCESS_KEY and is standard MinIO default

### 2. **POSTGRES_PASSWORD** - CONFLICT
- **env.txt (current)**: `POSTGRES_PASSWORD=mordeaux123`
- **env copy.txt (main)**: `POSTGRES_PASSWORD=your_secure_password_here`
- **Resolution**: Keep current value `mordeaux123` (already in use) but note main uses placeholder

### 3. **VECTOR_INDEX** - CONFLICT
- **env.txt (current)**: `VECTOR_INDEX=faces`
- **env copy.txt (main)**: `VECTOR_INDEX=faces_v1`
- **Resolution**: Use `faces_v1` (main branch) - more specific versioning

### 4. **PINECONE_INDEX** - CONFLICT
- **env.txt (current)**: `PINECONE_INDEX=faces`
- **env copy.txt (main)**: `PINECONE_INDEX=faces_v1`
- **Resolution**: Use `faces_v1` (main branch) - matches VECTOR_INDEX

### 5. **VITE_API_BASE** - CONFLICT
- **env.txt (current)**: `VITE_API_BASE=/api`
- **env copy.txt (main)**: `VITE_API_BASE=http://localhost:8000`
- **Resolution**: Use `/api` (current) - works with proxy setup, but main uses full URL

### 6. **Face Detection Configuration** - MAJOR STRUCTURAL DIFFERENCE
- **env.txt (current)**: Uses old naming:
  - `FACE_MODEL_NAME=buffalo_l`
  - `FACE_DETECTION_SIZE_WIDTH=1024`
  - `FACE_DETECTION_SIZE_HEIGHT=1024`
  - `MIN_FACE_DETECTION_SCORE=0.6`
  - `MIN_FACE_QUALITY_SCORE=0.5`
  - `MIN_SIMILARITY_THRESHOLD=0.4`
  - `MIN_CONFIDENCE_THRESHOLD=0.4`

- **env copy.txt (main)**: Uses new naming:
  - `DETECTOR_MODEL=buffalo_l`
  - `DETECTOR_CTX_ID=-1`
  - `DETECTOR_SIZE_WIDTH=640`
  - `DETECTOR_SIZE_HEIGHT=640`
  - `EMBEDDING_DIM=512`
  - `NORMALIZE_EMBEDDINGS=true`
  - `MIN_FACE_QUALITY=0.5`
  - `MIN_FACE_SIZE=50`
  - `MAX_BLUR_SCORE=100.0`
  - `MIN_SHARPNESS=100.0`
  - `MIN_BRIGHTNESS=30.0`
  - `MAX_BRIGHTNESS=225.0`
  - `MAX_POSE_ANGLE=45.0`
  - `MIN_OVERALL_QUALITY=0.7`

- **Resolution**: Merge both - keep new naming from main, but preserve GPU-specific settings from current

### 7. **GPU Configuration** - CURRENT BRANCH ONLY
- **env.txt (current)**: Has extensive GPU configuration:
  - `GPU_TYPE=directml`
  - `GPU_BACKEND=directml`
  - `ALL_GPU=false`
  - `FACE_DETECTION_GPU=true`
  - `FACE_EMBEDDING_GPU=true`
  - `IMAGE_PROCESSING_GPU=false`
  - `IMAGE_ENHANCEMENT_GPU=false`
  - `QUALITY_CHECKS_GPU=false`
  - `GPU_DEVICE_ID=0`
  - `GPU_MEMORY_LIMIT_GB=8`
  - `GPU_BATCH_SIZE=128`
  - `GPU_WORKER_ENABLED=true`
  - `GPU_WORKER_URL=http://host.docker.internal:8765`
  - `GPU_WORKER_TIMEOUT=60.0`
  - `GPU_WORKER_MAX_RETRIES=3`

- **env copy.txt (main)**: No GPU configuration
- **Resolution**: KEEP ALL GPU settings from current branch (critical for GPU worker functionality)

### 8. **Dynamic Resource Management** - CURRENT BRANCH ONLY
- **env.txt (current)**: Has extensive dynamic resource management:
  - `ENABLE_DYNAMIC_RESOURCES=true`
  - `ADJUSTMENT_INTERVAL_S=0.3`
  - `TARGET_CPU_UTILIZATION=90.0`
  - `TARGET_GPU_UTILIZATION=92.0`
  - `TARGET_MEMORY_UTILIZATION=75.0`
  - `MIN_CONCURRENT_DOWNLOADS=20`
  - `MAX_CONCURRENT_DOWNLOADS=200`
  - `MIN_BATCH_SIZE=10`
  - `MAX_BATCH_SIZE=256`
  - `MIN_CONCURRENT_SITES=5`
  - `MAX_CONCURRENT_SITES=30`
  - `AGGRESSIVE_SCALING=false`
  - `SMOOTHING_FACTOR=0.3`
  - `ADJUSTMENT_STEP_SIZE=5`
  - `ADJUSTMENT_STEP_SIZE_DECREMENT=2`
  - `ADJUSTMENT_STEP_PERCENT=10.0`
  - `ADJUSTMENT_STEP_PERCENT_DECREMENT=5.0`
  - `UTILIZATION_DEADBAND=3.0`
  - `WARMUP_PERIOD_S=5.0`
  - `MAX_ADJUSTMENT_PER_MINUTE=100`

- **env copy.txt (main)**: No dynamic resource management
- **Resolution**: KEEP ALL dynamic resource settings from current branch

### 9. **Face Alignment & Model Ensemble** - CURRENT BRANCH ONLY
- **env.txt (current)**: 
  - `ENABLE_FACE_ALIGNMENT=true`
  - `ENABLE_MODEL_ENSEMBLE=false`
  - `ENSEMBLE_MODELS=buffalo_l,buffalo_m,buffalo_s`
  - `ENSEMBLE_FUSION_STRATEGY=weighted_avg`

- **env copy.txt (main)**: Not present
- **Resolution**: KEEP from current branch

### 10. **Vector Database Settings** - CURRENT BRANCH ONLY
- **env.txt (current)**: 
  - `DEFAULT_SEARCH_TOPK=10`
  - `MAX_SEARCH_TOPK=100`
  - `QDRANT_URL=http://qdrant:6333`

- **env copy.txt (main)**: Has `QDRANT_URL` but not the TOPK settings
- **Resolution**: KEEP all from current branch

### 11. **Additional Settings in Main Branch**
- **env copy.txt (main)**: Has settings not in current:
  - `TZ=America/Los_Angeles`
  - `S3_BUCKET_AUDIT=audit-logs`
  - `REDIS_URL=redis://redis:6379/0`
  - `API_HOST=0.0.0.0`
  - `API_PORT=8000`
  - `API_WORKERS=4`
  - `API_RELOAD=false`
  - `CORS_ORIGINS=*`
  - `LOG_FORMAT=json`
  - `ENABLE_METRICS=true`
  - `METRICS_PORT=9090`
  - `MAX_FACES_PER_IMAGE=10`
  - `BATCH_SIZE=32`
  - `MAX_CONCURRENT_TASKS=5`
  - `ENABLE_DEDUPLICATION=true`

- **Resolution**: ADD these from main branch

### 12. **File Structure**
- **env.txt (current)**: Minimal comments, flat structure
- **env copy.txt (main)**: Well-organized with section headers and comments
- **Resolution**: Use main branch's organized structure with section headers

## Recommended Merge Strategy

1. **Use main branch structure** (organized sections with comments)
2. **Keep all GPU-related settings** from current branch
3. **Keep all dynamic resource management** from current branch
4. **Use new face detection naming** from main branch
5. **Merge all unique settings** from both branches
6. **Resolve conflicts** using the resolutions above


