# Complete File Trace for `docker compose up new-crawler`

This document lists every file that is used when running `docker compose up new-crawler`, tracing all method calls and dependencies.

## Execution Flow

1. **Docker Compose** → `docker-compose.yml` (service definition)
2. **Dockerfile** → `backend/Dockerfile` (builds container)
3. **Entry Point** → `backend/new_crawler/main.py` (CLI entry point)
4. **Orchestrator** → Starts worker processes
5. **Workers** → Process sites, extract images, detect faces, store results
6. **GPU Worker** → External HTTP service (runs separately, called via HTTP)
   - **Note:** GPU worker has a very close relationship with the crawler
   - All face detection requests go through `gpu_interface.py` → `worker.py`
   - Uses same data structures for seamless integration

---

## Core Configuration Files

### Docker & Infrastructure
- `docker-compose.yml` - Service definition for `new-crawler`
- `backend/Dockerfile` - Container build instructions
- `backend/requirements.txt` - Python dependencies
- `backend/sites.txt` - Input sites file (mounted as volume)

### Environment Configuration
- `.env` - Environment variables (loaded via docker-compose)
- `backend/new_crawler/config.py` - Configuration management (Pydantic settings)

---

## Main Entry Point

### `backend/new_crawler/main.py`
**Imports:**
- `argparse` (stdlib)
- `asyncio` (stdlib)
- `logging` (stdlib)
- `sys` (stdlib)
- `pathlib.Path` (stdlib)
- `typing.List` (stdlib)
- `backend/new_crawler/config.py` → `get_config`, `validate_configuration`
- `backend/new_crawler/orchestrator.py` → `Orchestrator`
- `backend/new_crawler/redis_manager.py` → `get_redis_manager`
- `backend/new_crawler/cache_manager.py` → `get_cache_manager`
- `backend/new_crawler/storage_manager.py` → `get_storage_manager`
- `backend/new_crawler/gpu_interface.py` → `get_gpu_interface`

**Functions:**
- `setup_logging()` - Configures logging
- `load_sites_from_file()` - Reads sites.txt
- `load_sites_from_args()` - Parses CLI site arguments
- `health_check()` - System health check
- `run_crawl()` - Main crawl orchestration
- `print_results()` - Results display
- `main()` - CLI entry point

---

## Configuration Module

### `backend/new_crawler/config.py`
**Imports:**
- `os` (stdlib)
- `logging` (stdlib)
- `redis` (external: `redis==5.0.7`)
- `typing.Optional, Dict, Any` (stdlib)
- `pydantic.BaseModel, Field, field_validator, model_validator` (external: `pydantic-settings==2.5.2`)
- `pydantic_settings.BaseSettings` (external: `pydantic-settings==2.5.2`)
- `urllib.parse.urlparse` (stdlib)

**Classes:**
- `CrawlerConfig` - Main configuration class with 100+ settings

**Functions:**
- `get_config()` - Singleton config instance
- `reload_config()` - Reload configuration
- `validate_configuration()` - Validate config

---

## Orchestrator Module

### `backend/new_crawler/orchestrator.py`
**Imports:**
- `asyncio` (stdlib)
- `logging` (stdlib)
- `multiprocessing` (stdlib)
- `signal` (stdlib)
- `sys` (stdlib)
- `time` (stdlib)
- `typing.List, Dict, Any, Optional` (stdlib)
- `multiprocessing.Process, Queue` (stdlib)
- `json` (stdlib)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/redis_manager.py` → `get_redis_manager`
- `backend/new_crawler/data_structures.py` → `SiteTask, ProcessingStats, SystemMetrics, CrawlResults, FaceResult`
- `backend/new_crawler/timing_logger.py` → `get_timing_logger`

**Classes:**
- `Orchestrator` - Main coordinator

**Worker Process Entry Points:**
- `_run_crawler_worker()` → `backend/new_crawler/crawler_worker.py` → `crawler_worker_process()`
- `_run_extractor_worker()` → `backend/new_crawler/extractor_worker.py` → `extractor_worker_process()`
- `_run_gpu_processor_worker()` → `backend/new_crawler/gpu_processor_worker.py` → `gpu_processor_worker_process()`
- `_run_storage_worker()` → `backend/new_crawler/storage_worker.py` → `storage_worker_process()`

---

## Data Structures

### `backend/new_crawler/data_structures.py`
**Imports:**
- `base64` (stdlib)
- `json` (stdlib)
- `typing.List, Optional, Dict, Any, Union` (stdlib)
- `pydantic.BaseModel, Field, validator, model_serializer, field_validator` (external: `pydantic-settings==2.5.2`)
- `datetime.datetime` (stdlib)
- `enum.Enum` (stdlib)

**Classes:**
- `TaskStatus` (Enum)
- `SiteTask`
- `CandidateImage`
- `ImageTask`
- `FaceDetection`
- `FaceResult`
- `StorageTask`
- `ProcessingStats`
- `BatchRequest`
- `BatchResponse`
- `QueueMetrics`
- `SystemMetrics`
- `CrawlResults`

---

## Redis Manager

### `backend/new_crawler/redis_manager.py`
**Imports:**
- `json` (stdlib)
- `logging` (stdlib)
- `time` (stdlib)
- `datetime.datetime` (stdlib)
- `typing.Optional, List, Dict, Any, Union` (stdlib)
- `contextlib.asynccontextmanager` (stdlib)
- `redis` (external: `redis==5.0.7`)
- `redis.asyncio` (external: `redis==5.0.7`)
- `redis.connection.ConnectionPool` (external: `redis==5.0.7`)
- `redis.exceptions.RedisError, ConnectionError, TimeoutError` (external: `redis==5.0.7`)
- `pydantic.BaseModel` (external: `pydantic-settings==2.5.2`)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/data_structures.py` → `SiteTask, CandidateImage, ImageTask, FaceResult, BatchRequest, QueueMetrics, TaskStatus, StorageTask`

**Classes:**
- `RedisManager` - Redis operations manager

**Key Methods:**
- Queue operations (push/pop for sites, candidates, images, results, storage)
- Cache operations (set/get/delete/exists)
- URL deduplication
- Site statistics tracking
- Active task counting
- Domain rendering strategy management

---

## Cache Manager

### `backend/new_crawler/cache_manager.py`
**Imports:**
- `hashlib` (stdlib)
- `logging` (stdlib)
- `time` (stdlib)
- `typing.Optional, Dict, Any, Tuple` (stdlib)
- `pathlib.Path` (stdlib)
- `imagehash` (external: `imagehash==4.3.1`)
- `PIL.Image` (external: `Pillow==10.4.0`)
- `io` (stdlib)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/redis_manager.py` → `get_redis_manager`
- `backend/new_crawler/data_structures.py` → `CandidateImage, ImageTask, FaceResult`

**Classes:**
- `CacheManager` - Image deduplication and caching

**Key Methods:**
- `compute_phash()` - Perceptual hash computation
- `compute_phash_from_bytes()` - Hash from bytes
- `is_image_cached()` - Cache check
- `cache_face_result()` - Cache face detection results

---

## Storage Manager

### `backend/new_crawler/storage_manager.py`
**Imports:**
- `asyncio` (stdlib)
- `io` (stdlib)
- `logging` (stdlib)
- `os` (stdlib)
- `time` (stdlib)
- `typing.Optional, Dict, Any, Tuple, List` (stdlib)
- `pathlib.Path` (stdlib)
- `hashlib` (stdlib)
- `json` (stdlib)
- `datetime.datetime` (stdlib)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/data_structures.py` → `ImageTask, FaceDetection, FaceResult, StorageTask`

**External Dependencies:**
- `minio` (external: `minio>=7.2.0,<8.0.0`)
- `boto3` (external: `boto3==1.34.162`)
- `urllib3.PoolManager` (via minio)

**Classes:**
- `StorageManager` - MinIO/S3 storage operations

**Key Methods:**
- `save_raw_image()` - Save raw image to MinIO
- `save_face_thumbnail()` - Save face thumbnail
- `save_metadata()` - Save metadata sidecar
- `save_storage_task_async()` - Process storage task

---

## GPU Interface

### `backend/new_crawler/gpu_interface.py`
**Imports:**
- `asyncio` (stdlib)
- `base64` (stdlib)
- `json` (stdlib)
- `logging` (stdlib)
- `os` (stdlib)
- `time` (stdlib)
- `typing.List, Optional, Dict, Any, Tuple` (stdlib)
- `httpx` (external: `httpx==0.27.2`)
- `numpy` (external: `numpy==1.26.4`)
- `cv2` (external: `opencv-python-headless==4.10.0.84`)
- `PIL.Image` (external: `Pillow==10.4.0`)
- `io` (stdlib)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/data_structures.py` → `ImageTask, FaceDetection, BatchRequest, BatchResponse`
- `backend/new_crawler/gpu_worker_logger.py` → `GPUWorkerLogger`

**External Dependencies:**
- `insightface` (external: `insightface==0.7.3`) - CPU fallback face detection

**Classes:**
- `GPUInterface` - GPU worker client with CPU fallback

**Key Methods:**
- `process_batch()` - Process image batch
- `_gpu_worker_request()` - HTTP request to GPU worker (calls external service at `gpu_worker_url`)
- `_cpu_fallback()` - CPU-based face detection fallback

**HTTP Endpoints Called:**
- `GET /health` - Health check endpoint
- `POST /detect_faces_batch_multipart` - Batch face detection with multipart/form-data

---

## GPU Worker Service (External HTTP Service)

**Note:** The GPU worker is a separate FastAPI service that runs independently (typically on Windows with DirectML support). The crawler communicates with it via HTTP. This service has a very close relationship with the crawler as it processes all face detection requests.

### `backend/gpu_worker/worker.py`
**Imports:**
- `asyncio` (stdlib)
- `json` (stdlib)
- `logging` (stdlib)
- `os` (stdlib)
- `sys` (stdlib)
- `threading` (stdlib)
- `time` (stdlib)
- `typing.List, Optional, Dict, Any` (stdlib)
- `concurrent.futures.ThreadPoolExecutor` (stdlib)
- `cv2` (external: `opencv-python>=4.8.0`)
- `numpy` (external: `numpy>=1.24.0`)
- `fastapi.FastAPI, HTTPException, UploadFile, File, Form` (external: `fastapi>=0.104.0`)
- `fastapi.middleware.cors.CORSMiddleware` (external: `fastapi>=0.104.0`)
- `pydantic.BaseModel, Field` (external: `pydantic-settings==2.5.2`)
- `insightface` (external: `insightface>=0.7.3`)
- `insightface.app.FaceAnalysis` (external: `insightface>=0.7.3`)
- `onnxruntime` (external: `onnxruntime>=1.20.0`)
- `backend/gpu_worker/detectors.scrfd_onnx` → `SCRFDOnnx` (relative import)

**External Dependencies:**
- `uvicorn` (external: `uvicorn[standard]>=0.24.0`) - ASGI server
- `python-multipart` (external: `python-multipart>=0.0.6`) - Multipart form parsing

**Global Variables:**
- `_face_app` - InsightFace FaceAnalysis instance (singleton)
- `_batched_detector` - SCRFD batched detector instance
- `_model_lock` - Thread lock for model access
- `_initialization_lock` - Lock for initialization
- `_batch_metrics` - Metrics tracking dictionary
- `_thread_pool` - ThreadPoolExecutor for CPU operations

**Pydantic Models:**
- `HealthResponse` - Health check response
- `FaceDetection` - Face detection result (matches crawler's FaceDetection)
- `BatchResponsePhash` - Batch processing response keyed by phash

**FastAPI App:**
- `app` - FastAPI application instance
- CORS middleware enabled for all origins

**Key Functions:**
- `_check_directml_availability()` - Check if DirectML execution provider is available
- `_check_actual_gpu_usage()` - Verify GPU is actually being used
- `_patch_insightface_for_directml()` - Patch InsightFace to use DirectML
- `_load_face_model()` - Load InsightFace model with DirectML support
- `_activate_backoff()` - Auto-backoff for GPU errors (reduces batch size)
- `_load_batched_detector()` - Load SCRFD batched detector
- `_decode_image_bytes()` - Decode binary image bytes to numpy array
- `_log_metrics_summary()` - Log periodic metrics summary
- `_detect_faces_batch()` - Detect faces in batch (uses batched detector or fallback)

**FastAPI Endpoints:**
- `GET /health` - Health check endpoint
  - Returns: `HealthResponse` with status, GPU availability, model loaded status, uptime
- `POST /detect_faces_batch_multipart` - Batch face detection
  - Accepts: `List[UploadFile]` (images), `image_hashes` (JSON string), `min_face_quality`, `require_face`, `crop_faces`, `face_margin`
  - Returns: `BatchResponsePhash` with results keyed by phash
  - Process: Decodes images → Calls `_detect_faces_batch()` → Returns results

**Startup/Shutdown Events:**
- `startup_event()` - Initialize GPU worker on startup (loads models)
- `shutdown_event()` - Cleanup on shutdown

**Entry Point:**
- `if __name__ == "__main__"` - Runs uvicorn server on port 8765

---

### `backend/gpu_worker/detectors/scrfd_onnx.py`
**Imports:**
- `os` (stdlib)
- `logging` (stdlib)
- `numpy` (external: `numpy>=1.24.0`)
- `cv2` (external: `opencv-python>=4.8.0`)
- `typing.List, Dict, Optional, Tuple` (stdlib)
- `onnxruntime` (external: `onnxruntime>=1.20.0`)
- `insightface` (external: `insightface>=0.7.3`)
- `insightface.app.FaceAnalysis` (external: `insightface>=0.7.3`)
- `backend/gpu_worker/detectors.common` → `letterbox, LetterboxInfo, nms, DetectionResult` (relative import)

**Classes:**
- `SCRFDOnnx` - SCRFD (Scalable CNN-based Real-time Face Detector) batched ONNX implementation

**Key Methods:**
- `__init__()` - Initialize detector (loads ONNX session from InsightFace or path)
- `_load_from_path()` - Load ONNX model from file path
- `_load_from_face_app()` - Load ONNX session from InsightFace FaceAnalysis
- `_verify_dynamic_batch()` - Verify model supports dynamic batch dimension
- `_precompute_anchors()` - Precompute anchor grids for SCRFD decode
- `preprocess()` - Preprocess images into batched tensor (letterbox, normalize, stack)
- `infer()` - Run batched forward pass (ONNX inference)
- `decode()` - Decode SCRFD outputs to bounding boxes, scores, keypoints
- `_decode_single_image()` - Decode outputs for single image
- `_decode_fallback()` - Fallback decode when output format unclear
- `process_batch()` - End-to-end batch processing (preprocess → infer → decode → map coordinates)

**Key Features:**
- Supports dynamic batch dimension
- Uses DirectML execution provider on Windows
- Detailed timing breakdown (preprocess, GPU inference, decode, coordinate mapping)
- Coordinate mapping from letterbox space back to original image coordinates

---

### `backend/gpu_worker/detectors/common.py`
**Imports:**
- `numpy` (external: `numpy>=1.24.0`)
- `cv2` (external: `opencv-python>=4.8.0`)
- `typing.Tuple, List, Dict, Optional` (stdlib)
- `logging` (stdlib)

**Classes:**
- `LetterboxInfo` - Information about letterbox transformation
  - `map_box_to_original()` - Map bounding box from letterbox to original coordinates
  - `map_kps_to_original()` - Map keypoints from letterbox to original coordinates
- `DetectionResult` - Container for detection results per image
  - Attributes: `boxes` (K, 4), `scores` (K,), `kps` (K, 5, 2) or None

**Functions:**
- `letterbox()` - Resize image to target_size while preserving aspect ratio (padding)
- `nms()` - Non-Maximum Suppression to remove overlapping detections
- `_compute_iou()` - Compute IoU between boxes

---

### `backend/gpu_worker/detectors/__init__.py`
**Exports:**
- `SCRFDOnnx` from `.scrfd_onnx`
- `letterbox, nms, DetectionResult` from `.common`

---

### `backend/gpu_worker/requirements.txt`
**Dependencies:**
- `fastapi>=0.104.0` - API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `onnxruntime>=1.20.0` - ONNX Runtime (includes DirectML provider on Windows)
- `insightface>=0.7.3` - Face detection and embedding
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `pillow>=10.0.0` - Image processing
- `httpx>=0.25.0` - HTTP client (for health checks)
- `psutil>=5.9.0` - Process management
- `python-multipart>=0.0.6` - Multipart form parsing

---

### GPU Worker Communication Flow

```
backend/new_crawler/gpu_processor_worker.py
  └─> backend/new_crawler/gpu_interface.py
       └─> HTTP POST /detect_faces_batch_multipart
            └─> backend/gpu_worker/worker.py
                 ├─> _decode_image_bytes() - Decode binary images
                 ├─> _detect_faces_batch() - Batch face detection
                 │    ├─> backend/gpu_worker/detectors/scrfd_onnx.py
                 │    │    ├─> SCRFDOnnx.process_batch()
                 │    │    │    ├─> preprocess() - Letterbox and normalize
                 │    │    │    ├─> infer() - ONNX forward pass (GPU)
                 │    │    │    ├─> decode() - Decode outputs
                 │    │    │    └─> map coordinates to original
                 │    │    └─> backend/gpu_worker/detectors/common.py
                 │    │         ├─> letterbox() - Image preprocessing
                 │    │         ├─> nms() - Non-maximum suppression
                 │    │         └─> DetectionResult - Result container
                 │    └─> Fallback: Single-image processing via InsightFace
                 └─> Returns BatchResponsePhash (results keyed by phash)
```

**Key Relationship:**
- The GPU worker is called via HTTP from `gpu_interface.py` in the crawler
- Results are keyed by phash for reliable linkage between crawler and GPU worker
- GPU worker uses the same `FaceDetection` data structure as the crawler
- GPU worker supports batched processing for efficiency (SCRFD detector)
- Falls back to single-image processing if batched detector fails
- Auto-backoff mechanism reduces batch size on GPU errors

---

## GPU Worker Logger

### `backend/new_crawler/gpu_worker_logger.py`
**Imports:**
- `logging` (stdlib)
- `typing.Optional` (stdlib)

**Classes:**
- `GPUWorkerLogger` - Dedicated GPU operation logger

---

## GPU Scheduler

### `backend/new_crawler/gpu_scheduler.py`
**Imports:**
- `time` (stdlib)
- `logging` (stdlib)
- `collections.deque` (stdlib)
- `typing.List, Optional` (stdlib)
- `backend/new_crawler/data_structures.py` → `ImageTask`

**Classes:**
- `GPUScheduler` - Centralized GPU batch scheduling

**Key Methods:**
- `feed()` - Pull items from Redis queue
- `next_batch()` - Get next batch to process
- `mark_launched()` - Mark batch as launched
- `mark_completed()` - Mark batch as completed

---

## Crawler Worker

### `backend/new_crawler/crawler_worker.py`
**Imports:**
- `asyncio` (stdlib)
- `logging` (stdlib)
- `multiprocessing` (stdlib)
- `os` (stdlib)
- `signal` (stdlib)
- `sys` (stdlib)
- `time` (stdlib)
- `typing.List, Optional` (stdlib)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/redis_manager.py` → `get_redis_manager`
- `backend/new_crawler/selector_miner.py` → `get_selector_miner`
- `backend/new_crawler/http_utils.py` → `get_http_utils`
- `backend/new_crawler/data_structures.py` → `SiteTask, CandidateImage, TaskStatus`
- `backend/new_crawler/timing_logger.py` → `get_timing_logger`

**Classes:**
- `CrawlerWorker` - HTML fetching and selector mining worker

**Key Methods:**
- `process_site()` - Process a single site
- `run()` - Main worker loop

**Process Entry Point:**
- `crawler_worker_process()` - Multiprocessing entry point

---

## Selector Miner

### `backend/new_crawler/selector_miner.py`
**Imports:**
- `asyncio` (stdlib)
- `logging` (stdlib)
- `time` (stdlib)
- `re` (stdlib)
- `typing.List, Dict, Any, Optional, Set, AsyncIterator, Tuple` (stdlib)
- `urllib.parse.urljoin, urlparse` (stdlib)
- `bs4.BeautifulSoup` (external: `beautifulsoup4==4.12.2`)
- `httpx` (external: `httpx==0.27.2`)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/http_utils.py` → `get_http_utils`
- `backend/new_crawler/data_structures.py` → `CandidateImage`

**Classes:**
- `SelectorMiner` - HTML selector mining with 3x3 crawl

**Key Methods:**
- `mine_selectors()` - Extract image candidates from HTML
- `mine_with_3x3_crawl()` - 3x3 crawling strategy
- `_extract_image_url()` - Extract image URLs from HTML elements

---

## HTTP Utils

### `backend/new_crawler/http_utils.py`
**Imports:**
- `asyncio` (stdlib)
- `logging` (stdlib)
- `random` (stdlib)
- `time` (stdlib)
- `os` (stdlib)
- `typing.Optional, Tuple, Dict, Any, List` (stdlib)
- `urllib.parse.urljoin, urlparse` (stdlib)
- `contextlib.asynccontextmanager` (stdlib)
- `collections.deque` (stdlib)
- `httpx` (external: `httpx==0.27.2`)
- `bs4.BeautifulSoup` (external: `beautifulsoup4==4.12.2`)
- `backend/new_crawler/config.py` → `get_config`

**Optional Imports:**
- `psutil` (external: `psutil==6.1.0`) - Memory monitoring
- `resource` (stdlib) - Memory monitoring fallback
- `playwright.async_api` (external: `playwright==1.48.0`) - JavaScript rendering

**Classes:**
- `DomainConnectionPool` - Per-domain HTTP connection pooling
- `HTTPUtils` - HTTP client with retry and JS rendering
- `BrowserPool` - Playwright browser instance pool

**Key Methods:**
- `fetch_html()` - Fetch HTML with JS fallback
- `download_to_temp()` - Download image to temp file
- `head_check()` - HEAD request validation
- `_fetch_with_js()` - JavaScript rendering via Playwright

---

## Extractor Worker

### `backend/new_crawler/extractor_worker.py`
**Imports:**
- `asyncio` (stdlib)
- `logging` (stdlib)
- `multiprocessing` (stdlib)
- `os` (stdlib)
- `signal` (stdlib)
- `sys` (stdlib)
- `tempfile` (stdlib)
- `time` (stdlib)
- `datetime.datetime` (stdlib)
- `typing.List, Optional, Tuple` (stdlib)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/redis_manager.py` → `get_redis_manager`
- `backend/new_crawler/cache_manager.py` → `get_cache_manager`
- `backend/new_crawler/http_utils.py` → `get_http_utils`
- `backend/new_crawler/data_structures.py` → `CandidateImage, ImageTask, BatchRequest, TaskStatus`
- `backend/new_crawler/timing_logger.py` → `get_timing_logger`

**Classes:**
- `ExtractorWorker` - Image download and batch preparation worker

**Key Methods:**
- `process_candidate()` - Download and process candidate image
- `run()` - Main worker loop

**Process Entry Point:**
- `extractor_worker_process()` - Multiprocessing entry point

---

## GPU Processor Worker

### `backend/new_crawler/gpu_processor_worker.py`
**Imports:**
- `asyncio` (stdlib)
- `io` (stdlib)
- `logging` (stdlib)
- `multiprocessing` (stdlib)
- `os` (stdlib)
- `signal` (stdlib)
- `sys` (stdlib)
- `tempfile` (stdlib)
- `time` (stdlib)
- `shutil` (stdlib)
- `datetime.datetime` (stdlib)
- `typing.List, Optional, Dict` (stdlib)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/redis_manager.py` → `get_redis_manager`
- `backend/new_crawler/cache_manager.py` → `get_cache_manager`
- `backend/new_crawler/gpu_interface.py` → `get_gpu_interface`
- `backend/new_crawler/gpu_scheduler.py` → `GPUScheduler`
- `backend/new_crawler/data_structures.py` → `BatchRequest, FaceResult, FaceDetection, TaskStatus, ImageTask, StorageTask`
- `backend/new_crawler/timing_logger.py` → `get_timing_logger`

**External Dependencies:**
- `PIL.Image` (external: `Pillow==10.4.0`) - Face cropping

**Classes:**
- `GPUProcessorWorker` - GPU batch processing worker

**Key Methods:**
- `process_batch()` - Process image batch
- `_crop_faces()` - Crop faces from images
- `_prepare_storage_task()` - Create storage task
- `run()` - Main worker loop with GPU scheduler

**Process Entry Point:**
- `gpu_processor_worker_process()` - Multiprocessing entry point

---

## Storage Worker

### `backend/new_crawler/storage_worker.py`
**Imports:**
- `asyncio` (stdlib)
- `logging` (stdlib)
- `multiprocessing` (stdlib)
- `os` (stdlib)
- `signal` (stdlib)
- `sys` (stdlib)
- `time` (stdlib)
- `typing.Optional` (stdlib)
- `backend/new_crawler/config.py` → `get_config`
- `backend/new_crawler/redis_manager.py` → `get_redis_manager`
- `backend/new_crawler/storage_manager.py` → `get_storage_manager`
- `backend/new_crawler/cache_manager.py` → `get_cache_manager`
- `backend/new_crawler/timing_logger.py` → `get_timing_logger`
- `backend/new_crawler/data_structures.py` → `StorageTask, FaceResult`

**Classes:**
- `StorageWorker` - Storage queue consumer worker

**Key Methods:**
- `_process_storage_task()` - Process storage task
- `run()` - Main worker loop

**Process Entry Point:**
- `storage_worker_process()` - Multiprocessing entry point

---

## Timing Logger

### `backend/new_crawler/timing_logger.py`
**Imports:**
- `threading` (stdlib)
- `time` (stdlib)
- `datetime.datetime` (stdlib)
- `typing.Optional` (stdlib)
- `os` (stdlib)

**Classes:**
- `TimingLogger` - Thread-safe timing logger (singleton)

**Key Methods:**
- `log_system_start()` - Log system start
- `log_crawl_start()` - Log crawl start
- `log_site_start()` - Log site start
- `log_page_start()` - Log page start
- `log_page_end()` - Log page end
- `log_site_end()` - Log site end
- `log_extraction_start()` - Log extraction start
- `log_extraction_end()` - Log extraction end
- `log_gpu_batch_start()` - Log GPU batch start
- `log_gpu_recognition_start()` - Log GPU recognition start
- `log_gpu_recognition_end()` - Log GPU recognition end
- `log_gpu_batch_end()` - Log GPU batch end
- `log_storage_start()` - Log storage start
- `log_storage_end()` - Log storage end
- `log_crawl_end()` - Log crawl end
- `log_system_shutdown()` - Log system shutdown

**Functions:**
- `get_timing_logger()` - Get singleton instance

---

## External Python Dependencies

### Crawler Dependencies (`backend/requirements.txt`):

### Core Framework
- `fastapi==0.115.0`
- `gunicorn==22.0.0`
- `uvicorn[standard]==0.30.6`
- `pydantic-settings==2.5.2`

### Database & Cache
- `psycopg[binary]==3.2.1`
- `psycopg_pool==3.2.3`
- `redis==5.0.7`
- `celery==5.4.0`

### HTTP & Web
- `httpx==0.27.2`
- `python-multipart==0.0.9`
- `beautifulsoup4==4.12.2`
- `aiofiles==23.2.1`
- `aiohttp==3.10.11`
- `playwright==1.48.0` (JavaScript rendering)

### Image Processing
- `Pillow==10.4.0`
- `imagehash==4.3.1`
- `opencv-python-headless==4.10.0.84`
- `numpy==1.26.4`

### Face Detection
- `insightface==0.7.3`
- `onnxruntime==1.19.2`

### Storage
- `boto3==1.34.162`
- `minio>=7.2.0,<8.0.0`

### Vector Databases
- `qdrant-client==1.10.1`
- `pinecone-client==3.2.2`

### Utilities
- `python-dotenv==1.1.1`
- `psutil==6.1.0`
- `blake3==1.0.7`

### Testing
- `pytest==7.4.3`
- `pytest-asyncio==0.21.1`

### GPU Worker Dependencies (`backend/gpu_worker/requirements.txt`):
- `fastapi>=0.104.0` - API framework
- `uvicorn[standard]>=0.24.0` - ASGI server
- `onnxruntime>=1.20.0` - ONNX Runtime (includes DirectML provider on Windows)
- `insightface>=0.7.3` - Face detection and embedding
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.24.0` - Numerical operations
- `pillow>=10.0.0` - Image processing
- `httpx>=0.25.0` - HTTP client (for health checks)
- `psutil>=5.9.0` - Process management
- `python-multipart>=0.0.6` - Multipart form parsing

**Note:** GPU worker runs as a separate service with its own dependencies. The crawler only needs `httpx` to communicate with it.

---

## Standard Library Modules Used

### Core
- `argparse` - CLI argument parsing
- `asyncio` - Async I/O
- `base64` - Base64 encoding
- `collections.deque` - Queue data structures
- `contextlib.asynccontextmanager` - Async context managers
- `datetime` - Date/time handling
- `enum` - Enumerations
- `hashlib` - Hashing algorithms
- `io` - I/O utilities
- `json` - JSON serialization
- `logging` - Logging framework
- `multiprocessing` - Process management
- `os` - OS interface
- `pathlib` - Path handling
- `random` - Random number generation
- `re` - Regular expressions
- `resource` - Resource usage (optional)
- `shutil` - File operations
- `signal` - Signal handling
- `sys` - System-specific parameters
- `tempfile` - Temporary file creation
- `threading` - Thread management
- `time` - Time-related functions
- `typing` - Type hints
- `urllib.parse` - URL parsing

---

## Summary Statistics

### Python Files in `backend/new_crawler/`
1. `__init__.py` (empty)
2. `main.py` - Entry point
3. `config.py` - Configuration
4. `orchestrator.py` - Main coordinator
5. `data_structures.py` - Pydantic models
6. `redis_manager.py` - Redis operations
7. `cache_manager.py` - Image caching
8. `storage_manager.py` - MinIO/S3 storage
9. `gpu_interface.py` - GPU worker client
10. `gpu_worker_logger.py` - GPU logging
11. `gpu_scheduler.py` - GPU batch scheduling
12. `crawler_worker.py` - HTML crawling worker
13. `selector_miner.py` - Selector mining
14. `http_utils.py` - HTTP client utilities
15. `extractor_worker.py` - Image extraction worker
16. `gpu_processor_worker.py` - GPU processing worker
17. `storage_worker.py` - Storage worker
18. `timing_logger.py` - Performance timing

### Python Files in `backend/gpu_worker/` (External Service)
1. `worker.py` - FastAPI service entry point
2. `detectors/__init__.py` - Detector package exports
3. `detectors/scrfd_onnx.py` - SCRFD batched detector
4. `detectors/common.py` - Common utilities (letterbox, NMS)

### Configuration Files
- `docker-compose.yml` - Docker Compose service definition
- `backend/Dockerfile` - Container build
- `backend/requirements.txt` - Python dependencies (crawler)
- `backend/gpu_worker/requirements.txt` - Python dependencies (GPU worker)
- `backend/sites.txt` - Input sites (mounted volume)
- `.env` - Environment variables

### Total Files: 22 Python files (crawler) + 4 Python files (GPU worker) + 6 configuration files = 32 files

---

## Execution Path Summary

```
docker-compose.yml
  └─> backend/Dockerfile
       └─> backend/requirements.txt (installs dependencies)
            └─> Command: python -m new_crawler.main --sites-file sites.txt
                 └─> backend/new_crawler/main.py
                      ├─> backend/new_crawler/config.py
                      ├─> backend/new_crawler/orchestrator.py
                      │    ├─> backend/new_crawler/redis_manager.py
                      │    ├─> backend/new_crawler/timing_logger.py
                      │    ├─> backend/new_crawler/data_structures.py
                      │    └─> Starts worker processes:
                      │         ├─> backend/new_crawler/crawler_worker.py
                      │         │    ├─> backend/new_crawler/selector_miner.py
                      │         │    └─> backend/new_crawler/http_utils.py
                      │         ├─> backend/new_crawler/extractor_worker.py
                      │         │    ├─> backend/new_crawler/cache_manager.py
                      │         │    └─> backend/new_crawler/http_utils.py
                      │         ├─> backend/new_crawler/gpu_processor_worker.py
                      │         │    ├─> backend/new_crawler/gpu_interface.py
                      │         │    │    └─> HTTP POST to GPU Worker Service
                      │         │    │         └─> backend/gpu_worker/worker.py (EXTERNAL SERVICE)
                      │         │    │              ├─> backend/gpu_worker/detectors/scrfd_onnx.py
                      │         │    │              │    └─> backend/gpu_worker/detectors/common.py
                      │         │    │              └─> Returns BatchResponsePhash
                      │         │    ├─> backend/new_crawler/gpu_scheduler.py
                      │         │    └─> backend/new_crawler/gpu_worker_logger.py
                      │         └─> backend/new_crawler/storage_worker.py
                      │              └─> backend/new_crawler/storage_manager.py
                      ├─> backend/new_crawler/cache_manager.py
                      ├─> backend/new_crawler/storage_manager.py
                      └─> backend/new_crawler/gpu_interface.py
```

---

## Notes

- All workers run as separate processes (multiprocessing)
- Redis is used for inter-process communication (queues)
- MinIO/S3 is used for persistent storage
- **GPU worker is an external HTTP service** (runs separately, typically on Windows with DirectML)
  - Communicates via HTTP from `gpu_interface.py` → `worker.py`
  - Uses same `FaceDetection` data structure for consistency
  - Results keyed by phash for reliable linkage
  - Supports batched processing via SCRFD detector
  - Falls back to single-image processing if batched fails
  - Auto-backoff reduces batch size on GPU errors
- CPU fallback available in crawler if GPU worker unavailable
- Playwright is used for JavaScript rendering (optional)
- All timing/logging goes to `timings04.txt` and stdout

## GPU Worker Service Details

**Deployment:**
- Runs as separate FastAPI service (not in Docker container for new-crawler)
- Typically runs on Windows with DirectML support for AMD GPU
- Listens on port 8765 (configurable via `gpu_worker_url`)
- Accessible via `host.docker.internal:8765` from Docker containers

**Key Integration Points:**
1. **Health Check**: `gpu_interface.py` calls `GET /health` to verify availability
2. **Batch Processing**: `gpu_interface.py` calls `POST /detect_faces_batch_multipart` with:
   - Binary image files (multipart/form-data, no base64 encoding)
   - Image hashes JSON mapping (phash → index)
   - Processing parameters (min_face_quality, face_margin, etc.)
3. **Response Format**: Returns `BatchResponsePhash` with results keyed by phash
4. **Error Handling**: Auto-backoff on GPU errors, falls back to CPU in crawler if service unavailable

**Data Structure Alignment:**
- GPU worker's `FaceDetection` model matches crawler's `FaceDetection` exactly
- Both use same bbox format: `[x1, y1, x2, y2]`
- Both use same quality scoring and landmark format
- Ensures seamless data flow between services

---

## Complete File List

### Configuration Files
- `docker-compose.yml`
- `backend/Dockerfile`
- `backend/requirements.txt`
- `backend/gpu_worker/requirements.txt`
- `backend/sites.txt`
- `.env`

### Crawler Python Files
- `backend/new_crawler/__init__.py`
- `backend/new_crawler/main.py`
- `backend/new_crawler/config.py`
- `backend/new_crawler/orchestrator.py`
- `backend/new_crawler/data_structures.py`
- `backend/new_crawler/redis_manager.py`
- `backend/new_crawler/cache_manager.py`
- `backend/new_crawler/storage_manager.py`
- `backend/new_crawler/gpu_interface.py`
- `backend/new_crawler/gpu_worker_logger.py`
- `backend/new_crawler/gpu_scheduler.py`
- `backend/new_crawler/crawler_worker.py`
- `backend/new_crawler/selector_miner.py`
- `backend/new_crawler/http_utils.py`
- `backend/new_crawler/extractor_worker.py`
- `backend/new_crawler/gpu_processor_worker.py`
- `backend/new_crawler/storage_worker.py`
- `backend/new_crawler/timing_logger.py`

### GPU Worker Python Files
- `backend/gpu_worker/worker.py`
- `backend/gpu_worker/detectors/__init__.py`
- `backend/gpu_worker/detectors/scrfd_onnx.py`
- `backend/gpu_worker/detectors/common.py`

