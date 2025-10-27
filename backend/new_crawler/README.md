# New Crawler System

A clean, production-ready multiprocess crawler system that eliminates the complexity and fragility of the current implementation. Uses 5-1-1-1 worker allocation with proper batching, Redis caching, and GPU worker integration.

## Architecture

### Process Layout (8 cores total)

- **5 Crawler Workers**: HTML fetching, parsing, selector mining (async I/O)
- **1 Extractor Worker**: Image download, HEAD/GET validation, batch preparation
- **1 Orchestrator Worker**: Queue management, back-pressure control, metrics
- **1 GPU Worker** (existing): Native Windows service with CPU fallback

### Data Flow

```
Orchestrator → Redis(sites_queue) → Crawlers → Redis(candidates_queue) →
Extractor → Redis(image_batch_queue) → GPU Worker → MinIO + Redis Cache
```

## Key Features

1. **Clean separation**: Each worker has single responsibility
2. **No event loop conflicts**: Proper async/sync boundaries
3. **Consistent data**: Pydantic validation everywhere
4. **Better batching**: No single-image GPU requests
5. **Deduplication**: Redis cache with phash before processing
6. **Addressing**: Environment-based URL configuration
7. **Fallbacks**: JS rendering and CPU processing when needed
8. **Monitoring**: Clear dataflow logging at each stage
9. **Back-pressure**: Orchestrator prevents queue overflow

## Quick Start

### 1. Prerequisites

- Python 3.8+
- Redis server running
- MinIO/S3 storage configured
- Windows GPU worker service (optional)

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Update `env.txt` with your configuration:

```bash
# New Crawler Configuration
NC_BATCH_SIZE=64
NC_MAX_QUEUE_DEPTH=512
NUM_CRAWLERS=5
NUM_EXTRACTORS=1
NUM_GPU_PROCESSORS=1

# GPU Worker URL (override in docker-compose for containers)
GPU_WORKER_URL=http://localhost:8765

# Timeouts (generous for functionality)
NC_HTTP_TIMEOUT=30.0
NC_GPU_TIMEOUT=60.0
NC_JS_RENDER_TIMEOUT=15.0
```

### 4. Run Tests

```bash
# Run all tests
python -m backend.new_crawler.test_suite

# Health check only
python -m backend.new_crawler.main --health-check
```

### 5. Start Crawling

```bash
# Crawl sites from file
python -m backend.new_crawler.main --sites-file sites.txt

# Crawl specific sites
python -m backend.new_crawler.main --sites https://example1.com https://example2.com

# With custom configuration
python -m backend.new_crawler.main --sites-file sites.txt --num-crawlers 3 --batch-size 32
```

## Configuration

### Environment Variables

| Variable                   | Default | Description                     |
| -------------------------- | ------- | ------------------------------- |
| `NC_BATCH_SIZE`            | 64      | Batch size for GPU processing   |
| `NC_MAX_QUEUE_DEPTH`       | 512     | Maximum queue depth             |
| `NUM_CRAWLERS`             | 5       | Number of crawler workers       |
| `NUM_EXTRACTORS`           | 1       | Number of extractor workers     |
| `NUM_GPU_PROCESSORS`       | 1       | Number of GPU processor workers |
| `NC_USE_3X3_MINING`        | true    | Enable 3x3 selector mining      |
| `NC_MAX_SELECTOR_PATTERNS` | 30      | Maximum selector patterns       |
| `NC_HTTP_TIMEOUT`          | 30.0    | HTTP request timeout            |
| `NC_GPU_TIMEOUT`           | 60.0    | GPU worker timeout              |
| `MIN_FACE_QUALITY`         | 0.5     | Minimum face quality threshold  |
| `FACE_MARGIN`              | 0.2     | Face crop margin                |

### Docker/Windows Addressing

The system automatically handles addressing between Docker containers and Windows:

- **Docker containers**: Use `host.docker.internal:8765` for GPU worker
- **Windows native**: Use `localhost:8765` for GPU worker

Override in `docker-compose.yml`:

```yaml
environment:
  - GPU_WORKER_URL=http://host.docker.internal:8765
```

## Components

### Data Structures (`data_structures.py`)

Clean Pydantic models for all queue messages:

- `SiteTask`: URL to crawl + config
- `CandidateImage`: page_url, img_url, selector_hint, site
- `ImageTask`: temp_path, metadata (no bytes in queue)
- `FaceResult`: crop_path, bbox, embedding, metadata

### Configuration (`config.py`)

- Load from environment variables
- Support Docker container addressing
- Support Windows native addressing
- GPU worker URL configuration
- Timeout settings (generous defaults)

### Redis Manager (`redis_manager.py`)

- Queue operations: push/pop with timeout
- Queue naming: `nc:sites`, `nc:candidates`, `nc:images`, `nc:results`
- Back-pressure monitoring: queue depth checks
- Connection pooling

### Cache Manager (`cache_manager.py`)

- Phash computation for images
- Redis key format: `nc:cache:phash:{hash}`
- Check before processing (return cached result)
- Store after processing (phash → MinIO key)
- TTL management (90 days default)

### HTTP Utils (`http_utils.py`)

- Reuse existing `http_service.py` patterns
- Standard HTTP first, JS rendering as fallback
- Detection of blocking: cloudflare, captcha, access denied
- Generous timeouts: 30s read, 15s connect
- Retry with exponential backoff

### Selector Miner (`selector_miner.py`)

- Clean up existing `selector_mining.py`
- Remove bloated patterns, keep core 20-30 patterns
- 3x3 crawl: 3 category pages × 3 content pages
- Return validated candidates only
- JS rendering fallback if <3 candidates

### GPU Interface (`gpu_interface.py`)

- GPU worker client with CPU fallback
- Consistent data structures
- Proper batching (no single-image requests)
- Circuit breaker pattern
- Health checking

### Storage Manager (`storage_manager.py`)

- Save raw image to MinIO `raw-images` bucket
- Save face crops to MinIO `thumbnails` bucket
- Content-addressed naming (hash-based)
- Metadata sidecar JSON files
- Update cache with MinIO keys

## Monitoring

### Queue Metrics

Monitor queue health and back-pressure:

```python
from backend.new_crawler.redis_manager import get_redis_manager

redis = get_redis_manager()
metrics = redis.get_all_queue_metrics()
for metric in metrics:
    print(f"{metric.queue_name}: {metric.depth}/{metric.max_depth} ({metric.utilization_percent:.1f}%)")
```

### System Health

Check overall system health:

```python
from backend.new_crawler.orchestrator import Orchestrator

orchestrator = Orchestrator()
health = orchestrator.check_worker_health()
print(f"Overall healthy: {health['overall_healthy']}")
```

### GPU Worker Status

Check GPU worker availability:

```python
from backend.new_crawler.gpu_interface import get_gpu_interface

gpu_interface = get_gpu_interface()
available = await gpu_interface._check_health()
print(f"GPU worker available: {available}")
```

## Troubleshooting

### Common Issues

1. **GPU Worker Not Available**

   - Check if Windows GPU worker service is running
   - Verify URL configuration (`GPU_WORKER_URL`)
   - Check network connectivity

2. **Redis Connection Failed**

   - Ensure Redis server is running
   - Check `REDIS_URL` configuration
   - Verify network connectivity

3. **Storage Errors**

   - Check MinIO/S3 credentials
   - Verify bucket permissions
   - Check `S3_ENDPOINT` configuration

4. **Queue Overflow**
   - Reduce `NC_CRAWLER_CONCURRENCY`
   - Increase `NC_MAX_QUEUE_DEPTH`
   - Check downstream processing speed

### Debug Mode

Run with debug logging:

```bash
python -m backend.new_crawler.main --sites-file sites.txt --log-level debug
```

### Health Check

Perform system health check:

```bash
python -m backend.new_crawler.main --health-check
```

## Performance Tuning

### Worker Configuration

- **Crawlers**: 5 workers, 32 concurrent requests each
- **Extractors**: 1 worker, 16 concurrent downloads
- **GPU Processors**: 1 worker, 4-6 concurrent GPU requests

### Batch Sizes

- **Default**: 64 images per batch
- **High throughput**: 128 images per batch
- **Low latency**: 32 images per batch

### Timeouts

- **HTTP**: 30s (generous for functionality)
- **GPU**: 60s (allows for processing time)
- **JS Rendering**: 15s (fallback only)

## Migration from Old Crawler

1. **Test alongside**: Run new crawler alongside existing system
2. **Compare results**: Verify accuracy and performance
3. **Switch over**: Update main entry points when stable
4. **Archive old**: Move old crawler code to archive

## Contributing

1. Follow existing code patterns
2. Add tests for new functionality
3. Update documentation
4. Use type hints and Pydantic models
5. Handle errors gracefully

## License

Same as main project.




