# Diabetes Crawler

A standalone, production-grade crawling pipeline that collects public face imagery and metadata, feeds it through GPU/CPU detection, and stores the results in MinIO/S3 plus an optional vector database. The system keeps the original multi-process architecture (orchestrator, crawlers, extractors, GPU processors, storage workers) but is now isolated inside this folder with zero external dependencies on the legacy backend/monolith.

## Folder Layout

```
diabetes-crawler/
├── README.md                 # This guide
├── requirements.txt          # Runtime dependencies
├── docker-compose.yml        # Redis + MinIO services
├── .env.example              # Copy to .env and customize
├── sites.txt                 # Example list of target sites
├── start-services.sh         # Start Redis/MinIO
├── stop-services.sh          # Stop services
├── activate.sh               # Activate venv + set PYTHONPATH
├── docs/POST_CRAWLER_ARCHITECTURE.md
└── src/diabetes_crawler/     # Python package
```

## Quick Start

1. **Start required services (Redis + MinIO)**
   ```bash
   cd diabetes-crawler
   ./start-services.sh
   ```
   This starts Redis on port 6379 and MinIO on port 9000 (console on 9001).
   Default MinIO credentials: `MINIOADMIN` / `MINIOADMIN`

2. **Set up Python**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configure environment**
   ```bash
   cp .env.example .env
   # edit .env with your Redis, MinIO/S3, GPU worker, and Qdrant settings
   # For local setup with docker-compose, defaults should work:
   # redis_url=redis://localhost:6379/0
   # s3_endpoint=http://localhost:9000
   # s3_access_key=MINIOADMIN
   # s3_secret_key=MINIOADMIN
   ```

4. **Export PYTHONPATH and activate environment**
   ```bash
   source activate.sh
   # This activates venv and sets PYTHONPATH=src
   ```

5. **Run a health check**
   ```bash
   python -m diabetes_crawler.main --health-check
   ```

6. **Start crawling**
   ```bash
   # Crawl URLs listed in sites.txt
   python -m diabetes_crawler.main --sites-file sites.txt

   # Crawl explicit URLs (space separated)
   python -m diabetes_crawler.main --sites https://example.com/gallery https://another.example.com/models

   # Override worker counts or limits on the fly
   python -m diabetes_crawler.main --sites-file sites.txt --num-crawlers 2 --max-pages-per-site 3
   ```

7. **Run the test suite**
   ```bash
   python -m diabetes_crawler.test_suite
   ```

## Environment Variables

All configuration is managed by `diabetes_crawler.config.CrawlerConfig` (Pydantic). The most useful knobs:

| Variable | Purpose |
| --- | --- |
| `redis_url` | Redis instance for all queues/caches |
| `num_crawlers`, `num_extractors`, `num_gpu_processors`, `num_storage_workers` | Process counts per worker type |
| `nc_max_pages_per_site`, `nc_max_images_per_site` | Hard limits per site to prevent runaway crawls |
| `gpu_worker_enabled`, `gpu_worker_url` | Toggle and address for the GPU microservice (CPU fallback runs automatically) |
| `s3_endpoint`, `s3_bucket_raw`, `s3_bucket_thumbs` | MinIO/S3 destination for raw images + thumbnails |
| `vectorization_enabled`, `qdrant_url`, `vector_index` | Enable vector upserts into Qdrant |
| `nc_debug_logging`, `nc_diagnostic_logging` | Verbose instrumentation for bottleneck hunting |
| `SKIP_REDIS_VALIDATION` | Set to `1` during local/offline development to skip Redis pings |

Copy `.env.example`, adjust the above, and the config loader will pick it up automatically.

## Logging & Debug Artifacts

- Structured logs are written to `diabetes_crawler.log` and stdout.
- Per-site traces, selector mining dumps, and candidate samples are stored under `crawl_output/debug/` (the folder is created automatically).
- Extraction traces live in the same debug folder with `extraction_trace_*.json` filenames.

## Useful Snippets

```python
from diabetes_crawler.redis_manager import get_redis_manager
from diabetes_crawler.orchestrator import Orchestrator
from diabetes_crawler.gpu_interface import get_gpu_interface

redis = get_redis_manager()
print(redis.get_all_queue_metrics())

orchestrator = Orchestrator()
print(orchestrator.check_worker_health())

gpu = get_gpu_interface()
healthy = asyncio.run(gpu._check_health())
print("GPU worker healthy?", healthy)
```

## Architecture Notes

The original architecture deep dive is preserved in `docs/POST_CRAWLER_ARCHITECTURE.md`. It covers the 5-1-1-1 worker split, batching strategy, Redis queue topology, and GPU scheduling logic.

## Troubleshooting

1. **GPU worker unavailable** – set `gpu_worker_enabled=false` or ensure the Windows service/container is reachable at `GPU_WORKER_URL`.
2. **Redis connection errors** – confirm the Redis instance is running and accessible from where you launch the orchestrator.
3. **Storage failures** – double-check MinIO/S3 credentials and bucket names; Stack traces land in `diabetes_crawler.log`.
4. **Queue overflow / back-pressure** – lower `num_crawlers` or increase downstream worker counts; monitor `nc_max_queue_depth`.

This folder now contains everything required to run, test, and extend the crawler in isolation—no more references to the legacy `/backend` tree.
