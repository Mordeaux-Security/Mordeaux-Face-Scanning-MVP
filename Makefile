SHELL := /bin/bash
PROFILE ?=
copy-env:
	@test -f .env || cp .env.example .env
up: copy-env
	docker compose up -d --build
	docker compose ps
logs:
	docker compose logs -f --tail=200
down:
	docker compose down
clean:
	docker compose down -v
bash-backend:
	docker compose exec backend-cpu bash || true
seed:
	docker compose exec backend-cpu python scripts/seed_demo.py || true
crawl:
	@echo "Usage: make crawl URL=<url> [METHOD=<method>] [MIN_FACE_QUALITY=<score>] [REQUIRE_FACE=<true/false>] [CROP_FACES=<true/false>] [FACE_MARGIN=<margin>] [CRAWL_MODE=<single/site>] [MAX_TOTAL_IMAGES=<number>] [MAX_PAGES=<number>] [MAX_CONCURRENT_IMAGES=<number>] [BATCH_SIZE=<number>] [TENANT_ID=<tenant_id>]"
	@echo "Example: make crawl URL=https://example.com METHOD=smart MIN_FACE_QUALITY=0.7 CROP_FACES=true FACE_MARGIN=0.2 CRAWL_MODE=site MAX_TOTAL_IMAGES=50 MAX_PAGES=20 MAX_CONCURRENT_IMAGES=10 BATCH_SIZE=25 TENANT_ID=tenant_123"
	@if [ -z "$(URL)" ]; then echo "Error: URL is required"; exit 1; fi
	@METHOD=$${METHOD:-smart}; \
	MIN_FACE_QUALITY=$${MIN_FACE_QUALITY:-0.5}; \
	FACE_MARGIN=$${FACE_MARGIN:-0.2}; \
	MAX_TOTAL_IMAGES=$${MAX_TOTAL_IMAGES:-50}; \
	MAX_PAGES=$${MAX_PAGES:-20}; \
	CRAWL_MODE=$${CRAWL_MODE:-single}; \
	MAX_CONCURRENT_IMAGES=$${MAX_CONCURRENT_IMAGES:-10}; \
	BATCH_SIZE=$${BATCH_SIZE:-25}; \
	TENANT_ID=$${TENANT_ID:-default}; \
	REQUIRE_FACE_FLAG=""; \
	if [ "$${REQUIRE_FACE:-true}" = "false" ]; then REQUIRE_FACE_FLAG="--no-require-face"; fi; \
	CROP_FACES_FLAG=""; \
	if [ "$${CROP_FACES:-true}" = "false" ]; then CROP_FACES_FLAG="--no-crop-faces"; fi; \
	docker compose exec backend-cpu python scripts/crawl_images.py $(URL) --method $$METHOD --min-face-quality $$MIN_FACE_QUALITY --face-margin $$FACE_MARGIN --max-images $$MAX_TOTAL_IMAGES --max-pages $$MAX_PAGES --mode $$CRAWL_MODE --max-concurrent-images $$MAX_CONCURRENT_IMAGES --batch-size $$BATCH_SIZE --tenant-id $$TENANT_ID $$REQUIRE_FACE_FLAG $$CROP_FACES_FLAG

crawl2:
	@echo "V2 Crawler - Simplified image crawling with face detection and upscaling"
	@echo "Usage: make crawl2 URL=<url> [MAX_FILE_SIZE=<size_mb>] [MAX_CONCURRENT=<number>] [MIN_FACE_SIZE=<pixels>] [FACE_MARGIN=<margin>] [MAX_PAGES=<number>] [MAX_TOTAL_IMAGES=<number>]"
	@echo "Example: make crawl2 URL=https://example.com MAX_FILE_SIZE=10 MAX_CONCURRENT=10 MIN_FACE_SIZE=50 FACE_MARGIN=0.2 MAX_PAGES=3 MAX_TOTAL_IMAGES=150"
	@echo ""
	@if [ -z "$(URL)" ]; then echo "Error: URL is required"; exit 1; fi
	@echo "Starting V2 crawler for: $(URL)"
	@MAX_FILE_SIZE=$${MAX_FILE_SIZE:-10}; \
	MAX_CONCURRENT=$${MAX_CONCURRENT:-10}; \
	MIN_FACE_SIZE=$${MIN_FACE_SIZE:-50}; \
	FACE_MARGIN=$${FACE_MARGIN:-0.2}; \
	MAX_PAGES=$${MAX_PAGES:-5}; \
	MAX_TOTAL_IMAGES=$${MAX_TOTAL_IMAGES:-500}; \
	docker compose exec backend-cpu python -c "\
import asyncio; \
import sys; \
import logging; \
sys.path.insert(0, '/app'); \
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'); \
from app.services.crawler_v2 import crawl_images_v2; \
result = asyncio.run(crawl_images_v2('$(URL)', max_pages_to_visit=$(MAX_PAGES), max_total_images=$(MAX_TOTAL_IMAGES))); \
print('\\n' + '='*60); \
print('CRAWL RESULTS:'); \
print('='*60); \
print(f'URL: {result.url}'); \
print(f'Images found: {result.images_found}'); \
print(f'Raw images saved: {result.raw_images_saved}'); \
print(f'Face crops saved: {result.face_crops_saved}'); \
print(f'Upscaling factors: {result.upscaling_factors}'); \
print(f'Errors: {len(result.errors)}'); \
if result.errors: \
    print('\\nErrors encountered:'); \
    for error in result.errors: \
        print(f'  - {error}'); \
print('='*60);"

restart:
	docker compose restart

reset-cache:
	@echo "Clearing crawl cache database..."
	docker compose exec backend-cpu python -c "from app.core.config import get_settings; s = get_settings(); print(f'Clearing cache for db: {s.postgres_db}')"
	docker compose exec postgres psql -U mordeaux -d mordeaux -c "DELETE FROM crawl_cache;"

reset-minio:
	@echo "Clearing MinIO buckets..."
	docker compose exec backend-cpu python -c "from app.core.config import get_settings; s = get_settings(); print(f'Clearing buckets: {s.s3_bucket_raw}, {s.s3_bucket_thumbs}')"
	docker compose exec backend-cpu python scripts/clear_minio.py

reset-both: reset-cache reset-minio
	@echo "Cache and MinIO data cleared."

reset-all: reset-both clean
	@echo "All data cleared and containers stopped."
