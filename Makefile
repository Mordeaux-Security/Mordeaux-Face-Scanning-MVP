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
	docker compose exec backend-cpu python app/crawler/crawl_images.py $(URL) --method $$METHOD --min-face-quality $$MIN_FACE_QUALITY --face-margin $$FACE_MARGIN --max-images $$MAX_TOTAL_IMAGES --max-pages $$MAX_PAGES --mode $$CRAWL_MODE --max-concurrent-images $$MAX_CONCURRENT_IMAGES --batch-size $$BATCH_SIZE --tenant-id $$TENANT_ID $$REQUIRE_FACE_FLAG $$CROP_FACES_FLAG

crawl-list:
	@echo "Usage: make crawl-list [SITES_FILE=<file>] [OUTPUT_DIR=<dir>] [MAX_CONCURRENT=<number>] [MAX_PAGES_PER_SITE=<number>] [MAX_IMAGES_PER_SITE=<number>] [REQUIRE_FACE=<true/false>] [NO_SELECTOR_MINING=<true/false>]"
	@echo "Example: make crawl-list SITES_FILE=sites.txt MAX_CONCURRENT=2 MAX_PAGES_PER_SITE=5 MAX_IMAGES_PER_SITE=20"
	@SITES_FILE=$${SITES_FILE:-sites.txt}; \
	OUTPUT_DIR=$${OUTPUT_DIR:-list_crawl_results}; \
	MAX_CONCURRENT=$${MAX_CONCURRENT:-2}; \
	MAX_PAGES_PER_SITE=$${MAX_PAGES_PER_SITE:-5}; \
	MAX_IMAGES_PER_SITE=$${MAX_IMAGES_PER_SITE:-20}; \
	REQUIRE_FACE_FLAG=""; \
	if [ "$${REQUIRE_FACE:-false}" = "true" ]; then REQUIRE_FACE_FLAG="--require-face"; fi; \
	NO_SELECTOR_MINING_FLAG=""; \
	if [ "$${NO_SELECTOR_MINING:-false}" = "true" ]; then NO_SELECTOR_MINING_FLAG="--no-selector-mining"; fi; \
	docker compose exec backend-cpu python app/crawler/crawl_list.py --sites-file $$SITES_FILE --output-dir $$OUTPUT_DIR --max-concurrent $$MAX_CONCURRENT --max-pages-per-site $$MAX_PAGES_PER_SITE --max-images-per-site $$MAX_IMAGES_PER_SITE $$REQUIRE_FACE_FLAG $$NO_SELECTOR_MINING_FLAG

restart:
	docker compose restart

reset-cache:
	@echo "Clearing crawl cache database..."
	docker compose exec backend-cpu python -c "from app.core.config import get_settings; s = get_settings(); print(f'Clearing cache for db: {s.postgres_db}')"
	docker compose exec postgres psql -U mordeaux -d mordeaux -c "DELETE FROM crawl_cache;"

reset-redis:
	@echo "Clearing Redis cache..."
	python backend/scripts/reset_redis_cache.py --all

reset-redis-docker:
	@echo "Clearing Redis cache via Docker..."
	docker compose exec redis redis-cli FLUSHDB
	@echo "Redis cache cleared via Docker"

reset-redis-test:
	@echo "Clearing Redis test cache (DB 15)..."
	python backend/scripts/reset_redis_cache.py --db 15 --all

reset-redis-info:
	@echo "Redis cache information:"
	python backend/scripts/reset_redis_cache.py --info

reset-redis-all-methods:
	@echo "Trying all Redis reset methods:"
	bash backend/scripts/redis_reset_methods.sh

reset-minio:
	@echo "Clearing MinIO buckets..."
	docker compose exec backend-cpu python -c "from app.core.config import get_settings; s = get_settings(); print(f'Clearing buckets: {s.s3_bucket_raw}, {s.s3_bucket_thumbs}')"
	docker compose exec backend-cpu python scripts/clear_minio.py

reset-both: reset-cache reset-redis-docker reset-minio
	@echo "Cache, Redis, and MinIO data cleared."

reset-all: reset-both clean
	@echo "All data cleared and containers stopped."

download-thumb:
	@./scripts/download_images.sh thumbnails

download-raw:
	@./scripts/download_images.sh raw-images

download-both:
	@./scripts/download_images.sh both

clean-downloads:
	@echo "Cleaning up download directories..."
	@if [ -d "flat" ]; then \
		echo "  - Removing flat/ directory..."; \
		rm -rf flat; \
		echo "    ✓ Removed flat/ directory"; \
	else \
		echo "  - flat/ directory not found"; \
	fi
	@if [ -d "zips" ]; then \
		echo "  - Removing zips/ directory..."; \
		rm -rf zips; \
		echo "    ✓ Removed zips/ directory"; \
	else \
		echo "  - zips/ directory not found"; \
	fi
	@echo "Download directories cleaned."
