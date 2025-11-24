SHELL := /bin/bash

up:
	 docker compose up -d --build

down:
	 docker compose down -v

logs:
	 docker compose logs -f --tail=200

restart:
	 docker compose restart

ensure:
	 docker compose exec face-pipeline python -c "from pipeline.ensure import ensure_all; ensure_all(); print('ensure complete')"

health:
	 curl -s http://localhost/api/v1/health || true; echo
	 curl -s http://localhost/api/healthz || true; echo
	 curl -s http://localhost/pipeline/api/v1/health || true; echo

build-frontend:
	 cd frontend && npm ci && npm run build

seed-qdrant:
	docker compose exec -T face-pipeline python -c 'from qdrant_client import QdrantClient; from qdrant_client.http.models import PointStruct; import os,random; qc=QdrantClient(url=os.getenv("QDRANT_URL","http://qdrant:6333")); coll=os.getenv("QDRANT_COLLECTION","faces_v1"); pts=[PointStruct(id=i, vector=[random.random() for _ in range(512)], payload={"tenant_id":"demo","p_hash_prefix":"0000"}) for i in range(1000)]; qc.upsert(coll, points=pts, wait=True); print(f"seeded {len(pts)} points")'

worker:
	docker compose exec -e ENABLE_QUEUE_WORKER=true face-pipeline python worker.py

worker-once:
	docker compose exec -e ENABLE_QUEUE_WORKER=true face-pipeline python worker.py --once

publish-test:
	docker compose exec face-pipeline python publish_test_message.py \
		--tenant demo \
		--bucket raw-images \
		--key samples/person4.jpg \
		--url file:///app/samples/person4.jpg \
		--file /app/samples/person4.jpg

clean:
	docker compose down -v

bash-backend:
	docker compose exec api bash || true

seed:
	docker compose exec api python scripts/seed_demo.py || true

crawl:
	@echo "Usage: make crawl URL=<url> [METHOD=<method>] [MIN_FACE_QUALITY=<score>] [REQUIRE_FACE=<true/false>] [CROP_FACES=<true/false>] [FACE_MARGIN=<margin>] [CRAWL_MODE=<single/site>] [MAX_TOTAL_IMAGES=<number>] [MAX_PAGES=<number>] [MAX_CONCURRENT_IMAGES=<number>] [BATCH_SIZE=<number>] [TENANT_ID=<tenant_id>] [USE_3X3_MINING=<true/false>]"
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
	USE_3X3_MINING_FLAG=""; \
	if [ "$${USE_3X3_MINING:-false}" = "true" ]; then USE_3X3_MINING_FLAG="--use-3x3-mining"; fi; \
	docker compose exec api python scripts/crawl_multisite.py --sites $$URL --method $$METHOD --min-face-quality $$MIN_FACE_QUALITY --face-margin $$FACE_MARGIN --max-images $$MAX_TOTAL_IMAGES --max-pages $$MAX_PAGES --mode $$CRAWL_MODE --max-concurrent-images $$MAX_CONCURRENT_IMAGES --batch-size $$BATCH_SIZE --tenant-id $$TENANT_ID $$REQUIRE_FACE_FLAG $$CROP_FACES_FLAG $$USE_3X3_MINING_FLAG

crawl-multisite:
	@echo "Usage: make crawl-multisite SITES=\"<site1,site2,site3>\" [SITES_FILE=<file>] [METHOD=<method>] [MAX_IMAGES_PER_SITE=<number>] [MAX_PAGES_PER_SITE=<number>] [CONCURRENT_SITES=<number>] [MIN_FACE_QUALITY=<score>] [REQUIRE_FACE=<true/false>] [CROP_FACES=<true/false>] [FACE_MARGIN=<margin>] [TENANT_ID=<tenant_id>] [USE_3X3_MINING=<true/false>] [VERBOSE=<true/false>] [QUIET=<true/false>]"
	@echo "Examples:"
	@echo "  make crawl-multisite SITES=\"https://wikifeet.com,https://candidteens.net,https://forum.candidgirls.io\""
	@echo "  make crawl-multisite SITES_FILE=sites.txt MAX_IMAGES_PER_SITE=30 CONCURRENT_SITES=2"
	@echo "  make crawl-multisite SITES=\"https://site1.com,https://site2.com\" REQUIRE_FACE=true CROP_FACES=true MIN_FACE_QUALITY=0.7"
	@echo "Exit codes: 0=success, 1=partial success with errors, 2=<50% success, 3=complete failure"
	@if [ -z "$(SITES)" ] && [ -z "$(SITES_FILE)" ]; then echo "Error: Either SITES or SITES_FILE is required"; exit 1; fi
	@METHOD=$${METHOD:-smart}; \
	MAX_IMAGES_PER_SITE=$${MAX_IMAGES_PER_SITE:-20}; \
	MAX_PAGES_PER_SITE=$${MAX_PAGES_PER_SITE:-5}; \
	CONCURRENT_SITES=$${CONCURRENT_SITES:-3}; \
	MIN_FACE_QUALITY=$${MIN_FACE_QUALITY:-0.5}; \
	FACE_MARGIN=$${FACE_MARGIN:-0.2}; \
	TENANT_ID=$${TENANT_ID:-multisite}; \
	REQUIRE_FACE_FLAG=""; \
	if [ "$${REQUIRE_FACE:-false}" = "true" ]; then REQUIRE_FACE_FLAG="--require-face"; fi; \
	CROP_FACES_FLAG=""; \
	if [ "$${CROP_FACES:-true}" = "true" ]; then CROP_FACES_FLAG="--crop-faces"; fi; \
	VERBOSE_FLAG=""; \
	if [ "$${VERBOSE:-false}" = "true" ]; then VERBOSE_FLAG="--verbose"; fi; \
	QUIET_FLAG=""; \
	if [ "$${QUIET:-false}" = "true" ]; then QUIET_FLAG="--quiet"; fi; \
	USE_3X3_MINING_FLAG=""; \
	if [ "$${USE_3X3_MINING:-false}" = "true" ]; then USE_3X3_MINING_FLAG="--use-3x3-mining"; fi; \
	docker compose exec api python scripts/crawl_multisite.py \
		$$(if [ -n "$$SITES" ]; then echo "--sites $$SITES"; fi) \
		$$(if [ -n "$$SITES_FILE" ]; then echo "--sites-file /app/$$SITES_FILE"; fi) \
		--method $$METHOD \
		--max-images-per-site $$MAX_IMAGES_PER_SITE \
		--max-pages-per-site $$MAX_PAGES_PER_SITE \
		--concurrent-sites $$CONCURRENT_SITES \
		--min-face-quality $$MIN_FACE_QUALITY \
		--face-margin $$FACE_MARGIN \
		--tenant-id $$TENANT_ID \
		$$REQUIRE_FACE_FLAG \
		$$CROP_FACES_FLAG \
		$$USE_3X3_MINING_FLAG \
		$$VERBOSE_FLAG \
		$$QUIET_FLAG; \
	EXIT_CODE=$$?; \
	if [ $$EXIT_CODE -eq 0 ]; then \
		echo "✅ Crawl completed successfully"; \
	elif [ $$EXIT_CODE -eq 1 ]; then \
		echo "⚠️  Crawl completed with some errors (non-critical)"; \
	elif [ $$EXIT_CODE -eq 2 ]; then \
		echo "⚠️  Warning: Less than 50% of sites successful"; \
	else \
		echo "❌ Error: Crawl failed"; \
	fi; \
	exit $$EXIT_CODE

reset-minio:
	@echo "Clearing MinIO buckets..."
	docker compose exec api python scripts/clear_minio.py

clean-downloads:
	@echo "Removing flat and zips directories..."
	rm -rf flat/ zips/
	@echo "Flat and zips directories removed."

test-integration:
	@echo "Running integration tests..."
	docker compose exec api python -m pytest tests/test_crawler_integration.py -v

test-quick:
	@echo "Running quick integration tests (excluding slow tests)..."
	docker compose exec api python -m pytest tests/test_crawler_integration.py -v -m "not slow"

download-thumb:
	@./scripts/download_images.sh thumbnails

download-raw:
	@./scripts/download_images.sh raw-images

download-both:
	@./scripts/download_images.sh both