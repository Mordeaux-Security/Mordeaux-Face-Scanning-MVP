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
	@echo "Usage: make crawl URL=<url> [METHOD=<method>] [MIN_FACE_QUALITY=<score>] [REQUIRE_FACE=<true/false>] [CROP_FACES=<true/false>] [FACE_MARGIN=<margin>] [CRAWL_MODE=<single/site>] [MAX_TOTAL_IMAGES=<number>] [MAX_PAGES=<number>] [SAVE_BOTH=<true/false>]"
	@echo "Example: make crawl URL=https://example.com METHOD=smart MIN_FACE_QUALITY=0.7 CROP_FACES=true FACE_MARGIN=0.2 CRAWL_MODE=site MAX_TOTAL_IMAGES=50 MAX_PAGES=20 SAVE_BOTH=true"
	@if [ -z "$(URL)" ]; then echo "Error: URL is required"; exit 1; fi
	@METHOD=$${METHOD:-smart}; \
	MIN_FACE_QUALITY=$${MIN_FACE_QUALITY:-0.5}; \
	FACE_MARGIN=$${FACE_MARGIN:-0.2}; \
	MAX_TOTAL_IMAGES=$${MAX_TOTAL_IMAGES:-50}; \
	MAX_PAGES=$${MAX_PAGES:-20}; \
	CRAWL_MODE=$${CRAWL_MODE:-single}; \
	SAVE_BOTH=$${SAVE_BOTH:-false}; \
	REQUIRE_FACE_FLAG=""; \
	if [ "$${REQUIRE_FACE:-true}" = "false" ]; then REQUIRE_FACE_FLAG="--no-require-face"; fi; \
	CROP_FACES_FLAG=""; \
	if [ "$${CROP_FACES:-true}" = "false" ]; then CROP_FACES_FLAG="--no-crop-faces"; fi; \
	SAVE_BOTH_FLAG=""; \
	if [ "$$SAVE_BOTH" = "true" ]; then SAVE_BOTH_FLAG="--save-both"; fi; \
	docker compose exec backend-cpu python scripts/crawl_images.py $(URL) --method $$METHOD --min-face-quality $$MIN_FACE_QUALITY --face-margin $$FACE_MARGIN --max-total-images $$MAX_TOTAL_IMAGES --max-pages $$MAX_PAGES --crawl-mode $$CRAWL_MODE $$REQUIRE_FACE_FLAG $$CROP_FACES_FLAG $$SAVE_BOTH_FLAG
restart:
	docker compose restart
